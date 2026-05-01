"""
xDiT HTTP Service — Multi-model inference server with Ray + FastAPI.

Supports both image and video diffusion models with distributed inference.

Usage:
    python entrypoints/service.py \
        --model_path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
        --world_size 3 \
        --fully_shard_degree 3
"""

import os
import io
import sys
import time
import base64
import logging
import tempfile
import argparse
from typing import Optional, List, Dict, Any

import torch
import ray
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    guidance_scale_2: Optional[float] = None
    max_sequence_length: Optional[int] = None
    num_frames: Optional[int] = None
    flow_shift: Optional[float] = None
    seed: Optional[int] = 42
    output_format: Optional[str] = Field("auto", pattern="^(auto|png|jpeg|webp|mp4)$")
    save_disk_path: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    world_size: int
    output_type: str
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Ray worker — one per GPU
# ---------------------------------------------------------------------------

@ray.remote(num_gpus=1)
class InferenceWorker:
    """Each worker owns one GPU and its share of the distributed pipeline."""

    def __init__(self, xfuser_args, rank: int, world_size: int):
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        self.rank = rank
        self.logger = logging.getLogger(f"xdit.worker.{rank}")
        logging.basicConfig(level=logging.INFO,
                            format=f"%(asctime)s [W{rank}] %(levelname)s %(message)s")

        self._init_pipeline(xfuser_args)

    def _init_pipeline(self, xfuser_args):
        from xfuser.config import xFuserArgs
        from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY
        import xfuser.model_executor.models.runner_models

        self.xfuser_args = xfuser_args
        engine_config, input_config = xfuser_args.create_config()
        self.engine_config = engine_config
        self.input_config = input_config

        model_name = xfuser_args.model
        model_cls = MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            for key in MODEL_REGISTRY:
                if key.endswith(model_name) or model_name.endswith(key.split("/")[-1]):
                    model_cls = MODEL_REGISTRY[key]
                    break
        if model_cls is None:
            raise ValueError(
                f"Model '{model_name}' not in registry. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        self.model = model_cls(xfuser_args)
        self.output_type = self.model.settings.model_output_type
        self.fps = getattr(self.model.settings, "fps", 16) or 16

        default_args = self._build_default_input_args()
        input_args = self.model.preprocess_args(default_args)
        self.model.initialize(input_args)
        self.logger.info("Pipeline ready on GPU %d (output_type=%s)", self.rank, self.output_type)

    def _build_default_input_args(self) -> dict:
        dv = self.model.default_input_values
        args = {
            "prompt": "warmup",
            "negative_prompt": getattr(dv, "negative_prompt", None) or "",
            "height": dv.height or 1024,
            "width": dv.width or 1024,
            "num_inference_steps": dv.num_inference_steps or 28,
            "guidance_scale": dv.guidance_scale if dv.guidance_scale is not None else 3.5,
            "max_sequence_length": getattr(dv, "max_sequence_length", None) or 512,
            "seed": 42,
            "input_images": [],
            "output_directory": "/tmp/xdit_output",
        }
        if getattr(dv, "num_frames", None):
            args["num_frames"] = dv.num_frames
        if getattr(dv, "flow_shift", None) is not None:
            args["flow_shift"] = dv.flow_shift
        if getattr(dv, "guidance_scale_2", None) is not None:
            args["guidance_scale_2"] = dv.guidance_scale_2
        return args

    def generate(self, req: dict) -> Optional[dict]:
        from xfuser.model_executor.models.runner_models.base_model import DiffusionOutput

        dv = self.model.default_input_values
        input_args = {
            "prompt": req["prompt"],
            "negative_prompt": req.get("negative_prompt") or getattr(dv, "negative_prompt", None) or "",
            "height": req.get("height") or dv.height or 1024,
            "width": req.get("width") or dv.width or 1024,
            "num_inference_steps": req.get("num_inference_steps") or dv.num_inference_steps or 28,
            "guidance_scale": req.get("guidance_scale") if req.get("guidance_scale") is not None else (dv.guidance_scale if dv.guidance_scale is not None else 3.5),
            "max_sequence_length": req.get("max_sequence_length") or getattr(dv, "max_sequence_length", None) or 512,
            "seed": req.get("seed", 42),
            "input_images": [],
            "output_directory": "/tmp/xdit_output",
        }

        if getattr(dv, "num_frames", None):
            input_args["num_frames"] = req.get("num_frames") or dv.num_frames
        if hasattr(dv, "flow_shift"):
            input_args["flow_shift"] = req.get("flow_shift") if req.get("flow_shift") is not None else dv.flow_shift
        if hasattr(dv, "guidance_scale_2"):
            input_args["guidance_scale_2"] = req.get("guidance_scale_2") if req.get("guidance_scale_2") is not None else dv.guidance_scale_2

        print(
            f"[XDIT] Generate: {input_args.get('width', 0)}x{input_args.get('height', 0)}, "
            f"frames={input_args.get('num_frames', 'N/A')}, steps={input_args.get('num_inference_steps', '?')}, "
            f"guidance={input_args.get('guidance_scale', 0)}, seed={input_args.get('seed', '?')}, "
            f"prompt={input_args.get('prompt', '')[:80]}",
            flush=True,
        )

        start = time.time()
        output, timings = self.model.run(input_args)
        elapsed = time.time() - start

        from xfuser.core.utils.runner_utils import is_last_process
        if not is_last_process():
            return None

        if self.output_type == "video":
            return self._encode_video_output(output, elapsed, timings, req)
        else:
            return self._encode_image_output(output, elapsed, timings, req)

    def _encode_image_output(self, output, elapsed, timings, req) -> Optional[dict]:
        images = output.images or []
        if not images:
            return None

        fmt = (req.get("output_format") or "png").upper()
        if fmt == "AUTO":
            fmt = "PNG"
        mime_map = {"JPEG": "image/jpeg", "WEBP": "image/webp", "PNG": "image/png"}
        mime = mime_map.get(fmt, "image/png")
        if fmt not in mime_map:
            fmt = "PNG"

        results = []
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.save(buf, format=fmt)
            results.append({
                "base64": base64.b64encode(buf.getvalue()).decode(),
                "mime": mime,
            })

        return {
            "type": "image",
            "images": results,
            "elapsed_seconds": round(elapsed, 3),
            "timings": timings,
        }

    def _encode_video_output(self, output, elapsed, timings, req) -> Optional[dict]:
        videos = output.videos or []
        videos = [v for v in videos if v is not None]
        if not videos:
            return None

        import numpy as np

        results = []
        for i, video_frames in enumerate(videos):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                self._export_h264(video_frames, tmp_path, fps=self.fps)
                with open(tmp_path, "rb") as f:
                    video_bytes = f.read()
                results.append({
                    "base64": base64.b64encode(video_bytes).decode(),
                    "mime": "video/mp4",
                    "fps": self.fps,
                    "num_frames": len(video_frames) if isinstance(video_frames, list) else None,
                })
            finally:
                os.unlink(tmp_path)

        return {
            "type": "video",
            "videos": results,
            "elapsed_seconds": round(elapsed, 3),
            "timings": timings,
        }

    @staticmethod
    def _export_h264(frames, output_path: str, fps: float = 16.0):
        """Export frames to H.264 MP4 using imageio-ffmpeg (browser-compatible)."""
        import numpy as np
        try:
            import imageio_ffmpeg

            frame_list = []
            for frame in frames:
                if hasattr(frame, "numpy"):
                    frame = frame.numpy()
                if isinstance(frame, np.ndarray):
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).clip(0, 255).astype(np.uint8)
                else:
                    frame = np.array(frame, dtype=np.uint8)
                frame_list.append(frame)

            h, w = frame_list[0].shape[:2]
            writer = imageio_ffmpeg.write_frames(
                output_path,
                size=(w, h),
                fps=fps,
                codec="libx264",
                output_params=["-crf", "23", "-preset", "fast",
                               "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
            writer.send(None)
            for frame in frame_list:
                writer.send(np.ascontiguousarray(frame))
            writer.close()
        except ImportError:
            from diffusers.utils import export_to_video
            export_to_video(frames, output_path, fps=int(fps))

    def ping(self) -> str:
        return f"worker-{self.rank}-ok"

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model.settings.model_name,
            "output_type": self.model.settings.model_output_type,
            "fps": getattr(self.model.settings, "fps", None),
            "capabilities": {
                k: getattr(self.model.capabilities, k)
                for k in self.model.capabilities.__dataclass_fields__
            },
            "defaults": {
                k: getattr(self.model.default_input_values, k)
                for k in self.model.default_input_values.__dataclass_fields__
                if getattr(self.model.default_input_values, k) is not None
            },
        }


# ---------------------------------------------------------------------------
# Engine — orchestrates Ray workers
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, world_size: int, xfuser_args):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self.world_size = world_size
        self.workers = [
            InferenceWorker.remote(xfuser_args, rank=r, world_size=world_size)
            for r in range(world_size)
        ]
        results = ray.get([w.ping.remote() for w in self.workers])
        logging.info("All workers ready: %s", results)

    async def generate(self, req: dict) -> dict:
        futures = [w.generate.remote(req) for w in self.workers]
        results = ray.get(futures)
        for r in results:
            if r is not None:
                return r
        raise RuntimeError("No worker produced output")

    def get_model_info(self) -> dict:
        return ray.get(self.workers[0].get_model_info.remote())


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

START_TIME = time.time()
app = FastAPI(
    title="xDiT Inference Service",
    description="Multi-GPU parallel inference for Diffusion Transformers (image & video)",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Engine = None
_launch_args: dict = {}


@app.get("/health", response_model=HealthResponse)
async def health():
    info = engine.get_model_info()
    return HealthResponse(
        status="ok",
        model=_launch_args.get("model_path", "unknown"),
        world_size=_launch_args.get("world_size", 0),
        output_type=info.get("output_type", "image"),
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.get("/models")
async def list_models():
    info = engine.get_model_info()
    return {
        "loaded": info,
        "world_size": engine.world_size,
        "parallel_config": {
            "ulysses_degree": _launch_args.get("ulysses_degree", 1),
            "ring_degree": _launch_args.get("ring_degree", 1),
            "pipefusion_degree": _launch_args.get("pipefusion_degree", 1),
            "fully_shard_degree": _launch_args.get("fully_shard_degree", 1),
            "cfg_parallel": _launch_args.get("use_cfg_parallel", False),
        },
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(400, "prompt is required")

    try:
        result = await engine.generate(request.model_dump())
        return result
    except Exception as e:
        logging.exception("Generation failed")
        raise HTTPException(500, str(e))


@app.post("/generate/raw")
async def generate_raw(request: GenerateRequest):
    """Return the first output as raw bytes."""
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(400, "prompt is required")

    req = request.model_dump()
    req["save_disk_path"] = None
    result = await engine.generate(req)

    if result.get("type") == "video":
        video_data = result["videos"][0]
        raw = base64.b64decode(video_data["base64"])
        return Response(content=raw, media_type="video/mp4")
    else:
        img_data = result["images"][0]
        raw = base64.b64decode(img_data["base64"])
        return Response(content=raw, media_type=img_data["mime"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="xDiT HTTP Inference Service")
    p.add_argument("--model_path", type=str, required=True,
                   help="HuggingFace model ID or local path")
    p.add_argument("--world_size", type=int, default=4,
                   help="Number of GPUs to use")
    p.add_argument("--ulysses_degree", type=int, default=1)
    p.add_argument("--ring_degree", type=int, default=1)
    p.add_argument("--pipefusion_degree", type=int, default=1)
    p.add_argument("--fully_shard_degree", type=int, default=1)
    p.add_argument("--use_cfg_parallel", action="store_true")
    p.add_argument("--attention_backend", type=str, default=None,
                   help="Attention backend: SDPA, SDPA_FLASH, SDPA_EFFICIENT, CUDNN, etc.")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=6000)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [SERVICE] %(levelname)s %(message)s")

    args = parse_args()
    _launch_args = {
        "model_path": args.model_path,
        "world_size": args.world_size,
        "ulysses_degree": args.ulysses_degree,
        "ring_degree": args.ring_degree,
        "pipefusion_degree": args.pipefusion_degree,
        "fully_shard_degree": args.fully_shard_degree,
        "use_cfg_parallel": args.use_cfg_parallel,
    }

    from xfuser.config import xFuserArgs

    xfuser_kwargs = dict(
        model=args.model_path,
        trust_remote_code=True,
        warmup_steps=0,
        use_parallel_vae=False,
        use_torch_compile=False,
        use_ray=True,
        ray_world_size=args.world_size,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        pipefusion_parallel_degree=args.pipefusion_degree,
        fully_shard_degree=args.fully_shard_degree,
        use_cfg_parallel=args.use_cfg_parallel,
    )
    if args.attention_backend:
        xfuser_kwargs["attention_backend"] = args.attention_backend

    xfuser_args = xFuserArgs(**xfuser_kwargs)

    logging.info(
        "Starting xDiT service: model=%s gpus=%d ulysses=%d ring=%d pp=%d fsdp=%d cfg=%s",
        args.model_path, args.world_size, args.ulysses_degree,
        args.ring_degree, args.pipefusion_degree, args.fully_shard_degree,
        args.use_cfg_parallel,
    )

    engine = Engine(world_size=args.world_size, xfuser_args=xfuser_args)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
