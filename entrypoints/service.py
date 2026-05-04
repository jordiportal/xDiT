"""
xDiT HTTP Service — On-demand multi-model inference server with Ray + FastAPI.

Models are loaded into GPU only when a request arrives and unloaded after
an inactivity timeout (default 10 min). Supports T2V and I2V with automatic
model switching.

Usage:
    python entrypoints/service.py \
        --t2v_model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
        --i2v_model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
        --world_size 4 --ulysses_degree 4 --fully_shard_degree 2
"""

import os
import io
import sys
import time
import base64
import asyncio
import logging
import tempfile
import argparse
import threading
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
    image: Optional[str] = None  # base64 image for I2V
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


class HealthResponse(BaseModel):
    status: str
    loaded_model: Optional[str] = None
    available_models: Dict[str, str]
    world_size: int
    uptime_seconds: float
    idle_timeout_seconds: int


# ---------------------------------------------------------------------------
# Ray worker — one per GPU
# ---------------------------------------------------------------------------

@ray.remote(num_gpus=1)
class InferenceWorker:
    """Each worker owns one GPU and its share of the distributed pipeline."""

    def __init__(self, xfuser_args, rank: int, world_size: int, env_vars: dict = None):
        if env_vars:
            for k, v in env_vars.items():
                os.environ[k] = v

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
        h = dv.height or 1024
        w = dv.width or 1024

        input_images = []
        is_i2v = getattr(self.model.capabilities, "require_input_image", False) or \
                 "i2v" in self.xfuser_args.model.lower()
        if is_i2v:
            from PIL import Image as PILImage
            placeholder = PILImage.new("RGB", (w, h), color=(128, 128, 128))
            input_images = [placeholder]

        args = {
            "prompt": "warmup",
            "negative_prompt": getattr(dv, "negative_prompt", None) or "",
            "height": h,
            "width": w,
            "num_inference_steps": dv.num_inference_steps or 28,
            "guidance_scale": dv.guidance_scale if dv.guidance_scale is not None else 3.5,
            "max_sequence_length": getattr(dv, "max_sequence_length", None) or 512,
            "seed": 42,
            "input_images": input_images,
            "output_directory": "/tmp/xdit_output",
        }
        if is_i2v:
            args["image"] = input_images[0]
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

        if req.get("image"):
            from PIL import Image as PILImage
            img_data = base64.b64decode(req["image"])
            pil_img = PILImage.open(io.BytesIO(img_data)).convert("RGB")
            input_args["input_images"] = [pil_img]
            input_args["image"] = pil_img

        if getattr(dv, "num_frames", None):
            input_args["num_frames"] = req.get("num_frames") or dv.num_frames
        if hasattr(dv, "flow_shift"):
            input_args["flow_shift"] = req.get("flow_shift") if req.get("flow_shift") is not None else dv.flow_shift
        if hasattr(dv, "guidance_scale_2"):
            input_args["guidance_scale_2"] = req.get("guidance_scale_2") if req.get("guidance_scale_2") is not None else dv.guidance_scale_2

        mode = "I2V" if input_args["input_images"] else "T2V"
        print(
            f"[XDIT] Generate ({mode}): {input_args.get('width', 0)}x{input_args.get('height', 0)}, "
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
        }


# ---------------------------------------------------------------------------
# Engine — lazy loading with inactivity timeout
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, world_size: int, base_xfuser_kwargs: dict,
                 t2v_model: str, i2v_model: Optional[str] = None,
                 idle_timeout: int = 600):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self.world_size = world_size
        self.base_xfuser_kwargs = base_xfuser_kwargs
        self.t2v_model = t2v_model
        self.i2v_model = i2v_model
        self.idle_timeout = idle_timeout

        self.workers: List = []
        self.loaded_model: Optional[str] = None
        self._lock = asyncio.Lock()
        self._unload_task: Optional[asyncio.Task] = None

        logging.info(
            "Engine initialized (lazy). T2V=%s, I2V=%s, timeout=%ds",
            t2v_model, i2v_model or "not configured", idle_timeout,
        )

    def _determine_model(self, req: dict) -> str:
        if req.get("image"):
            if not self.i2v_model:
                raise ValueError("I2V model not configured but request contains image")
            return self.i2v_model
        return self.t2v_model

    async def _load_model(self, model_path: str):
        """Load model into GPU workers."""
        if self.loaded_model == model_path:
            return

        if self.workers:
            await self._unload_workers()

        logging.info("Loading model: %s (world_size=%d)...", model_path, self.world_size)
        start = time.time()

        from xfuser.config import xFuserArgs
        kwargs = {**self.base_xfuser_kwargs, "model": model_path}
        xfuser_args = xFuserArgs(**kwargs)

        env_vars = {
            k: os.environ[k] for k in
            ("HF_TOKEN", "HF_HOME", "TRANSFORMERS_CACHE",
             "HF_HUB_DISABLE_XET", "HF_HUB_OFFLINE")
            if k in os.environ
        }

        self.workers = [
            InferenceWorker.remote(xfuser_args, rank=r, world_size=self.world_size,
                                   env_vars=env_vars)
            for r in range(self.world_size)
        ]
        results = ray.get([w.ping.remote() for w in self.workers])
        elapsed = time.time() - start
        self.loaded_model = model_path
        logging.info("Model loaded in %.1fs. Workers: %s", elapsed, results)

    async def _unload_workers(self):
        """Kill all workers and free GPU memory."""
        if not self.workers:
            return
        logging.info("Unloading model: %s", self.loaded_model)
        for w in self.workers:
            ray.kill(w)
        self.workers = []
        self.loaded_model = None
        logging.info("GPUs released.")

    async def _schedule_unload(self):
        """Schedule model unload after idle timeout."""
        if self._unload_task and not self._unload_task.done():
            self._unload_task.cancel()
        self._unload_task = asyncio.create_task(self._idle_unload_timer())

    async def _idle_unload_timer(self):
        try:
            await asyncio.sleep(self.idle_timeout)
            async with self._lock:
                if self.workers:
                    logging.info("Idle timeout (%ds) reached. Unloading model.", self.idle_timeout)
                    await self._unload_workers()
        except asyncio.CancelledError:
            pass

    async def generate(self, req: dict) -> dict:
        async with self._lock:
            needed_model = self._determine_model(req)

            if self._unload_task and not self._unload_task.done():
                self._unload_task.cancel()

            await self._load_model(needed_model)

        futures = [w.generate.remote(req) for w in self.workers]
        results = await asyncio.get_event_loop().run_in_executor(
            None, ray.get, futures
        )

        await self._schedule_unload()

        for r in results:
            if r is not None:
                return r
        raise RuntimeError("No worker produced output")

    def get_status(self) -> dict:
        return {
            "loaded_model": self.loaded_model,
            "workers_active": len(self.workers),
            "t2v_model": self.t2v_model,
            "i2v_model": self.i2v_model,
        }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

START_TIME = time.time()
app = FastAPI(
    title="xDiT Inference Service",
    description="On-demand multi-GPU parallel inference for Diffusion Transformers",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Engine = None
_launch_args: dict = {}


@app.get("/health")
async def health():
    status = engine.get_status()
    models = {"t2v": engine.t2v_model}
    if engine.i2v_model:
        models["i2v"] = engine.i2v_model
    return {
        "status": "ok",
        "loaded_model": status["loaded_model"],
        "available_models": models,
        "world_size": engine.world_size,
        "workers_active": status["workers_active"],
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "idle_timeout_seconds": engine.idle_timeout,
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(400, "prompt is required")

    try:
        result = await engine.generate(request.model_dump())
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logging.exception("Generation failed")
        raise HTTPException(500, str(e))


@app.post("/generate/raw")
async def generate_raw(request: GenerateRequest):
    """Return the first output as raw bytes."""
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(400, "prompt is required")

    result = await engine.generate(request.model_dump())

    if result.get("type") == "video":
        video_data = result["videos"][0]
        raw = base64.b64decode(video_data["base64"])
        return Response(content=raw, media_type="video/mp4")
    else:
        img_data = result["images"][0]
        raw = base64.b64decode(img_data["base64"])
        return Response(content=raw, media_type=img_data["mime"])


@app.post("/unload")
async def unload_model():
    """Manually unload model from GPUs."""
    async with engine._lock:
        if engine._unload_task and not engine._unload_task.done():
            engine._unload_task.cancel()
        await engine._unload_workers()
    return {"status": "unloaded"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="xDiT On-Demand HTTP Inference Service")
    p.add_argument("--t2v_model", type=str, default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                   help="HuggingFace model ID for Text-to-Video")
    p.add_argument("--i2v_model", type=str, default="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                   help="HuggingFace model ID for Image-to-Video")
    p.add_argument("--model_path", type=str, default=None,
                   help="(Deprecated) Alias for --t2v_model")
    p.add_argument("--world_size", type=int, default=4)
    p.add_argument("--ulysses_degree", type=int, default=1)
    p.add_argument("--ring_degree", type=int, default=1)
    p.add_argument("--pipefusion_degree", type=int, default=1)
    p.add_argument("--fully_shard_degree", type=int, default=1)
    p.add_argument("--use_cfg_parallel", action="store_true")
    p.add_argument("--attention_backend", type=str, default=None)
    p.add_argument("--idle_timeout", type=int, default=600,
                   help="Seconds of inactivity before unloading model from GPUs (default: 600)")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=6000)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [SERVICE] %(levelname)s %(message)s")

    args = parse_args()

    t2v_model = args.model_path or args.t2v_model
    i2v_model = args.i2v_model

    base_xfuser_kwargs = dict(
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
        base_xfuser_kwargs["attention_backend"] = args.attention_backend

    _launch_args = {
        "t2v_model": t2v_model,
        "i2v_model": i2v_model,
        "world_size": args.world_size,
        "ulysses_degree": args.ulysses_degree,
        "fully_shard_degree": args.fully_shard_degree,
        "idle_timeout": args.idle_timeout,
    }

    logging.info(
        "Starting xDiT on-demand service: T2V=%s, I2V=%s, GPUs=%d, "
        "ulysses=%d, fsdp=%d, idle_timeout=%ds",
        t2v_model, i2v_model, args.world_size,
        args.ulysses_degree, args.fully_shard_degree, args.idle_timeout,
    )

    engine = Engine(
        world_size=args.world_size,
        base_xfuser_kwargs=base_xfuser_kwargs,
        t2v_model=t2v_model,
        i2v_model=i2v_model,
        idle_timeout=args.idle_timeout,
    )

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
