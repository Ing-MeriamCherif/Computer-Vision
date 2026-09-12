"""Offline-ready Depth Anything V2 provider for webcam/demo input."""

from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path

import numpy as np

from .state import DepthState


@dataclass(frozen=True, slots=True)
class DepthInferenceDiagnostics:
    device: str
    inference_ms: float
    min_depth: float
    max_depth: float
    peak_vram_mb: float | None


class DepthAnythingProvider:
    """Lazy local-checkpoint monocular relative-depth inference."""

    def __init__(self, model_path: str | Path = "models/depth-anything-v2-small", *, device: str = "auto", use_fp16: bool = False, input_size: int | tuple[int, int] | None = None) -> None:
        self.model_path = str(model_path)
        self.requested_device = device
        self.use_fp16 = bool(use_fp16)
        if input_size is not None:
            if isinstance(input_size, int):
                input_size = (input_size, input_size)
            if len(input_size) != 2 or min(input_size) < 14:
                raise ValueError("input_size must be an integer or a (height, width) pair >= 14")
            self.input_size = (int(input_size[0]), int(input_size[1]))
        else:
            self.input_size = None
        self.processor = None
        self.model = None
        self.device = None
        self.last_diagnostics: DepthInferenceDiagnostics | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise RuntimeError("Torch, Torchvision, and Transformers are required for monocular depth") from exc
        device = self.requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA depth inference requested but CUDA is unavailable")
        path = Path(self.model_path)
        if not path.exists():
            raise RuntimeError(f"depth model not found at {path}; run tools/setup_gpu_ui.sh")
        self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
        if self.input_size is not None:
            self.processor.size = {"height": self.input_size[0], "width": self.input_size[1]}
        self.model = AutoModelForDepthEstimation.from_pretrained(path, local_files_only=True).to(device).eval()
        if device.startswith("cuda") and self.use_fp16:
            self.model = self.model.half()
        self.device = torch.device(device)

    def compute(self, rgb_frame: np.ndarray, source_frame_id: int | str, timestamp: float) -> DepthState:
        self.load()
        import torch
        import torch.nn.functional as functional
        from PIL import Image
        frame = np.asarray(rgb_frame)
        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise ValueError("rgb_frame must have shape (H, W, 3/4)")
        frame = np.clip(frame[..., :3], 0, 255).astype(np.uint8)
        inputs = self.processor(images=Image.fromarray(frame), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if self.device.type == "cuda" and self.use_fp16:
            inputs = {key: value.half() if value.is_floating_point() else value for key, value in inputs.items()}
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model(**inputs).predicted_depth
            resized = functional.interpolate(output.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False).squeeze(1)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = (time.perf_counter() - start) * 1000.0
        depth = resized.float().detach().cpu().numpy()
        finite = np.isfinite(depth) & (depth > 1e-6)
        if not finite.any():
            raise RuntimeError("depth model produced no finite positive depth")
        # Stable session-relative units: normalize robust median to 2.0.
        depth = (depth * (2.0 / max(float(np.median(depth[finite])), 1e-6))).astype(np.float32)
        confidence = np.where(finite, 1.0, 0.0).astype(np.float32)
        peak = float(torch.cuda.max_memory_allocated(self.device) / 1048576.0) if self.device.type == "cuda" else None
        self.last_diagnostics = DepthInferenceDiagnostics(str(self.device), elapsed, float(depth[finite].min()), float(depth[finite].max()), peak)
        return DepthState(depth, timestamp, source_frame_id, "relative", valid_mask=finite, confidence=confidence)
