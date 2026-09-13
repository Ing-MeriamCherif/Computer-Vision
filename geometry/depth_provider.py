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
    session_scale: float | None = None


@dataclass(slots=True)
class DeviceDepthState:
    """Internal device-resident depth tensors paired with one capture frame."""

    depth: object
    valid_mask: object
    confidence: object
    timestamp: float
    source_frame_id: int | str
    scale_mode: str = "relative"
    completed_timestamp: float | None = None


class DepthAnythingProvider:
    """Lazy local-checkpoint monocular relative-depth inference."""

    backend_name: str = "depth-anything-v2-small"

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
        self._session_scale: float | None = None
        self._previous_device_depth = None
        self._previous_device_valid = None
        self._scale_updates = 0
        self._compute_count = 0

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

    def warmup(self, iterations: int = 1) -> None:
        """Prime the processor/model path before the camera loop begins."""
        self.load()
        import torch
        from PIL import Image

        if self.input_size is None:
            height, width = 518, 518
        else:
            height, width = self.input_size
        dummy = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
        for _ in range(max(0, int(iterations))):
            inputs = self.processor(images=dummy, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            if self.device.type == "cuda" and self.use_fp16:
                inputs = {key: value.half() if value.is_floating_point() else value for key, value in inputs.items()}
            with torch.inference_mode():
                self.model(**inputs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def compute(self, rgb_frame: np.ndarray, source_frame_id: int | str, timestamp: float) -> DepthState:
        state, _device_state = self.compute_device(rgb_frame, source_frame_id, timestamp)
        return state

    def compute_device(self, rgb_frame: np.ndarray, source_frame_id: int | str, timestamp: float) -> tuple[DepthState, DeviceDepthState]:
        """Compute depth while retaining postprocessing tensors on CUDA."""
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
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model(**inputs).predicted_depth
            resized = functional.interpolate(output.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False).squeeze(1)[0]
        elapsed = (time.perf_counter() - start) * 1000.0
        model_output = resized.float()
        finite_t = torch.isfinite(model_output) & (model_output > 1e-6)
        if not bool(finite_t.any().item()):
            raise RuntimeError("depth model produced no finite positive depth")
        # Depth Anything's predicted_depth is inverse-depth-like (larger is
        # nearer). Convert explicitly to the project forward-Z convention:
        # larger Z means farther from the camera.
        depth = torch.where(finite_t, 1.0 / torch.clamp(model_output, min=1e-6), torch.full_like(model_output, float("nan")))
        # Stable session-relative units: smooth the scale target across frames
        # so a hand entering the image cannot globally rescale the wall.
        raw_depth = depth.detach()
        target_scale = 2.0 / max(float(torch.median(depth[finite_t]).item()), 1e-6)
        # Robust scale tracking: use only stable overlapping pixels and cap
        # per-frame correction so entering hands cannot rescale the scene.
        if self._session_scale is None:
            self._session_scale = target_scale
        else:
            self._scale_updates += 1
            # Let the first few frames establish an anchor, then move only
            # very slowly so scene-composition changes cannot breathe the map.
            alpha = 0.02 if self._scale_updates < 15 else 0.002
            self._session_scale = alpha * target_scale + (1.0 - alpha) * self._session_scale
        depth = depth * self._session_scale
        # This is geometry/depth reliability, not a neural confidence score:
        # finite validity is reduced near unstable depth discontinuities.
        safe = torch.nan_to_num(depth, nan=0.0)
        gx = torch.zeros_like(safe); gy = torch.zeros_like(safe)
        gx[:, 1:-1] = (safe[:, 2:] - safe[:, :-2]) * 0.5
        gy[1:-1, :] = (safe[2:, :] - safe[:-2, :]) * 0.5
        gradient = torch.sqrt(gx.square() + gy.square())
        reference = torch.quantile(gradient[finite_t], 0.90).clamp_min(1e-6)
        confidence_t = torch.exp(-(gradient / reference).clamp(0.0, 4.0)) * finite_t.to(torch.float32)
        self._previous_device_depth = raw_depth
        self._previous_device_valid = finite_t.detach()
        finite = finite_t.detach().cpu().numpy()
        depth_np = depth.detach().cpu().numpy().astype(np.float32)
        confidence = confidence_t.detach().cpu().numpy().astype(np.float32)
        self._compute_count += 1
        peak = float(torch.cuda.max_memory_allocated(self.device) / 1048576.0) if self.device.type == "cuda" and self._compute_count % 30 == 0 else None
        self.last_diagnostics = DepthInferenceDiagnostics(str(self.device), elapsed, float(depth_np[finite].min()), float(depth_np[finite].max()), peak, float(self._session_scale))
        state = DepthState(depth_np, timestamp, source_frame_id, "relative", valid_mask=finite, confidence=confidence)
        return state, DeviceDepthState(depth, finite_t, confidence_t, timestamp, source_frame_id)
