"""Opt-in adapter for the colleague ``feature/depth`` implementation.

The upstream package is intentionally preserved under ``integrations/``. This
adapter loads its original ``DepthModel`` entry point and maps its
``DepthState`` into our shared state contract, including source frame IDs.
Set ``NRW_DEPTH_SOURCE=colleague`` to use it; the tested local-checkpoint
provider remains the default for offline and low-latency runs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from .depth_provider import DepthInferenceDiagnostics
from .state import DepthState


class ColleagueDepthProvider:
    backend_name = "colleague-depth-anything-v2-small"

    def __init__(self, *, device: str = "auto", input_size: int = 518, fp16: bool = True) -> None:
        root = Path(__file__).resolve().parents[1] / "integrations" / "colleague_depth"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from depth.model import DepthModel  # the preserved upstream module

        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self._model = DepthModel(device=device, input_size=input_size, fp16=fp16)
        self.last_diagnostics: DepthInferenceDiagnostics | None = None

    def compute(self, rgb_frame: np.ndarray, source_frame_id: int | str, timestamp: float) -> DepthState:
        import cv2

        frame = np.asarray(rgb_frame, dtype=np.uint8)[..., :3]
        t0 = time.perf_counter()
        upstream = self._model.infer(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        elapsed = (time.perf_counter() - t0) * 1000.0

        depth_arr = np.asarray(upstream.depth_map, dtype=np.float32)
        valid = upstream.valid_mask if upstream.valid_mask is not None else np.isfinite(depth_arr)
        min_d = float(np.min(depth_arr[valid])) if valid.any() else float(np.min(depth_arr))
        max_d = float(np.max(depth_arr[valid])) if valid.any() else float(np.max(depth_arr))
        peak_vram = None
        try:
            import torch
            if torch.cuda.is_available():
                peak_vram = float(torch.cuda.max_memory_allocated() / 1048576.0)
        except Exception:
            pass

        self.last_diagnostics = DepthInferenceDiagnostics(
            str(getattr(self._model, "device", "unknown")),
            elapsed,
            min_d,
            max_d,
            peak_vram,
        )

        return DepthState(
            depth_arr,
            timestamp if timestamp is not None else time.time(),
            source_frame_id,
            upstream.scale_mode,
            valid_mask=valid,
            confidence=np.asarray(valid, dtype=np.float32),
        )

