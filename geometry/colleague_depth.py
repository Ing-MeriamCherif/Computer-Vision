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

    def compute(self, rgb_frame: np.ndarray, source_frame_id: int | str, timestamp: float) -> DepthState:
        import cv2

        frame = np.asarray(rgb_frame, dtype=np.uint8)[..., :3]
        upstream = self._model.infer(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return DepthState(
            np.asarray(upstream.depth_map, dtype=np.float32),
            timestamp if timestamp is not None else time.time(),
            source_frame_id,
            upstream.scale_mode,
            valid_mask=upstream.valid_mask,
            confidence=np.asarray(upstream.valid_mask, dtype=np.float32) if upstream.valid_mask is not None else None,
        )

