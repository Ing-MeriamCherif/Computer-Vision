from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class DepthState:
    depth_map: np.ndarray          # (H, W) float32, relative or metric depth
    timestamp: float               # time when inference completed
    scale_mode: Literal["relative", "metric"] = "relative"
    valid_mask: np.ndarray | None = None  # (H, W) bool, False = unreliable pixels
    backend_name: str = ""
    inference_ms: float = 0.0


class DepthModel:
    """Model-agnostic depth inference wrapper.

    Usage:
        model = DepthModel(backend="depth_anything_v2_small", device="cuda")
        state = model.infer(bgr_frame)
    """

    def __init__(
        self,
        backend: str = "depth_anything_v2_small",
        device: str = "cuda",
        input_size: int = 518,
        fp16: bool = True,
        metric: bool = False,
    ):
        from depth.backends import get_backend

        BackendCls = get_backend(backend)
        self._backend = BackendCls(device=device, input_size=input_size, fp16=fp16, metric=metric)
        self._backend_name = backend

    def warmup(self, iterations: int = 3) -> None:
        """Run a few dummy inferences to stabilize CUDA/cuDNN kernels."""
        self._backend.warmup(iterations)

    def infer(self, frame: np.ndarray) -> DepthState:
        """Run depth inference on a single BGR/RGB frame.

        Args:
            frame: (H, W, 3) uint8 numpy array.

        Returns:
            DepthState with depth_map, valid_mask, timing, and metadata.
        """
        t0 = time.perf_counter()
        depth_map, valid_mask = self._backend.predict(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return DepthState(
            depth_map=depth_map,
            timestamp=time.time(),
            scale_mode=self._backend.scale_mode,
            valid_mask=valid_mask,
            backend_name=self._backend_name,
            inference_ms=elapsed_ms,
        )
