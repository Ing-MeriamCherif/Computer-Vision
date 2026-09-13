"""YOLO26 depth estimation backend.

Wraps Ultralytics YOLO26 depth models (yolo26n/s/m/l/x-depth).
These output metric depth in meters by default.

Models:
    yolo26n-depth  (6.4M params,  fastest)
    yolo26s-depth  (13.2M params)
    yolo26m-depth  (23.3M params)
    yolo26l-depth  (27.7M params)
    yolo26x-depth  (57.0M params, most accurate)
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch


class YOLO26Depth:
    """YOLO26 depth backend (any size)."""

    scale_mode = "metric"

    # Default model — overridden by subclasses
    _model_name = "yolo26n-depth.pt"

    def __init__(
        self,
        device: str = "cuda",
        input_size: int = 640,
        fp16: bool = True,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.fp16 = fp16 and self.device == "cuda"
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._model = YOLO(self._model_name)

    def warmup(self, iterations: int = 3) -> None:
        self._load()
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.predict(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run depth inference.

        Args:
            frame: (H, W, 3) uint8 BGR image.

        Returns:
            depth_map: (H, W) float32, values in meters.
            valid_mask: (H, W) bool, True where depth is valid.
        """
        self._load()

        h_orig, w_orig = frame.shape[:2]

        # YOLO expects RGB BGR is fine — predict handles conversion
        results = self._model.predict(
            source=frame,
            imgsz=self.input_size,
            verbose=False,
        )

        # result.depth is (H, W) float32 tensor in meters
        depth_tensor = results[0].depth
        if depth_tensor is None:
            depth_np = np.zeros((h_orig, w_orig), dtype=np.float32)
        else:
            depth_np = depth_tensor.cpu().numpy().astype(np.float32)

        # Resize to original resolution if needed
        if depth_np.shape != (h_orig, w_orig):
            depth_np = cv2.resize(depth_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        # Valid mask: finite values
        valid_mask = np.isfinite(depth_np)

        return depth_np, valid_mask


class YOLO26nDepth(YOLO26Depth):
    """YOLO26 Nano depth (6.4M params, fastest)."""
    _model_name = "yolo26n-depth.pt"


class YOLO26sDepth(YOLO26Depth):
    """YOLO26 Small depth (13.2M params)."""
    _model_name = "yolo26s-depth.pt"


class YOLO26mDepth(YOLO26Depth):
    """YOLO26 Medium depth (23.3M params)."""
    _model_name = "yolo26m-depth.pt"


class YOLO26lDepth(YOLO26Depth):
    """YOLO26 Large depth (27.7M params)."""
    _model_name = "yolo26l-depth.pt"


class YOLO26xDepth(YOLO26Depth):
    """YOLO26 Extra-Large depth (57.0M params, most accurate)."""
    _model_name = "yolo26x-depth.pt"
