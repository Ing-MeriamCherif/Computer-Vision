from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch


class DepthAnythingV2Small:
    """Depth Anything V2 backend (Small / Base / Large, relative or metric).

    Wraps any HuggingFace `depth-anything/Depth-Anything-V2-*-hf` model.
    Set metric=True to use metric models (output in meters).
    Install: pip install transformers
    """

    scale_mode = "relative"

    # Relative model IDs (normalized 0-1 output)
    _model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    # Metric model IDs (output in meters)
    _model_id_metric = "depth-anything/Depth-Anything-V2-Small-metric-hf"

    def __init__(
        self,
        device: str = "cuda",
        input_size: int = 518,
        fp16: bool = True,
        metric: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.fp16 = fp16 and self.device.type == "cuda"
        self.metric = metric
        self._pipe = None

        # Override scale_mode based on metric flag
        if metric:
            self.scale_mode = "metric"

    def _load(self) -> None:
        if self._pipe is not None:
            return
        from transformers import pipeline

        model_id = self._model_id_metric if self.metric else self._model_id
        self._pipe = pipeline(
            task="depth-estimation",
            model=model_id,
            device=0 if self.device.type == "cuda" else -1,
            torch_dtype=torch.float16 if self.fp16 else torch.float32,
        )

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
            depth_map: (H, W) float32.
                relative mode: values in [0, 1], closer=1.0 (bright).
                metric mode: depth in meters, closer=smaller values.
            valid_mask: (H, W) bool, True where depth is valid.
        """
        self._load()

        h_orig, w_orig = frame.shape[:2]

        # HuggingFace pipeline expects PIL RGB image; input frame is BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(
            rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA,
        )
        pil_img = self._to_pil(resized)

        with torch.inference_mode():
            result = self._pipe(pil_img)
        depth_tensor = result["depth"]  # PIL Image

        # Convert back to numpy
        depth_np = np.array(depth_tensor).astype(np.float32)

        # Resize to original resolution if pipeline changed it
        if depth_np.shape != (h_orig, w_orig):
            depth_np = cv2.resize(depth_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        if not self.metric:
            # Relative mode: normalize to [0, 1], invert so closer = 1.0 (bright)
            d_min, d_max = depth_np.min(), depth_np.max()
            if d_max - d_min > 1e-6:
                depth_np = 1.0 - (depth_np - d_min) / (d_max - d_min)
            else:
                depth_np = np.zeros_like(depth_np)
        # else: metric mode — keep raw meter values (no normalization)

        # Valid mask: all pixels valid for monocular depth
        valid_mask = np.ones((h_orig, w_orig), dtype=bool)

        return depth_np, valid_mask

    @staticmethod
    def _to_pil(rgb: np.ndarray):
        from PIL import Image
        return Image.fromarray(rgb)


class DepthAnythingV2Base(DepthAnythingV2Small):
    """Depth Anything V2 Base backend (97.5M params)."""
    _model_id = "depth-anything/Depth-Anything-V2-Base-hf"
    _model_id_metric = "depth-anything/Depth-Anything-V2-Base-metric-hf"


class DepthAnythingV2Large(DepthAnythingV2Small):
    """Depth Anything V2 Large backend (335.3M params)."""
    _model_id = "depth-anything/Depth-Anything-V2-Large-hf"
    _model_id_metric = "depth-anything/Depth-Anything-V2-Large-metric-hf"
