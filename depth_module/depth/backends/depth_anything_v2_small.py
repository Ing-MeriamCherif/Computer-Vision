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
        input_size: int | tuple[int, int] = 518,
        fp16: bool = True,
        metric: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.fp16 = fp16 and self.device.type == "cuda"
        self.metric = metric
        self._pipe = None
        self.last_device_depth = None
        self.last_device_valid_mask = None

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
            dtype=torch.float16 if self.fp16 else torch.float32,
        )
        # The pipeline processor otherwise resizes every frame to its model
        # default (518px), silently overriding the configured input_size.
        if getattr(self._pipe, "image_processor", None) is not None:
            self._pipe.image_processor.size = {
                "height": self.input_size[0],
                "width": self.input_size[1],
            }

    def warmup(self, iterations: int = 3) -> None:
        self._load()
        dummy = np.zeros((*self.input_size, 3), dtype=np.uint8)
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

        # Preserve the camera aspect ratio. The configured dimensions should
        # use the same ratio as the camera (for example 252x336 for 480x640).
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_AREA)
        pil_img = self._to_pil(resized)

        with torch.inference_mode():
            result = self._pipe(pil_img)
            # `depth` is only an 8-bit visualization. Keep the accompanying
            # floating-point tensor for geometry and accuracy.
            predicted = result["predicted_depth"]
            if predicted.ndim == 3:
                predicted = predicted[0]
            cpu_native_depth = predicted.float() if self.metric else predicted.float().clamp_min(1e-6).reciprocal()
            native_depth = cpu_native_depth.to(self.device, non_blocking=True)
            device_depth = torch.nn.functional.interpolate(
                native_depth[None, None], size=(h_orig, w_orig), mode="bicubic", align_corners=False,
            )[0, 0]
            device_valid = torch.isfinite(device_depth) & (device_depth > 1e-6)

        self.last_device_depth = device_depth
        self.last_device_valid_mask = device_valid
        # Transfer the smaller native model result and let OpenCV perform the
        # renderer-size resize. The full CUDA result remains resident for
        # geometry, avoiding a large blocking device-to-host copy here.
        depth_np = cpu_native_depth.detach().cpu().numpy().astype(np.float32, copy=False)
        if depth_np.shape != (h_orig, w_orig):
            depth_np = cv2.resize(depth_np, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        valid_mask = np.isfinite(depth_np) & (depth_np > 1e-6)

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
