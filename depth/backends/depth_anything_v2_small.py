from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch


class DepthAnythingV2Small:
    """Depth Anything V2 backend (Small / Base / Large).

    Wraps any HuggingFace `depth-anything/Depth-Anything-V2-*-hf` model.
    Install: pip install transformers
    """

    scale_mode = "relative"

    # Default model — overridden by Base/Large subclasses
    _model_id = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(
        self,
        device: str = "cuda",
        input_size: int = 518,
        fp16: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.fp16 = fp16 and self.device.type == "cuda"
        self._pipe = None

    def _load(self) -> None:
        if self._pipe is not None:
            return
        from transformers import pipeline

        self._pipe = pipeline(
            task="depth-estimation",
            model=self._model_id,
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
            depth_map: (H, W) float32, values in [0, 1] range (relative).
            valid_mask: (H, W) bool, True where depth is valid.
        """
        self._load()

        h_orig, w_orig = frame.shape[:2]

        # HuggingFace pipeline expects PIL RGB image; input frame is BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Honor input_size so --resolutions actually changes inference load:
        # 336 -> 336x336, 420 -> 420x420, 518 -> 518x518.
        # Note: square resize distorts aspect ratio; fine for benchmarking,
        # prefer the model's aspect-preserving processor for quality runs.
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

        # Normalize to [0, 1] — invert so closer = 1.0 (bright), farther = 0.0 (dark).
        # Depth Anything outputs higher values for closer objects.
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max - d_min > 1e-6:
            depth_np = 1.0 - (depth_np - d_min) / (d_max - d_min)
        else:
            depth_np = np.zeros_like(depth_np)

        # Valid mask: all pixels valid for monocular relative depth
        valid_mask = np.ones((h_orig, w_orig), dtype=bool)

        return depth_np, valid_mask

    @staticmethod
    def _to_pil(rgb: np.ndarray):
        from PIL import Image
        return Image.fromarray(rgb)


class DepthAnythingV2Base(DepthAnythingV2Small):
    """Depth Anything V2 Base backend (97.5M params)."""
    _model_id = "depth-anything/Depth-Anything-V2-Base-hf"


class DepthAnythingV2Large(DepthAnythingV2Small):
    """Depth Anything V2 Large backend (335.3M params)."""
    _model_id = "depth-anything/Depth-Anything-V2-Large-hf"
