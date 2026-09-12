"""Phase 2 — DepthAnythingV2 depth estimator (HuggingFace transformers).

Relative (non-metric) depth: normalized to 0.5-5m working volume for the
normals translator. Metric scale arrives with DepthPro/metric heads later;
hand Z keeps using the size proxy until then.
Model: depth-anything/Depth-Anything-V2-Small (cdc primary choice).
"""
from __future__ import annotations

import numpy as np

import config


class DepthEstimator:
    def __init__(self, model_id: str | None = None, device: str | None = None,
                 input_size: int | None = None) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch
        self.model_id = model_id or config.DEPTH_MODEL_ID
        dev = (device or config.DEPTH_DEVICE).lower()
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = dev
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self.model.to(dev).eval()
        self.size = input_size or config.DEPTH_INPUT_SIZE

    @property
    def metric(self) -> bool:
        return False  # relative depth; see module docstring

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> depth float32 meters-ish (0.5-5m working volume)."""
        import torch
        h, w = rgb.shape[:2]
        inputs = self.proc(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            out = self.model(pixel_values).predicted_depth[0]
        d = out.cpu().numpy()
        d = np.array(d, dtype=np.float64)
        # normalize relative disparity-ish output into a working volume
        d -= d.min()
        if d.max() > 1e-9:
            d /= d.max()
        depth = (0.5 + 4.5 * (1.0 - d / (d.max() + 1e-9))).astype(np.float64)
        import cv2
        if (depth.shape[0], depth.shape[1]) != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        return depth.astype(np.float32)
