"""P1/P2/P3-only diagnostic visualizations.

These helpers intentionally do not import or depend on the Person 4 renderer.
"""

from __future__ import annotations

import cv2
import numpy as np


_DEPTH_STOPS = np.asarray(
    [[255, 45, 25], [255, 220, 40], [40, 210, 120], [30, 190, 235], [50, 80, 220], [180, 55, 220]],
    dtype=np.float32,
)


def _robust_bounds(values: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)[valid]
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, (2.0, 98.0))
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def depth_to_rgb(depth: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Map canonical depth to RGB: near is warm, far is cool, invalid black."""
    arr = np.asarray(depth, dtype=np.float32)
    mask = np.isfinite(arr) & (arr > 1e-6)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    lo, hi = _robust_bounds(arr, mask)
    # Larger forward-Z is farther; the first warm stop therefore represents
    # the near surface and the final cool stop the far surface.
    t = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    x = t * (_DEPTH_STOPS.shape[0] - 1)
    i0 = np.floor(x).astype(np.int32).clip(0, _DEPTH_STOPS.shape[0] - 1)
    i1 = np.ceil(x).astype(np.int32).clip(0, _DEPTH_STOPS.shape[0] - 1)
    frac = (x - i0)[..., None]
    rgb = _DEPTH_STOPS[i0] * (1.0 - frac) + _DEPTH_STOPS[i1] * frac
    return np.where(mask[..., None], np.clip(rgb, 0, 255), 0).astype(np.uint8)


def normals_to_rgb_diagnostic(normals: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Encode documented normal channels R=Nx, G=Ny, B=Nz."""
    arr = np.asarray(normals, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("normals must have shape (H, W, 3)")
    mask = np.isfinite(arr).all(axis=-1)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    rgb = np.clip((arr * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return np.where(mask[..., None], rgb, 0)


def confidence_to_rgb(confidence: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    arr = np.clip(np.nan_to_num(np.asarray(confidence, dtype=np.float32)), 0.0, 1.0)
    mask = np.isfinite(arr) if valid is None else np.asarray(valid, dtype=bool) & np.isfinite(arr)
    gray = (arr * 255.0).astype(np.uint8)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return np.where(mask[..., None], out, 0)


def add_depth_legend(image: np.ndarray, lo: float, hi: float, median: float) -> np.ndarray:
    out = np.asarray(image).copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (8, h - 44), (min(w - 8, 340), h - 8), (15, 15, 15), -1)
    cv2.putText(out, "NEAR  <---- DEPTH ---->  FAR", (14, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(out, f"min {lo:.2f}  med {median:.2f}  max {hi:.2f}", (14, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 210, 220), 1, cv2.LINE_AA)
    return out
