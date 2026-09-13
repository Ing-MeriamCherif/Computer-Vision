"""Renderer-independent robust depth sampling for hand/depth fusion."""

from __future__ import annotations

import numpy as np


def _bilinear(depth: np.ndarray, u: float, v: float) -> float:
    h, w = depth.shape
    x = float(np.clip(u, 0.0, max(w - 1, 0)))
    y = float(np.clip(v, 0.0, max(h - 1, 0)))
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    return float(
        depth[y0, x0] * (1 - dx) * (1 - dy)
        + depth[y0, x1] * dx * (1 - dy)
        + depth[y1, x0] * (1 - dx) * dy
        + depth[y1, x1] * dx * dy
    )


def sample_depth(
    depth: np.ndarray,
    valid: np.ndarray | None,
    u: float,
    v: float,
    radius: int = 2,
) -> tuple[float, float]:
    """Return ``(z, reliability)`` at an RGB pixel.

    Bilinear sampling is used for a valid footprint.  At invalid or
    discontinuous pixels a bounded median neighborhood is used instead.  The
    reliability is evidence from valid samples, not neural-model certainty.
    """
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim != 2:
        raise ValueError("depth must be a 2D array")
    h, w = depth_arr.shape
    if h == 0 or w == 0:
        return 0.0, 0.0
    valid_arr = None if valid is None else np.asarray(valid, dtype=bool)
    if valid_arr is not None and valid_arr.shape != depth_arr.shape:
        raise ValueError("valid mask must match depth shape")
    x = int(np.clip(round(float(u)), 0, w - 1))
    y = int(np.clip(round(float(v)), 0, h - 1))
    center_valid = np.isfinite(depth_arr[y, x]) and depth_arr[y, x] > 1e-6
    if valid_arr is not None:
        center_valid = center_valid and bool(valid_arr[y, x])
    if center_valid:
        z = _bilinear(depth_arr, u, v)
        if np.isfinite(z) and z > 1e-6:
            return z, 1.0
    radius = max(0, int(radius))
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    patch = depth_arr[y0:y1, x0:x1]
    mask = np.isfinite(patch) & (patch > 1e-6)
    if valid_arr is not None:
        mask &= valid_arr[y0:y1, x0:x1]
    values = patch[mask]
    if values.size == 0:
        return 0.0, 0.0
    footprint = max((2 * radius + 1) ** 2, 1)
    return float(np.median(values)), float(min(1.0, values.size / footprint))


def camera_uv_to_depth_uv(
    uv: tuple[float, float],
    camera_size: tuple[int, int],
    depth_size: tuple[int, int],
) -> tuple[float, float]:
    """Map pixel coordinates from ``(width, height)`` RGB to depth space."""
    cw, ch = camera_size
    dw, dh = depth_size
    if cw <= 0 or ch <= 0 or dw <= 0 or dh <= 0:
        raise ValueError("image dimensions must be positive")
    return float(uv[0] * dw / cw), float(uv[1] * dh / ch)
