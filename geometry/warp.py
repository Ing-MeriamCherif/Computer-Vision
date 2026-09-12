"""Vectorized backward warping for dense temporal fields.

Backward flow is sampled at each current pixel and points to its source pixel
in the previous frame: ``source = current + backward_flow``.
"""

from __future__ import annotations

import numpy as np


def _validate_flow(field: np.ndarray, backward_flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flow = np.asarray(backward_flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError("backward_flow must have shape (H, W, 2)")
    value = np.asarray(field)
    if value.ndim not in (2, 3) or value.shape[:2] != flow.shape[:2]:
        raise ValueError("field must have shape (H, W) or (H, W, C) matching backward_flow")
    return value, flow


def warp_field_backward(
    field: np.ndarray,
    backward_flow: np.ndarray,
    source_valid_mask: np.ndarray | None = None,
    interpolation: str = "bilinear",
    fill_value: float | int = np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather a previous field into the current frame without pixel loops."""

    value, flow = _validate_flow(field, backward_flow)
    height, width = flow.shape[:2]
    if interpolation not in {"bilinear", "nearest"}:
        raise ValueError("interpolation must be 'bilinear' or 'nearest'")
    if source_valid_mask is None:
        source_valid = np.isfinite(value).all(axis=-1) if value.ndim == 3 else np.isfinite(value)
    else:
        source_valid = np.array(source_valid_mask, dtype=bool, copy=True)
        if source_valid.shape != (height, width):
            raise ValueError("source_valid_mask must have shape (H, W)")
        source_valid &= np.isfinite(value).all(axis=-1) if value.ndim == 3 else np.isfinite(value)

    yy, xx = np.indices((height, width), dtype=np.float32)
    source_x = xx + flow[..., 0]
    source_y = yy + flow[..., 1]
    finite_coords = np.isfinite(source_x) & np.isfinite(source_y)
    inside = finite_coords & (source_x >= 0) & (source_x <= width - 1) & (source_y >= 0) & (source_y <= height - 1)
    if interpolation == "nearest":
        x = np.rint(np.nan_to_num(source_x, nan=-1.0)).astype(np.intp)
        y = np.rint(np.nan_to_num(source_y, nan=-1.0)).astype(np.intp)
        safe = inside & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        safe &= np.where(safe, source_valid[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)], False)
        output = np.full_like(value, fill_value, dtype=np.result_type(value.dtype, np.float32))
        output[safe] = value[y[safe], x[safe]]
        return output, safe

    x0 = np.floor(np.nan_to_num(source_x, nan=-1.0)).astype(np.intp)
    y0 = np.floor(np.nan_to_num(source_y, nan=-1.0)).astype(np.intp)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = source_x - x0
    wy = source_y - y0
    output = np.full(value.shape, fill_value, dtype=np.result_type(value.dtype, np.float32))
    result_valid = np.zeros((height, width), dtype=bool)
    accum = np.zeros(value.shape, dtype=np.float32)
    total_weight = np.zeros((height, width), dtype=np.float32)
    for xi, yi, weight in ((x0, y0, (1 - wx) * (1 - wy)), (x1, y0, wx * (1 - wy)), (x0, y1, (1 - wx) * wy), (x1, y1, wx * wy)):
        valid_index = inside & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        clipped_x = np.clip(xi, 0, width - 1)
        clipped_y = np.clip(yi, 0, height - 1)
        valid_index &= source_valid[clipped_y, clipped_x]
        weight = np.where(valid_index, weight, 0.0).astype(np.float32)
        sample = value[clipped_y, clipped_x].astype(np.float32, copy=False)
        if value.ndim == 3:
            accum += sample * weight[..., None]
        else:
            accum += sample * weight
        total_weight += weight
    result_valid = total_weight > 0
    if value.ndim == 3:
        output[result_valid] = accum[result_valid] / total_weight[result_valid, None]
    else:
        output[result_valid] = accum[result_valid] / total_weight[result_valid]
    return output, result_valid


def warp_normals_backward(
    normals: np.ndarray,
    backward_flow: np.ndarray,
    source_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp normals bilinearly and re-normalize the valid result."""

    warped, valid = warp_field_backward(normals, backward_flow, source_valid_mask, "bilinear")
    lengths = np.linalg.norm(np.nan_to_num(warped), axis=-1)
    valid &= lengths > 1e-8
    warped = warped / np.where(valid, lengths, 1.0)[..., None]
    return np.where(valid[..., None], warped, np.nan).astype(np.float32), valid
