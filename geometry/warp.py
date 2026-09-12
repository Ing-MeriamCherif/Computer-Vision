"""Vectorized backward warping for dense temporal fields.

Backward flow is sampled at each current pixel and points to its source pixel
in the previous frame: ``source = current + backward_flow``.
"""

from __future__ import annotations

import warnings

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
        sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
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


def warp_depth_backward(
    depth: np.ndarray,
    backward_flow: np.ndarray,
    source_valid_mask: np.ndarray | None = None,
    discontinuity_threshold: float = 0.15,
    epsilon: float = 1e-6,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather a previous depth field with discontinuity-aware interpolation.

    Unlike generic bilinear interpolation, this function prevents creating
    fictitious intermediate surfaces across strong step discontinuities
    (e.g. averaging foreground z=1.0 and background z=3.0 into non-existent z=2.0).

    When a bilinear 2x2 footprint crosses a relative depth discontinuity exceeding
    ``discontinuity_threshold``, the output selects the highest-weight valid physical
    surface layer rather than blending across the boundary, and marks the boundary
    with reduced confidence.
    """

    value, flow = _validate_flow(depth, backward_flow)
    if value.ndim != 2:
        raise ValueError("depth must have shape (H, W)")
    if discontinuity_threshold <= 0:
        raise ValueError("discontinuity_threshold must be positive")
    height, width = flow.shape[:2]
    if source_valid_mask is None:
        source_valid = np.isfinite(value) & (value > epsilon)
    else:
        source_valid = np.asarray(source_valid_mask, dtype=bool)
        if source_valid.shape != (height, width):
            raise ValueError("source_valid_mask must have shape (H, W)")
        source_valid &= np.isfinite(value) & (value > epsilon)

    yy, xx = np.indices((height, width), dtype=np.float32)
    source_x = xx + flow[..., 0]
    source_y = yy + flow[..., 1]
    finite_coords = np.isfinite(source_x) & np.isfinite(source_y)
    inside = finite_coords & (source_x >= 0) & (source_x <= width - 1) & (source_y >= 0) & (source_y <= height - 1)

    x0 = np.floor(np.nan_to_num(source_x, nan=-1.0)).astype(np.intp)
    y0 = np.floor(np.nan_to_num(source_y, nan=-1.0)).astype(np.intp)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = source_x - x0
    wy = source_y - y0

    coords = (
        (x0, y0, (1.0 - wx) * (1.0 - wy)),
        (x1, y0, wx * (1.0 - wy)),
        (x0, y1, (1.0 - wx) * wy),
        (x1, y1, wx * wy),
    )

    neighbor_samples = []
    neighbor_weights = []
    neighbor_valids = []

    for xi, yi, raw_w in coords:
        valid_idx = inside & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        clipped_x = np.clip(xi, 0, width - 1)
        clipped_y = np.clip(yi, 0, height - 1)
        valid_idx &= source_valid[clipped_y, clipped_x]
        w = np.where(valid_idx, np.maximum(raw_w, 0.0), 0.0).astype(np.float32)
        sample = np.where(valid_idx, value[clipped_y, clipped_x], np.nan).astype(np.float32)
        neighbor_samples.append(sample)
        neighbor_weights.append(w)
        neighbor_valids.append(valid_idx)

    samples_arr = np.stack(neighbor_samples, axis=0)
    weights_arr = np.stack(neighbor_weights, axis=0)
    valids_arr = np.stack(neighbor_valids, axis=0)

    total_weight = np.sum(weights_arr, axis=0)
    has_support = total_weight > 0

    accum = np.sum(np.nan_to_num(samples_arr) * weights_arr, axis=0)
    bilinear_depth = np.where(has_support, accum / np.maximum(total_weight, 1e-12), np.nan)

    active_samples = np.where(valids_arr & (weights_arr > 1e-6), samples_arr, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        min_depth = np.nanmin(active_samples, axis=0)
        max_depth = np.nanmax(active_samples, axis=0)
    min_depth = np.where(has_support, np.nan_to_num(min_depth, nan=0.0), 0.0)
    max_depth = np.where(has_support, np.nan_to_num(max_depth, nan=0.0), 0.0)

    denominator = np.maximum(min_depth, epsilon)
    rel_spread = np.where(has_support, (max_depth - min_depth) / denominator, 0.0)

    crosses_discontinuity = has_support & (rel_spread > discontinuity_threshold)

    masked_weights = np.where(valids_arr, weights_arr, -1.0)
    best_neighbor = np.argmax(masked_weights, axis=0)
    fallback_depth = np.take_along_axis(samples_arr, best_neighbor[None, ...], axis=0)[0]

    output_depth = np.where(crosses_discontinuity, fallback_depth, bilinear_depth)
    output_depth = np.where(has_support, output_depth, fill_value).astype(np.float32)
    output_valid = has_support & np.isfinite(output_depth) & (output_depth > epsilon)

    warp_conf = np.where(
        crosses_discontinuity,
        np.clip(0.5 - 0.1 * (rel_spread / np.maximum(discontinuity_threshold, 1e-6)), 0.35, 0.49),
        np.clip(1.0 - 0.5 * (rel_spread / np.maximum(discontinuity_threshold, 1e-6)), 0.7, 1.0),
    ).astype(np.float32)
    warp_conf = np.where(output_valid, warp_conf, 0.0).astype(np.float32)

    return output_depth, output_valid, warp_conf
