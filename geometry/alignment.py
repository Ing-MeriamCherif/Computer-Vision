"""Robust affine alignment for relative monocular depth history."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backproject import DepthScaleMode


@dataclass(frozen=True, slots=True)
class DepthAlignmentResult:
    scale: float
    shift: float
    sample_count: int
    fit_residual: float
    fit_success: bool


def align_inverse_depth(
    current_depth: np.ndarray,
    history_depth: np.ndarray,
    correspondence_mask: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    min_samples: int = 32,
    residual_threshold: float = 0.04,
    epsilon: float = 1e-6,
    iterations: int = 4,
) -> DepthAlignmentResult:
    """Fit ``1/current_z = scale * 1/history_z + shift`` robustly.

    The fit uses weighted least squares followed by Huber-like residual
    reweighting. The final residual median/MAD gate excludes moving-object and
    correspondence outliers without requiring SciPy.
    """

    current = np.asarray(current_depth, dtype=np.float64)
    history = np.asarray(history_depth, dtype=np.float64)
    mask = np.asarray(correspondence_mask, dtype=bool)
    if current.shape != history.shape or current.shape != mask.shape or current.ndim != 2:
        raise ValueError("depth arrays and correspondence_mask must have matching shape (H, W)")
    if min_samples < 2 or residual_threshold <= 0 or epsilon <= 0 or iterations < 1:
        raise ValueError("alignment parameters are invalid")
    if weights is None:
        supplied_weights = np.ones(current.shape, dtype=np.float64)
    else:
        supplied_weights = np.asarray(weights, dtype=np.float64)
        if supplied_weights.shape != current.shape:
            raise ValueError("weights must have shape (H, W)")
    valid = mask & np.isfinite(current) & np.isfinite(history) & (current > epsilon) & (history > epsilon) & np.isfinite(supplied_weights) & (supplied_weights > 0)
    sample_count = int(valid.sum())
    if sample_count < min_samples:
        return DepthAlignmentResult(1.0, 0.0, sample_count, float("inf"), False)
    x = (1.0 / history[valid]).ravel()
    y = (1.0 / current[valid]).ravel()
    base_weights = supplied_weights[valid].ravel()
    robust_weights = base_weights.copy()
    scale, shift = 1.0, 0.0
    for _ in range(iterations):
        design = np.column_stack((x, np.ones_like(x)))
        weighted_design = design * np.sqrt(robust_weights)[:, None]
        weighted_y = y * np.sqrt(robust_weights)
        solution, *_ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
        scale, shift = float(solution[0]), float(solution[1])
        residual = y - (scale * x + shift)
        abs_residual = np.abs(residual)
        mad = float(np.median(np.abs(abs_residual - np.median(abs_residual))))
        huber_scale = max(1.4826 * mad, residual_threshold / 4.0, epsilon)
        robust_factor = np.minimum(1.0, huber_scale / np.maximum(abs_residual, epsilon))
        robust_weights = base_weights * robust_factor
    residual = y - (scale * x + shift)
    abs_residual = np.abs(residual)
    median_abs = float(np.median(abs_residual))
    mad = float(np.median(np.abs(abs_residual - median_abs)))
    gate = max(residual_threshold, median_abs + 3.0 * 1.4826 * mad)
    inliers = abs_residual <= gate
    if int(inliers.sum()) < min_samples or not np.isfinite(scale + shift) or scale <= epsilon:
        return DepthAlignmentResult(1.0, 0.0, sample_count, float(np.sqrt(np.mean(residual**2))), False)
    fit_residual = float(np.sqrt(np.average(residual[inliers] ** 2, weights=base_weights[inliers])))
    success = fit_residual <= residual_threshold
    return DepthAlignmentResult(scale if success else 1.0, shift if success else 0.0, int(inliers.sum()), fit_residual, success)


def align_history_depth(
    history_depth: np.ndarray,
    result: DepthAlignmentResult,
    scale_mode: DepthScaleMode | str,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a successful alignment and return aligned Z-depth plus validity."""

    mode = DepthScaleMode(scale_mode)
    history = np.asarray(history_depth, dtype=np.float32)
    valid = np.isfinite(history) & (history > epsilon)
    if mode is DepthScaleMode.METRIC or not result.fit_success:
        return history.copy(), valid
    if mode is not DepthScaleMode.RELATIVE:
        raise ValueError("inverse depth must be explicitly converted before history alignment")
    inverse = np.zeros_like(history, dtype=np.float32)
    inverse[valid] = 1.0 / history[valid]
    aligned_inverse = result.scale * inverse + result.shift
    aligned_valid = valid & np.isfinite(aligned_inverse) & (aligned_inverse > epsilon)
    aligned = np.full(history.shape, np.nan, dtype=np.float32)
    aligned[aligned_valid] = 1.0 / aligned_inverse[aligned_valid]
    return aligned, aligned_valid
