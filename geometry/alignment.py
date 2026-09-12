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
    model_used: str = "none"  # "affine", "scale_only", "none"
    normalized_fit_residual: float = 0.0
    inlier_count: int = 0
    input_sample_count: int = 0
    conditioning: float = 0.0


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

    The fit uses scale-normalized residuals ``r_norm = (q_cur - (scale * q_hist + shift)) / s_q``,
    where ``s_q = max(median(|q_cur|), epsilon)`` is the robust characteristic inverse-depth
    scale of the current scene. This makes Huber reweighting, inlier gating, and fit-success
    thresholds scale-invariant across relative depth maps with arbitrary global scale.

    If affine fitting is ill-conditioned (e.g. fronto-parallel surfaces with very low depth
    variation), a robust scale-only fallback (``shift = 0``) is used.
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
    valid = (
        mask
        & np.isfinite(current)
        & np.isfinite(history)
        & (current > epsilon)
        & (history > epsilon)
        & np.isfinite(supplied_weights)
        & (supplied_weights > 0)
    )
    sample_count = int(valid.sum())
    if sample_count < min_samples:
        return DepthAlignmentResult(
            1.0, 0.0, sample_count, float("inf"), False,
            model_used="none", normalized_fit_residual=float("inf"), inlier_count=0,
            input_sample_count=sample_count, conditioning=0.0
        )

    x = (1.0 / history[valid]).ravel()
    y = (1.0 / current[valid]).ravel()
    base_weights = supplied_weights[valid].ravel()

    # Characteristic inverse-depth scale for dimensionless normalization
    s_q = max(float(np.median(np.abs(y))), epsilon)

    # Measure depth diversity and design conditioning
    w_sum = float(np.sum(base_weights))
    x_mean = float(np.sum(base_weights * x) / max(w_sum, epsilon))
    x_var = float(np.sum(base_weights * (x - x_mean) ** 2) / max(w_sum, epsilon))
    rel_std_x = float(np.sqrt(max(x_var, 0.0)) / max(abs(x_mean), epsilon))

    design = np.column_stack((x, np.ones_like(x)))
    weighted_design = design * np.sqrt(base_weights)[:, None]
    try:
        _, s, _ = np.linalg.svd(weighted_design, full_matrices=False)
        conditioning = float(s[1] / max(s[0], epsilon)) if len(s) > 1 else 0.0
    except np.linalg.LinAlgError:
        conditioning = 0.0

    # Low-variation / degeneracy test: if spread is small or matrix is ill-conditioned, use scale-only
    is_well_conditioned = (rel_std_x >= 0.02) and (conditioning >= 1e-4)

    # 1. Try affine model if well-conditioned
    if is_well_conditioned:
        scale_init = float(np.median(y) / max(np.median(x), epsilon))
        shift_init = float(np.median(y - scale_init * x))
        r_init = (y - (scale_init * x + shift_init)) / s_q
        abs_r_init = np.abs(r_init)
        med_init = float(np.median(abs_r_init))
        mad_init = float(np.median(np.abs(abs_r_init - med_init)))
        huber_init = max(1.4826 * mad_init, residual_threshold / 4.0, epsilon)
        robust_weights = base_weights * np.minimum(1.0, huber_init / np.maximum(abs_r_init, epsilon))
        scale, shift = scale_init, shift_init
        for _ in range(iterations):
            w_des = design * np.sqrt(robust_weights)[:, None]
            w_y = y * np.sqrt(robust_weights)
            solution, *_ = np.linalg.lstsq(w_des, w_y, rcond=None)
            scale, shift = float(solution[0]), float(solution[1])
            residual = y - (scale * x + shift)
            r_norm = residual / s_q
            abs_r_norm = np.abs(r_norm)
            med_norm = float(np.median(abs_r_norm))
            mad_norm = float(np.median(np.abs(abs_r_norm - med_norm)))
            huber_scale_norm = max(1.4826 * mad_norm, residual_threshold / 4.0, epsilon)
            robust_factor = np.minimum(1.0, huber_scale_norm / np.maximum(abs_r_norm, epsilon))
            robust_weights = base_weights * robust_factor

        residual = y - (scale * x + shift)
        r_norm = residual / s_q
        abs_r_norm = np.abs(r_norm)
        med_norm = float(np.median(abs_r_norm))
        mad_norm = float(np.median(np.abs(abs_r_norm - med_norm)))
        gate_norm = max(residual_threshold, med_norm + 3.0 * 1.4826 * mad_norm)
        inliers = abs_r_norm <= gate_norm
        inlier_count = int(inliers.sum())

        if inlier_count >= min_samples and np.isfinite(scale + shift) and scale > epsilon:
            norm_res = float(np.sqrt(np.average(r_norm[inliers] ** 2, weights=base_weights[inliers])))
            if norm_res <= residual_threshold:
                return DepthAlignmentResult(
                    scale=scale,
                    shift=shift,
                    sample_count=inlier_count,
                    fit_residual=norm_res,
                    fit_success=True,
                    model_used="affine",
                    normalized_fit_residual=norm_res,
                    inlier_count=inlier_count,
                    input_sample_count=sample_count,
                    conditioning=conditioning,
                )

    # 2. Scale-only fallback (b = 0): q_cur = a * q_hist
    robust_weights = base_weights.copy()
    scale = 1.0
    shift = 0.0
    for _ in range(iterations):
        denom = float(np.sum(robust_weights * (x ** 2)))
        if denom <= epsilon:
            break
        scale = float(np.sum(robust_weights * x * y) / denom)
        if scale <= epsilon or not np.isfinite(scale):
            break
        residual = y - scale * x
        r_norm = residual / s_q
        abs_r_norm = np.abs(r_norm)
        med_norm = float(np.median(abs_r_norm))
        mad_norm = float(np.median(np.abs(abs_r_norm - med_norm)))
        huber_scale_norm = max(1.4826 * mad_norm, residual_threshold / 4.0, epsilon)
        robust_factor = np.minimum(1.0, huber_scale_norm / np.maximum(abs_r_norm, epsilon))
        robust_weights = base_weights * robust_factor

    residual = y - scale * x
    r_norm = residual / s_q
    abs_r_norm = np.abs(r_norm)
    med_norm = float(np.median(abs_r_norm))
    mad_norm = float(np.median(np.abs(abs_r_norm - med_norm)))
    gate_norm = max(residual_threshold, med_norm + 3.0 * 1.4826 * mad_norm)
    inliers = abs_r_norm <= gate_norm
    inlier_count = int(inliers.sum())

    if inlier_count >= min_samples and np.isfinite(scale) and scale > epsilon:
        norm_res = float(np.sqrt(np.average(r_norm[inliers] ** 2, weights=base_weights[inliers])))
        if norm_res <= residual_threshold:
            return DepthAlignmentResult(
                scale=scale,
                shift=0.0,
                sample_count=inlier_count,
                fit_residual=norm_res,
                fit_success=True,
                model_used="scale_only",
                normalized_fit_residual=norm_res,
                inlier_count=inlier_count,
                input_sample_count=sample_count,
                conditioning=conditioning,
            )

    # 3. Fit failed
    raw_norm_res = float(np.sqrt(np.mean((residual / s_q) ** 2)))
    return DepthAlignmentResult(
        scale=1.0,
        shift=0.0,
        sample_count=sample_count,
        fit_residual=raw_norm_res,
        fit_success=False,
        model_used="none",
        normalized_fit_residual=raw_norm_res,
        inlier_count=inlier_count,
        input_sample_count=sample_count,
        conditioning=conditioning,
    )


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
