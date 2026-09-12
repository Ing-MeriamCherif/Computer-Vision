"""Vectorized spatial surface-normal estimation and geometry confidence.

Normals use the project camera convention and face the camera whenever the
orientation is unambiguous: ``normal dot position <= 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import warnings

import numpy as np

from .backproject import backproject_depth
from .camera import CameraModel
from .state import DepthState, GeometryState


class NormalMode(str, Enum):
    BASELINE = "baseline"
    EDGE_AWARE = "edge_aware"
    MULTI_SCALE = "multi_scale"


@dataclass(frozen=True, slots=True)
class NormalConfig:
    """Named, interpretable controls for spatial normal estimation."""

    discontinuity_threshold: float = 0.15
    depth_epsilon: float = 1e-6
    min_tangent_norm: float = 1e-6
    min_tangent_conditioning: float = 1e-8
    min_cross_norm: float | None = None
    radii: tuple[int, ...] = (1, 2, 4)
    multi_scale_acceptance: float = 0.75
    edge_confidence_penalty: float = 0.5

    def __post_init__(self) -> None:
        if self.discontinuity_threshold <= 0:
            raise ValueError("discontinuity_threshold must be positive")
        if self.min_cross_norm is not None:
            warnings.warn("min_cross_norm is deprecated; use min_tangent_conditioning", DeprecationWarning, stacklevel=2)
            if self.min_tangent_conditioning != 1e-8 and not np.isclose(self.min_tangent_conditioning, self.min_cross_norm):
                raise ValueError("min_cross_norm conflicts with min_tangent_conditioning")
            object.__setattr__(self, "min_tangent_conditioning", float(self.min_cross_norm))
        if self.depth_epsilon <= 0 or self.min_tangent_norm <= 0 or self.min_tangent_conditioning <= 0:
            raise ValueError("geometry minimums must be positive")
        raw_radii = tuple(self.radii)
        if not raw_radii or any(isinstance(radius, bool) or not isinstance(radius, (int, np.integer)) or radius <= 0 for radius in raw_radii):
            raise ValueError("radii must contain positive values")
        normalized_radii = tuple(sorted(set(int(radius) for radius in raw_radii)))
        object.__setattr__(self, "radii", normalized_radii)
        if not 0 <= self.multi_scale_acceptance <= 1:
            raise ValueError("multi_scale_acceptance must be in [0, 1]")
        if not 0 <= self.edge_confidence_penalty <= 1:
            raise ValueError("edge_confidence_penalty must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class NormalResult:
    normals: np.ndarray
    normal_valid_mask: np.ndarray
    confidence: np.ndarray
    discontinuity_strength: np.ndarray
    selected_radius: np.ndarray
    neighbor_support: np.ndarray


@dataclass(frozen=True, slots=True)
class AngularMetrics:
    mean_degrees: float
    median_degrees: float
    p95_degrees: float
    valid_percentage: float
    sample_count: int


def _validate_inputs(points: np.ndarray, valid_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    points_arr = np.asarray(points, dtype=np.float32)
    if points_arr.ndim != 3 or points_arr.shape[-1] != 3:
        raise ValueError("positions must have shape (H, W, 3)")
    finite = np.isfinite(points_arr).all(axis=-1)
    if valid_mask is None:
        valid = finite
    else:
        valid = np.array(valid_mask, dtype=bool, copy=True)
        if valid.shape != points_arr.shape[:2]:
            raise ValueError("valid_mask must have shape (H, W)")
        valid &= finite
    return points_arr, valid


def _neighbor(points: np.ndarray, valid: np.ndarray, radius: int, axis: int, direction: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a shifted neighbor with out-of-frame pixels marked invalid."""

    shifted = np.full_like(points, np.nan)
    shifted_valid = np.zeros_like(valid)
    if axis == 1:
        source = slice(None, -radius) if direction > 0 else slice(radius, None)
        target = slice(radius, None) if direction > 0 else slice(None, -radius)
    else:
        source = slice(None, -radius) if direction > 0 else slice(radius, None)
        target = slice(radius, None) if direction > 0 else slice(None, -radius)
    if axis == 1:
        shifted[:, target] = points[:, source]
        shifted_valid[:, target] = valid[:, source]
    else:
        shifted[target, :] = points[source, :]
        shifted_valid[target, :] = valid[source, :]
    return shifted, shifted_valid


def _relative_jump(center_z: np.ndarray, neighbor_z: np.ndarray, valid: np.ndarray, epsilon: float) -> np.ndarray:
    denominator = np.maximum(np.minimum(np.abs(center_z), np.abs(neighbor_z)), epsilon)
    jump = np.abs(center_z - neighbor_z) / denominator
    return np.where(valid & np.isfinite(jump), jump, 0.0)


def _axis_tangent(
    points: np.ndarray,
    valid: np.ndarray,
    depth: np.ndarray,
    radius: int,
    axis: int,
    edge_aware: bool,
    config: NormalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left, left_valid = _neighbor(points, valid, radius, axis, -1)
    right, right_valid = _neighbor(points, valid, radius, axis, 1)
    center_valid = valid
    left_pair = center_valid & left_valid
    right_pair = center_valid & right_valid
    left_jump = _relative_jump(depth, left[..., 2], left_pair, config.depth_epsilon)
    right_jump = _relative_jump(depth, right[..., 2], right_pair, config.depth_epsilon)
    if edge_aware:
        left_pair &= left_jump <= config.discontinuity_threshold
        right_pair &= right_jump <= config.discontinuity_threshold

    left_tangent = points - left
    right_tangent = right - points
    both = left_pair & right_pair
    choose_left = ~both & left_pair
    choose_right = ~both & ~choose_left & right_pair
    tangent = np.zeros_like(points)
    tangent[both] = right[both] - left[both]
    tangent[choose_left] = left_tangent[choose_left]
    tangent[choose_right] = right_tangent[choose_right]

    tangent_length = np.linalg.norm(tangent, axis=-1)

    # Scale-invariant tangent quality:
    # Normalize tangent length by local depth magnitude to obtain dimensionless relative span rho.
    local_scale = np.maximum(np.abs(depth), 1e-12)
    rho = tangent_length / local_scale
    tangent_valid = (both | choose_left | choose_right) & (rho > 1e-12)
    tangent_quality = rho / (rho + config.min_tangent_norm)
    tangent_quality = np.where(tangent_valid, np.clip(tangent_quality, 0.0, 1.0), 0.0)

    support = (left_pair.astype(np.float32) + right_pair.astype(np.float32)) / 2.0
    discontinuity = np.maximum(left_jump, right_jump)
    discontinuity = np.clip(discontinuity / config.discontinuity_threshold, 0.0, 1.0)
    return tangent, tangent_valid, tangent_quality, np.stack((support, discontinuity), axis=-1)


def _estimate_at_radius(
    points: np.ndarray,
    valid: np.ndarray,
    depth: np.ndarray,
    radius: int,
    edge_aware: bool,
    config: NormalConfig,
) -> NormalResult:
    tangent_x, valid_x, quality_x, support_x = _axis_tangent(points, valid, depth, radius, 1, edge_aware, config)
    tangent_y, valid_y, quality_y, support_y = _axis_tangent(points, valid, depth, radius, 0, edge_aware, config)
    cross = np.cross(tangent_y, tangent_x)
    cross_norm = np.linalg.norm(cross, axis=-1)

    len_x = np.linalg.norm(tangent_x, axis=-1)
    len_y = np.linalg.norm(tangent_y, axis=-1)
    lengths_product = len_x * len_y

    # Dimensionless cross-product conditioning: sin(theta) = ||Ty x Tx|| / (||Tx|| ||Ty||)
    guard = 1e-12 * np.maximum(lengths_product, 1e-12)
    conditioning_quality = cross_norm / (lengths_product + guard)
    conditioning_quality = np.where(valid_x & valid_y, np.clip(conditioning_quality, 0.0, 1.0), 0.0)

    valid_normals = (
        valid_x
        & valid_y
        & (lengths_product > 1e-24)
        & (cross_norm > 1e-12 * lengths_product)
        & (conditioning_quality >= config.min_tangent_conditioning)
    )
    normals = np.zeros_like(points)
    safe_norm = np.where(valid_normals, cross_norm, 1.0)
    normals = cross / safe_norm[..., None]
    # Face the camera. This is deterministic and prevents arbitrary sign flips.
    flip = np.sum(normals * points, axis=-1) > 0
    normals = np.where(flip[..., None], -normals, normals)
    normals = np.where(valid_normals[..., None], normals, np.nan)

    tangent_quality = np.sqrt(quality_x * quality_y)
    support = (support_x[..., 0] + support_y[..., 0]) / 2.0
    discontinuity = np.maximum(support_x[..., 1], support_y[..., 1])
    edge_penalty = 1.0 - config.edge_confidence_penalty * discontinuity
    confidence = support * tangent_quality * conditioning_quality * edge_penalty
    confidence = np.where(valid_normals, np.clip(confidence, 0.0, 1.0), 0.0).astype(np.float32)
    return NormalResult(
        normals.astype(np.float32),
        valid_normals,
        confidence,
        discontinuity.astype(np.float32),
        np.where(valid_normals, radius, 0).astype(np.int16),
        support.astype(np.float32),
    )


def _select_multiscale(
    candidates: list[NormalResult],
    acceptance_threshold: float,
) -> NormalResult:
    """Select normals across scale candidates with acceptance-first, best-fallback policy.

    Pass 1 selects the first (smallest radius) valid candidate whose confidence
    meets or exceeds ``acceptance_threshold``.
    Pass 2 handles remaining unselected pixels by picking the valid candidate with
    the highest confidence (breaking ties by preferring the smaller radius).
    Pixels with no valid candidate remain invalid with zero confidence and radius 0.
    """
    if not candidates:
        raise ValueError("candidates list cannot be empty")

    shape = candidates[0].normal_valid_mask.shape
    point_shape = (*shape, 3)

    selected_normals = np.full(point_shape, np.nan, dtype=np.float32)
    selected_valid = np.zeros(shape, dtype=bool)
    selected_confidence = np.zeros(shape, dtype=np.float32)
    selected_discontinuity = np.zeros(shape, dtype=np.float32)
    selected_radius = np.zeros(shape, dtype=np.int16)
    selected_support = np.zeros(shape, dtype=np.float32)

    accepted_mask = np.zeros(shape, dtype=bool)

    # Pass 1: first valid candidate meeting acceptance threshold
    for candidate in candidates:
        take = (~accepted_mask) & candidate.normal_valid_mask & (candidate.confidence >= acceptance_threshold)
        selected_normals[take] = candidate.normals[take]
        selected_confidence[take] = candidate.confidence[take]
        selected_discontinuity[take] = candidate.discontinuity_strength[take]
        selected_radius[take] = candidate.selected_radius[take]
        selected_support[take] = candidate.neighbor_support[take]
        selected_valid[take] = True
        accepted_mask[take] = True

    # Pass 2: fallback to valid candidate with maximum confidence
    fallback_mask = ~accepted_mask
    has_best = np.zeros(shape, dtype=bool)
    best_confidence = np.full(shape, -1.0, dtype=np.float32)

    for candidate in candidates:
        eligible = fallback_mask & candidate.normal_valid_mask
        better = eligible & ((~has_best) | (candidate.confidence > best_confidence))
        selected_normals[better] = candidate.normals[better]
        selected_confidence[better] = candidate.confidence[better]
        selected_discontinuity[better] = candidate.discontinuity_strength[better]
        selected_radius[better] = candidate.selected_radius[better]
        selected_support[better] = candidate.neighbor_support[better]
        selected_valid[better] = True
        best_confidence[better] = candidate.confidence[better]
        has_best[better] = True

    return NormalResult(
        normals=selected_normals,
        normal_valid_mask=selected_valid,
        confidence=selected_confidence,
        discontinuity_strength=selected_discontinuity,
        selected_radius=selected_radius,
        neighbor_support=selected_support,
    )


def _select_multiscale_streaming(
    points: np.ndarray,
    valid: np.ndarray,
    depth: np.ndarray,
    config: NormalConfig,
) -> NormalResult:
    """Memory-bounded equivalent of :func:`_select_multiscale`.

    One radius is materialized at a time; selection order and tie-breaking match
    the candidate-list reference implementation exactly.
    """
    shape = valid.shape
    selected_normals = np.full((*shape, 3), np.nan, dtype=np.float32)
    selected_valid = np.zeros(shape, dtype=bool)
    selected_confidence = np.zeros(shape, dtype=np.float32)
    selected_discontinuity = np.zeros(shape, dtype=np.float32)
    selected_radius = np.zeros(shape, dtype=np.int16)
    selected_support = np.zeros(shape, dtype=np.float32)
    accepted = np.zeros(shape, dtype=bool)
    best_confidence = np.full(shape, -1.0, dtype=np.float32)
    best_has = np.zeros(shape, dtype=bool)
    for radius in config.radii:
        candidate = _estimate_at_radius(points, valid, depth, radius, True, config)
        eligible = (~accepted) & candidate.normal_valid_mask
        better = eligible & ((~best_has) | (candidate.confidence > best_confidence))
        best_confidence[better] = candidate.confidence[better]
        best_has[better] = True
        # Keep fallback fields in the output buffers until acceptance is known.
        fallback_normals = candidate.normals
        fallback_discontinuity = candidate.discontinuity_strength
        fallback_radius = candidate.selected_radius
        fallback_support = candidate.neighbor_support
        fallback_confidence = candidate.confidence
        # Store the best candidate in temporary output slots; accepted pixels are overwritten below.
        selected_normals[better] = fallback_normals[better]
        selected_confidence[better] = fallback_confidence[better]
        selected_discontinuity[better] = fallback_discontinuity[better]
        selected_radius[better] = fallback_radius[better]
        selected_support[better] = fallback_support[better]
        take = (~accepted) & candidate.normal_valid_mask & (candidate.confidence >= config.multi_scale_acceptance)
        selected_normals[take] = candidate.normals[take]
        selected_confidence[take] = candidate.confidence[take]
        selected_discontinuity[take] = candidate.discontinuity_strength[take]
        selected_radius[take] = candidate.selected_radius[take]
        selected_support[take] = candidate.neighbor_support[take]
        selected_valid[take] = True
        accepted[take] = True
    selected_valid[~accepted & best_has] = True
    return NormalResult(selected_normals, selected_valid, selected_confidence, selected_discontinuity, selected_radius, selected_support)


def estimate_normals(
    positions: np.ndarray,
    valid_mask: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    mode: NormalMode | str = NormalMode.EDGE_AWARE,
    config: NormalConfig | None = None,
    input_confidence: np.ndarray | None = None,
) -> NormalResult:
    """Estimate camera-facing normals using baseline, edge-aware, or multi-scale geometry."""

    config = config or NormalConfig()
    mode = NormalMode(mode)
    points, valid = _validate_inputs(positions, valid_mask)
    depth_arr = points[..., 2] if depth is None else np.asarray(depth, dtype=np.float32)
    if depth_arr.shape != points.shape[:2]:
        raise ValueError("depth must have shape (H, W) matching positions")
    depth_arr = np.where(np.isfinite(depth_arr), depth_arr, points[..., 2])
    if mode is NormalMode.BASELINE:
        result = _estimate_at_radius(points, valid, depth_arr, 1, False, config)
    elif mode is NormalMode.EDGE_AWARE:
        result = _estimate_at_radius(points, valid, depth_arr, 1, True, config)
    else:
        result = _select_multiscale_streaming(points, valid, depth_arr, config)
    if input_confidence is not None:
        supplied = np.asarray(input_confidence, dtype=np.float32)
        if supplied.shape != valid.shape:
            raise ValueError("input_confidence must have shape (H, W)")
        clamped_input = np.clip(np.nan_to_num(supplied, nan=0.0), 0.0, 1.0)
        final_confidence = np.where(
            result.normal_valid_mask,
            np.clip(result.confidence * clamped_input, 0.0, 1.0),
            0.0,
        ).astype(np.float32)
        result = NormalResult(
            result.normals,
            result.normal_valid_mask,
            final_confidence,
            result.discontinuity_strength,
            result.selected_radius,
            result.neighbor_support,
        )
    return result


def normals_to_rgb(normals: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Map valid normals to RGB via ``0.5 * (N + 1)``; invalid is magenta."""

    normal_arr = np.asarray(normals, dtype=np.float32)
    if normal_arr.ndim != 3 or normal_arr.shape[-1] != 3:
        raise ValueError("normals must have shape (H, W, 3)")
    valid = np.isfinite(normal_arr).all(axis=-1)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    rgb = np.full(normal_arr.shape, (255, 0, 255), dtype=np.uint8)
    mapped = np.clip((np.nan_to_num(normal_arr, nan=0.0) + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    rgb[valid] = mapped[valid]
    return rgb


def angular_metrics(predicted: np.ndarray, ground_truth: np.ndarray, valid_mask: np.ndarray | None = None) -> AngularMetrics:
    """Report angular normal error in degrees over finite, valid samples."""

    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(ground_truth, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("predicted and ground_truth must have matching shape (H, W, 3)")
    valid = np.isfinite(pred).all(axis=-1) & np.isfinite(truth).all(axis=-1)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    pred_norm = np.linalg.norm(pred, axis=-1)
    truth_norm = np.linalg.norm(truth, axis=-1)
    valid &= (pred_norm > 0) & (truth_norm > 0)
    if not valid.any():
        return AngularMetrics(float("nan"), float("nan"), float("nan"), 0.0, 0)
    dots = np.sum(pred[valid] * truth[valid], axis=-1) / (pred_norm[valid] * truth_norm[valid])
    errors = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return AngularMetrics(float(errors.mean()), float(np.median(errors)), float(np.percentile(errors, 95)), float(valid.mean() * 100.0), int(valid.sum()))


def geometry_from_depth_state(
    depth_state: DepthState,
    camera: CameraModel,
    mode: NormalMode | str = NormalMode.EDGE_AWARE,
    config: NormalConfig | None = None,
) -> GeometryState:
    """Build a Phase 2 ``GeometryState`` without coupling to a depth model."""

    positions, valid = backproject_depth(depth_state.depth, camera, depth_state.scale_mode, depth_state.valid_mask)
    result = estimate_normals(positions, valid, depth_state.depth, mode, config, depth_state.confidence)
    return GeometryState(
        timestamp=depth_state.timestamp,
        source_frame_id=depth_state.source_frame_id,
        depth=depth_state.depth,
        positions_3d=positions,
        valid_mask=valid,
        camera=camera,
        scale_mode=depth_state.scale_mode,
        normals=result.normals,
        confidence=result.confidence,
        normal_valid_mask=result.normal_valid_mask,
        normal_confidence=result.confidence,
        selected_radius=result.selected_radius,
    )
