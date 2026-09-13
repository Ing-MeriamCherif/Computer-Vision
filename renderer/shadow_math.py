"""Small CPU reference helpers for screen-space shadow validation.

The real shadow pass runs in ``renderer/shaders/shadow.frag``. These helpers
mirror its depth test so projection, unknown-depth, and deterministic synthetic
cases can be checked without creating an OpenGL context.
"""

from __future__ import annotations

import math

import numpy as np

from .projection import project_camera_point, reconstruct_camera_point


def compute_depth_edge_mask(
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    *,
    threshold_m: float,
) -> np.ndarray:
    """Mark both pixels adjacent to a reliable horizontal/vertical depth edge."""
    depth = np.asarray(depth_m)
    valid = np.asarray(valid_mask)
    if depth.ndim != 2 or valid.shape != depth.shape:
        raise ValueError("depth_m and valid_mask must be same-size HxW arrays")
    if not math.isfinite(float(threshold_m)) or threshold_m <= 0.0:
        raise ValueError("threshold_m must be finite and > 0")

    reliable = valid.astype(bool) & np.isfinite(depth) & (depth > 0.0)
    edges = np.zeros(depth.shape, dtype=np.bool_)
    horizontal = reliable[:, :-1] & reliable[:, 1:] & (
        np.abs(depth[:, :-1] - depth[:, 1:]) >= threshold_m
    )
    vertical = reliable[:-1, :] & reliable[1:, :] & (
        np.abs(depth[:-1, :] - depth[1:, :]) >= threshold_m
    )
    edges[:, :-1] |= horizontal
    edges[:, 1:] |= horizontal
    edges[:-1, :] |= vertical
    edges[1:, :] |= vertical
    return edges


def shadow_soft_offsets(samples: int, radius: float) -> np.ndarray:
    """Return the same symmetric cross/ring offsets used by the GLSL filter."""
    if samples not in (4, 8):
        raise ValueError("soft samples must be 4 or 8")
    if not math.isfinite(float(radius)) or radius <= 0.0:
        raise ValueError("radius must be finite and > 0")
    diagonal = 1.0 / math.sqrt(2.0)
    offsets = np.array(
        [
            [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
            [diagonal, diagonal], [-diagonal, -diagonal],
            [diagonal, -diagonal], [-diagonal, diagonal],
        ],
        dtype=np.float32,
    )
    return offsets[:samples] * np.float32(radius)


def filter_shadow_samples_reference(
    visibility_samples: np.ndarray,
    sample_depth_m: np.ndarray,
    *,
    receiver_depth_m: float,
    depth_edge_threshold_m: float,
    edge_aware: bool = True,
) -> float:
    """CPU reference for the compositor's depth-weighted shadow-mask average."""
    visibility = np.asarray(visibility_samples, dtype=np.float64)
    depths = np.asarray(sample_depth_m, dtype=np.float64)
    if visibility.ndim != 1 or depths.shape != visibility.shape or visibility.size == 0:
        raise ValueError("visibility_samples and sample_depth_m must be same-size non-empty vectors")
    if not np.isfinite(visibility).all() or np.any((visibility < 0.0) | (visibility > 1.0)):
        raise ValueError("visibility samples must be finite and in [0, 1]")
    if not math.isfinite(float(depth_edge_threshold_m)) or depth_edge_threshold_m <= 0.0:
        raise ValueError("depth_edge_threshold_m must be finite and > 0")

    weights = np.ones(visibility.shape, dtype=np.float64)
    if edge_aware:
        receiver_z = float(receiver_depth_m)
        if not math.isfinite(receiver_z) or receiver_z <= 0.0:
            return 1.0
        reliable = np.isfinite(depths) & (depths > 0.0)
        weights.fill(0.0)
        weights[reliable] = np.maximum(
            0.0,
            1.0 - np.abs(depths[reliable] - receiver_z) / depth_edge_threshold_m,
        )
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-12:
        return 1.0
    return float(np.dot(visibility, weights) / weight_sum)


def sample_is_occluded(
    scene_depth_m: float,
    depth_valid: bool,
    ray_depth_m: float,
    *,
    shadow_bias_m: float,
    shadow_thickness_m: float,
) -> bool:
    """Return whether one reliable camera-depth sample blocks a ray sample.

    A blocker is accepted only when the ray sample lies behind the visible
    surface by more than the self-shadow bias, but no farther than the
    configured thickness tolerance. Invalid, non-positive, or non-finite
    depths are unknown and therefore unblocked.
    """
    if not depth_valid:
        return False
    scene_z = float(scene_depth_m)
    ray_z = float(ray_depth_m)
    if not math.isfinite(scene_z) or not math.isfinite(ray_z) or scene_z <= 0.0 or ray_z <= 0.0:
        return False
    difference = ray_z - scene_z
    return shadow_bias_m < difference <= shadow_thickness_m


def trace_shadow_visibility(
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    pixel_x: float,
    pixel_y: float,
    light_position_camera_m: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    shadow_steps: int = 12,
    shadow_bias_m: float = 0.015,
    shadow_thickness_m: float = 0.15,
    ray_start_offset: float = 0.01,
) -> float:
    """Trace one camera-space light ray against the camera-visible depth map.

    Returns 1 for lit, unknown, invalid, or off-screen paths, and 0 for an
    occluded point. Sampling uses nearest-pixel lookup to mirror the GPU depth
    texture. This is a reference/test helper, not a per-frame renderer path.
    """
    depth = np.asarray(depth_m)
    valid = np.asarray(valid_mask)
    light = np.asarray(light_position_camera_m, dtype=np.float64)
    if depth.ndim != 2 or valid.shape != depth.shape:
        raise ValueError("depth_m and valid_mask must be same-size HxW arrays")
    if light.shape != (3,):
        raise ValueError("light_position_camera_m must have shape (3,)")
    if not np.isfinite(light).all():
        return 1.0
    if isinstance(shadow_steps, bool) or not isinstance(shadow_steps, int) or not 1 <= shadow_steps <= 64:
        raise ValueError("shadow_steps must be an integer between 1 and 64")
    if not math.isfinite(float(pixel_x)) or not math.isfinite(float(pixel_y)):
        return 1.0
    for name, value in (
        ("shadow_bias_m", shadow_bias_m),
        ("shadow_thickness_m", shadow_thickness_m),
        ("ray_start_offset", ray_start_offset),
    ):
        if not math.isfinite(float(value)) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")
    if shadow_thickness_m <= shadow_bias_m:
        raise ValueError("shadow_thickness_m must be greater than shadow_bias_m")

    height, width = depth.shape
    receiver_x = int(math.floor(float(pixel_x) + 0.5))
    receiver_y = int(math.floor(float(pixel_y) + 0.5))
    if receiver_x < 0 or receiver_x >= width or receiver_y < 0 or receiver_y >= height:
        return 1.0
    receiver_z = float(depth[receiver_y, receiver_x])
    if not bool(valid[receiver_y, receiver_x]) or not math.isfinite(receiver_z) or receiver_z <= 0.0:
        return 1.0

    point = reconstruct_camera_point(
        float(pixel_x), float(pixel_y), receiver_z, fx=fx, fy=fy, cx=cx, cy=cy
    )
    ray = light - point
    ray_length = float(np.linalg.norm(ray))
    if not math.isfinite(ray_length) or ray_length <= 1e-8:
        return 1.0
    start_t = min(max(float(ray_start_offset) / ray_length, 0.0), 0.99)

    for step_index in range(shadow_steps):
        t = start_t + (1.0 - start_t) * (step_index + 1) / (shadow_steps + 1)
        sample = point + t * ray
        if not np.isfinite(sample).all() or sample[2] <= 0.0:
            return 1.0
        try:
            projected_x, projected_y = project_camera_point(
                sample, fx=fx, fy=fy, cx=cx, cy=cy
            )
        except ValueError:
            return 1.0
        if projected_x < 0.0 or projected_x > width - 1 or projected_y < 0.0 or projected_y > height - 1:
            return 1.0

        sample_x = int(math.floor(projected_x + 0.5))
        sample_y = int(math.floor(projected_y + 0.5))
        scene_z = float(depth[sample_y, sample_x])
        if not bool(valid[sample_y, sample_x]) or not math.isfinite(scene_z) or scene_z <= 0.0:
            # Unknown space is conservative: this ray does not darken a pixel.
            return 1.0
        if sample_is_occluded(
            scene_z,
            True,
            float(sample[2]),
            shadow_bias_m=shadow_bias_m,
            shadow_thickness_m=shadow_thickness_m,
        ):
            return 0.0
    return 1.0
