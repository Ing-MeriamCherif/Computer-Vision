"""Small CPU reference helpers for screen-space shadow validation.

The real shadow pass runs in ``renderer/shaders/shadow.frag``. These helpers
mirror its depth test so projection, unknown-depth, and deterministic synthetic
cases can be checked without creating an OpenGL context.
"""

from __future__ import annotations

import math

import numpy as np

from .projection import project_camera_point, reconstruct_camera_point


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
