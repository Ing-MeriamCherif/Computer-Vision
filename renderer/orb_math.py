"""CPU reference math for projecting and depth-testing virtual light orbs."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LightOrbProjection:
    """Projected orb center and depth comparison for debug/validation."""

    u_px: float | None
    v_px: float | None
    light_z_m: float | None
    scene_z_m: float | None
    in_frame: bool
    depth_valid: bool
    occluded: bool
    visible: bool
    reason: str


def evaluate_light_orb(
    position_camera_m: np.ndarray,
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    occlusion_bias_m: float = 0.02,
    active: bool = True,
) -> LightOrbProjection:
    """Project a camera-space light and compare its Z with nearest scene depth.

    Pixel coordinates use a top-left origin and camera coordinates use
    +X right, +Y down, +Z forward. Unknown depth is conservatively treated as
    visible; it cannot prove that geometry occludes the light.
    """
    hidden = LightOrbProjection(None, None, None, None, False, False, False, False, "invalid")
    if not isinstance(active, (bool, np.bool_)):
        return hidden
    try:
        position = np.asarray(position_camera_m, dtype=np.float64)
        depth = np.asarray(depth_m)
        valid = np.asarray(valid_mask)
        intrinsics = tuple(float(value) for value in (fx, fy, cx, cy, occlusion_bias_m))
    except (TypeError, ValueError, OverflowError):
        return hidden
    if (
        position.shape != (3,)
        or depth.ndim != 2
        or depth.size == 0
        or valid.shape != depth.shape
        or valid.dtype != np.bool_
        or not np.issubdtype(depth.dtype, np.number)
        or not np.isfinite(position).all()
        or not all(math.isfinite(value) for value in intrinsics)
    ):
        return hidden
    fx_value, fy_value, cx_value, cy_value, bias = intrinsics
    if fx_value <= 0.0 or fy_value <= 0.0 or bias < 0.0:
        return hidden

    light_z = float(position[2])
    if light_z <= 0.0:
        return LightOrbProjection(None, None, light_z, None, False, False, False, False, "behind-camera")

    u_px = fx_value * float(position[0]) / light_z + cx_value
    v_px = fy_value * float(position[1]) / light_z + cy_value
    if not math.isfinite(u_px) or not math.isfinite(v_px):
        return LightOrbProjection(None, None, light_z, None, False, False, False, False, "invalid-projection")

    height, width = depth.shape
    if not (0.0 <= u_px < width and 0.0 <= v_px < height):
        return LightOrbProjection(u_px, v_px, light_z, None, False, False, False, False, "off-screen")

    # Match GL_NEAREST sampling at uv=((u + .5)/width, (v + .5)/height).
    sample_x = min(width - 1, max(0, int(math.floor(u_px + 0.5))))
    sample_y = min(height - 1, max(0, int(math.floor(v_px + 0.5))))
    scene_z = float(depth[sample_y, sample_x])
    depth_valid = bool(valid[sample_y, sample_x]) and math.isfinite(scene_z) and scene_z > 0.0
    occluded = depth_valid and scene_z < light_z - bias
    visible = bool(active) and not occluded
    if not active:
        reason = "inactive"
    elif occluded:
        reason = "occluded"
    elif not depth_valid:
        reason = "unknown-depth-visible"
    else:
        reason = "visible"
    return LightOrbProjection(
        u_px,
        v_px,
        light_z,
        scene_z if depth_valid else None,
        True,
        depth_valid,
        bool(occluded),
        visible,
        reason,
    )
