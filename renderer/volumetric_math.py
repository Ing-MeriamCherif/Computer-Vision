"""Small CPU reference helpers used to test the GLSL volumetric equations.

The interactive renderer performs all image pixels and ray samples in GLSL;
these functions are deterministic one-ray references for unit tests only.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from contracts.render_types import Light
from .config import VolumetricConfig


def reference_scattering_for_ray(
    ray_direction_camera: np.ndarray,
    surface_depth_m: float,
    depth_valid: bool,
    lights: Sequence[Light],
    config: VolumetricConfig,
    *,
    attenuation_k: float = 0.6,
    light_visibility: Sequence[float] = (1.0, 1.0),
) -> np.ndarray:
    """Integrate one view ray to a metric-depth endpoint for tests/reference."""
    result = np.zeros(3, dtype=np.float32)
    if not config.volumetric_enabled or not depth_valid:
        return result
    try:
        depth = float(surface_depth_m)
        ray = np.asarray(ray_direction_camera, dtype=np.float64)
        attenuation_k = float(attenuation_k)
    except (TypeError, ValueError, OverflowError):
        return result
    if (
        ray.shape != (3,)
        or not np.isfinite(ray).all()
        or not math.isfinite(depth)
        or depth <= 0.0
        or not math.isfinite(attenuation_k)
        or attenuation_k < 0.0
    ):
        return result
    ray_norm = float(np.linalg.norm(ray))
    if ray_norm <= 1e-9:
        return result
    ray /= ray_norm
    if ray[2] <= 1e-9:
        return result

    end_distance = depth / float(ray[2])
    step_m = end_distance / config.volumetric_samples
    step_decay = 1.0
    for sample_index in range(config.volumetric_samples):
        sample_position = ray * ((sample_index + 0.5) * step_m)
        for light_index, light in enumerate(lights[:2]):
            try:
                position = np.asarray(light.position_camera_m, dtype=np.float64)
                color = np.asarray(light.color_rgb, dtype=np.float64)
                light_intensity = float(light.intensity)
                active = bool(light.active)
                visibility = float(light_visibility[light_index])
            except (TypeError, ValueError, OverflowError, IndexError):
                continue
            if (
                not active
                or position.shape != (3,)
                or color.shape != (3,)
                or not np.isfinite(position).all()
                or not np.isfinite(color).all()
                or np.any(color < 0.0)
                or np.any(color > 1.0)
                or not math.isfinite(light_intensity)
                or light_intensity <= 0.0
                or not math.isfinite(visibility)
            ):
                continue
            light_distance_sq = float(np.dot(position - sample_position, position - sample_position))
            attenuation = 1.0 / (1.0 + attenuation_k * light_distance_sq)
            result += (
                color.astype(np.float32)
                * np.float32(light_intensity * attenuation * np.clip(visibility, 0.0, 1.0))
                * np.float32(config.volumetric_density * step_m * config.volumetric_intensity * step_decay)
            )
        step_decay *= config.volumetric_decay

    if not np.isfinite(result).all():
        return np.zeros(3, dtype=np.float32)
    return result


def composite_scattering_reference(
    surface_linear: np.ndarray,
    volume_linear: np.ndarray,
    *,
    enabled: bool,
) -> np.ndarray:
    """Reference the shader's disable passthrough and highlight-safe addition."""
    surface = np.asarray(surface_linear, dtype=np.float32)
    if surface.shape != (3,):
        raise ValueError("surface_linear must have shape (3,)")
    if not enabled:
        return surface.copy()
    volume = np.asarray(volume_linear, dtype=np.float32)
    if volume.shape != (3,):
        raise ValueError("volume_linear must have shape (3,)")
    volume = np.maximum(np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    headroom = np.maximum(1.0 - np.clip(surface, 0.0, 1.0), 0.0)
    return surface + np.minimum(volume, headroom * 0.8)
