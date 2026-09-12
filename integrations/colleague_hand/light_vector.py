"""Person 1 light-vector math (cdc 4.1 + maybe-to-test.md).

Phase-1a: z_c is FIXED stub (no depth model yet).
Phase-1b: caller may pass measured depth_m to override.
Contract matches Person 4 `Light` (camera coords: +X right, +Y down, +Z fwd, meters).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import Intrinsics


@dataclass
class LightState:
    position_camera_m: np.ndarray  # (3,)
    intensity: float
    color_rgb: np.ndarray          # (3,) 0..1
    active: bool
    confidence: float
    timestamp_s: float


def backproject(u: float, v: float, z_c: float, intr: Intrinsics) -> np.ndarray:
    x_c = z_c * (u - intr.cx) / intr.fx
    y_c = z_c * (v - intr.cy) / intr.fy
    return np.array([x_c, y_c, z_c], dtype=np.float64)


def compute_intensity(light_pos: np.ndarray, d_ref: float = 0.5, i0: float = 1.0) -> float:
    d = float(np.linalg.norm(light_pos))
    return float(i0 / (1.0 + (d / d_ref) ** 2))


def depth_from_palm_size(fx: float, palm_px: float | None,
                         real_m: float = 0.085,
                         z_min: float = 0.2, z_max: float = 3.0,
                         fallback: float = 0.5) -> tuple[float, bool]:
    """Proxy depth from apparent palm width: z = fx * real / pixels.

    Returns (z_m, estimated_ok). Falls back to fixed z when no size.
    Clamped to [z_min, z_max]. Same idea as maybe-to-test.md hand-size
    alternative; depth map sampling replaces it in Phase-2.
    """
    if palm_px is None or palm_px < 8.0:
        return float(fallback), False
    z = float(fx * real_m / palm_px)
    return float(min(max(z, z_min), z_max)), True


def palm_to_light(
    u: float,
    v: float,
    intr: Intrinsics,
    timestamp_s: float,
    fixed_z: float = 0.5,
    depth_m: float | None = None,
    d_ref: float = 0.5,
    i0: float = 1.0,
    color_rgb: tuple[float, float, float] = (1.0, 0.85, 0.7),
    confidence: float = 1.0,
) -> LightState:
    z_c = float(depth_m) if depth_m is not None and depth_m > 0 else float(fixed_z)
    pos = backproject(u, v, z_c, intr)
    intensity = compute_intensity(pos, d_ref=d_ref, i0=i0)
    return LightState(
        position_camera_m=pos,
        intensity=intensity,
        color_rgb=np.array(color_rgb, dtype=np.float64),
        active=True,
        confidence=float(confidence),
        timestamp_s=float(timestamp_s),
    )


def light_direction_to_pixel(light_pos: np.ndarray, pixel_pos: np.ndarray) -> np.ndarray:
    v = np.asarray(light_pos, dtype=np.float64) - np.asarray(pixel_pos, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return v / n
