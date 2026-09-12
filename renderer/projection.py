"""CPU reference math for the camera-space projection used by GLSL."""

from __future__ import annotations

import math

import numpy as np


def reconstruct_camera_point(
    u: float,
    v: float,
    z_m: float,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Back-project a pixel/depth sample to [X,Y,Z] in camera meters."""
    values = (u, v, z_m, fx, fy, cx, cy)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("pixel, depth, and intrinsics must be finite")
    if z_m <= 0.0:
        raise ValueError("depth must be > 0 meters")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("fx and fy must be > 0")
    return np.array(((u - cx) * z_m / fx, (v - cy) * z_m / fy, z_m), dtype=np.float64)


def project_camera_point(
    point_camera_m: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float]:
    """Project [X,Y,Z] camera meters to top-left-origin pixel coordinates."""
    point = np.asarray(point_camera_m, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("point_camera_m must be a finite length-3 vector")
    if point[2] <= 0.0:
        raise ValueError("camera-space Z must be > 0")
    if not all(math.isfinite(float(value)) for value in (fx, fy, cx, cy)):
        raise ValueError("intrinsics must be finite")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("fx and fy must be > 0")
    return fx * point[0] / point[2] + cx, fy * point[1] / point[2] + cy
