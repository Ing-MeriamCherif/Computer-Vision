"""Synthetic geometry generators, plane fitting, and lightweight PLY export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import CameraModel


def fronto_parallel_plane(camera: CameraModel, z: float = 2.0) -> np.ndarray:
    if z <= 0:
        raise ValueError("z must be positive")
    return np.full((camera.height, camera.width), z, dtype=np.float32)


def tilted_plane(camera: CameraModel, z0: float = 2.0, slope_x: float = 0.1, slope_y: float = -0.05) -> np.ndarray:
    v, u = np.indices((camera.height, camera.width), dtype=np.float64)
    depth = z0 + slope_x * (u - camera.cx) / camera.fx + slope_y * (v - camera.cy) / camera.fy
    if np.nanmin(depth) <= 0:
        raise ValueError("tilted plane reaches non-positive depth")
    return depth.astype(np.float32)


def step_depth(camera: CameraModel, foreground: float = 1.0, background: float = 3.0, split: float | None = None) -> np.ndarray:
    split = camera.width / 2 if split is None else split
    depth = np.full((camera.height, camera.width), background, dtype=np.float32)
    depth[:, np.arange(camera.width) < split] = foreground
    return depth


@dataclass(frozen=True, slots=True)
class PlaneFit:
    coefficients: np.ndarray
    rms_residual: float
    max_abs_residual: float
    point_count: int


def fit_plane(points: np.ndarray, valid_mask: np.ndarray | None = None) -> PlaneFit:
    points_arr = np.asarray(points, dtype=np.float64)
    if points_arr.ndim != 3 or points_arr.shape[-1] != 3:
        raise ValueError("points must have shape (H, W, 3)")
    valid = np.isfinite(points_arr).all(axis=-1)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    samples = points_arr[valid]
    if len(samples) < 3:
        raise ValueError("at least three finite points are required")
    centered = samples - samples.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    coefficients = np.r_[normal, -normal @ samples.mean(axis=0)]
    residual = samples @ coefficients[:3] + coefficients[3]
    return PlaneFit(coefficients, float(np.sqrt(np.mean(residual**2))), float(np.max(np.abs(residual))), len(samples))


def export_ply(path: str | Path, points: np.ndarray, valid_mask: np.ndarray | None = None) -> None:
    points_arr = np.asarray(points, dtype=np.float32)
    if points_arr.ndim != 3 or points_arr.shape[-1] != 3:
        raise ValueError("points must have shape (H, W, 3)")
    valid = np.isfinite(points_arr).all(axis=-1)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    samples = points_arr[valid]
    lines = ["ply", "format ascii 1.0", f"element vertex {len(samples)}", "property float x", "property float y", "property float z", "end_header"]
    lines.extend(f"{x:.8g} {y:.8g} {z:.8g}" for x, y, z in samples)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
