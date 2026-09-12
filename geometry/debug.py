"""Synthetic geometry generators, plane fitting, and lightweight PLY export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import CameraModel


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


def fronto_parallel_plane(camera: CameraModel, z: float = 2.0) -> np.ndarray:
    if z <= 0:
        raise ValueError("z must be positive")
    return np.full((camera.height, camera.width), z, dtype=np.float32)


def exact_plane_depth(
    camera: CameraModel,
    normal: ArrayLike = (0.1, -0.05, 1.0),
    distance: float = -2.0,
    eps: float = 1e-7,
) -> np.ndarray:
    """Generate an exact synthetic Z-depth image via ray-plane intersection.

    For a camera ray ``r(u, v) = [(u - cx)/fx, (v - cy)/fy, 1]^T`` and plane
    ``n . X + d = 0``, substituting ``X = t * r`` yields ``t = -d / (n . r)``.
    Since the camera ray is Z-normalized (r_z = 1), ``t`` is the exact Z-depth.

    The normal vector ``n`` is normalized to unit length and ``d`` is scaled
    accordingly, preserving the plane equation ``n . X + d = 0``.

    Intersections where ``|n . r| <= eps`` (ray nearly parallel to plane) or
    where the resulting depth ``t <= 0`` (intersection behind camera or at
    camera center) are invalidated and set to NaN.

    Parameters
    ----------
    camera : CameraModel
        Camera model defining resolution and intrinsics.
    normal : ArrayLike
        Plane normal vector (nx, ny, nz). Normalized internally to unit length.
    distance : float
        Plane equation offset d in n . X + d = 0.
    eps : float
        Threshold for detecting degenerate / near-parallel ray-plane intersections.

    Returns
    -------
    np.ndarray
        Array of shape (camera.height, camera.width) with float32 depth values,
        where non-positive or near-parallel intersections are set to NaN.
    """
    n_arr = np.asarray(normal, dtype=np.float64)
    if n_arr.shape != (3,):
        raise ValueError(f"plane normal must have 3 elements, got shape {n_arr.shape}")
    norm = float(np.linalg.norm(n_arr))
    if not np.isfinite(norm) or norm <= eps:
        raise ValueError("plane normal must have finite, non-zero magnitude")
    n_unit = n_arr / norm
    d_scaled = float(distance) / norm

    v, u = np.indices((camera.height, camera.width), dtype=np.float64)
    rays = camera.pixel_to_ray(u, v)
    denom = rays @ n_unit

    invalid = np.abs(denom) <= eps
    safe_denom = np.where(invalid, 1.0, denom)
    depth = -d_scaled / safe_denom
    invalid |= (depth <= 0) | ~np.isfinite(depth)

    return np.where(invalid, np.nan, depth).astype(np.float32)


def tilted_plane(
    camera: CameraModel,
    z0: float = 2.0,
    slope_x: float = 0.1,
    slope_y: float = -0.05,
    *,
    normal: ArrayLike | None = None,
    distance: float | None = None,
    eps: float = 1e-7,
) -> np.ndarray:
    """Generate an exact synthetic tilted plane depth image.

    This replaces the approximate formulation with exact ray-plane intersection.
    If ``normal`` and ``distance`` are provided, they define the plane
    ``n . X + d = 0``. Otherwise, the tangent plane at (0, 0, z0) with
    slopes ``slope_x`` and ``slope_y`` is constructed with normal
    ``(-slope_x, -slope_y, 1.0)`` and distance ``-z0``.

    Raises ValueError if all or any points in the field of view have
    non-positive depth or invalid intersections.
    """
    if normal is not None or distance is not None:
        if normal is None or distance is None:
            raise ValueError("both normal and distance must be provided together")
        n_param = normal
        d_param = distance
    else:
        n_param = (-slope_x, -slope_y, 1.0)
        d_param = -z0

    depth = exact_plane_depth(camera, normal=n_param, distance=d_param, eps=eps)
    if not np.isfinite(depth).all():
        raise ValueError("tilted plane produces non-positive depth or invalid intersections in camera FOV")
    return depth


def step_depth(camera: CameraModel, foreground: float = 1.0, background: float = 3.0, split: float | None = None) -> np.ndarray:
    split = camera.width / 2 if split is None else split
    depth = np.full((camera.height, camera.width), background, dtype=np.float32)
    depth[:, np.arange(camera.width) < split] = foreground
    return depth


def sphere_depth(
    camera: CameraModel,
    center: ArrayLike = (0.0, 0.0, 3.0),
    radius: float = 1.0,
) -> np.ndarray:
    """Generate nearest-positive Z-depth intersections with an analytic sphere."""

    center_arr = np.asarray(center, dtype=np.float64)
    if center_arr.shape != (3,) or not np.isfinite(center_arr).all():
        raise ValueError("sphere center must have three finite values")
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("sphere radius must be positive and finite")
    v, u = np.indices((camera.height, camera.width), dtype=np.float64)
    rays = camera.pixel_to_ray(u, v)
    a = np.sum(rays * rays, axis=-1)
    b = -2.0 * np.sum(rays * center_arr, axis=-1)
    c = float(np.dot(center_arr, center_arr) - radius * radius)
    discriminant = b * b - 4.0 * a * c
    has_hit = discriminant >= 0
    root = np.sqrt(np.maximum(discriminant, 0.0))
    near = (-b - root) / (2.0 * a)
    far = (-b + root) / (2.0 * a)
    depth = np.where((near > 0) & has_hit, near, np.where((far > 0) & has_hit, far, np.nan))
    return depth.astype(np.float32)


def sphere_normals(points: np.ndarray, center: ArrayLike = (0.0, 0.0, 3.0), radius: float = 1.0) -> np.ndarray:
    """Return camera-facing analytic sphere normals for reconstructed points."""

    points_arr = np.asarray(points, dtype=np.float64)
    center_arr = np.asarray(center, dtype=np.float64)
    if points_arr.ndim != 3 or points_arr.shape[-1] != 3 or center_arr.shape != (3,):
        raise ValueError("points must be (H, W, 3) and center must have three values")
    normals = points_arr - center_arr
    lengths = np.linalg.norm(normals, axis=-1)
    valid = np.isfinite(normals).all(axis=-1) & (lengths > 0)
    normals = normals / np.where(valid, lengths, 1.0)[..., None]
    normals = np.where((np.sum(normals * points_arr, axis=-1) > 0)[..., None], -normals, normals)
    return np.where(valid[..., None], normals, np.nan).astype(np.float32)


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
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
