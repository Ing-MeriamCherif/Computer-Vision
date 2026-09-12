"""GPU-friendly (and CPU-safe) interactive lighting over ``GeometryState``."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .camera import CameraModel
from .state import GeometryState


@dataclass(slots=True)
class LightState:
    position_camera_m: np.ndarray
    intensity: float
    color_rgb: np.ndarray
    confidence: float
    source_hand: int = 0
    timestamp: float = 0.0


def _bilinear(depth: np.ndarray, u: float, v: float) -> float:
    h, w = depth.shape
    x = float(np.clip(u, 0, max(w - 1, 0)))
    y = float(np.clip(v, 0, max(h - 1, 0)))
    x0, y0 = min(int(x), w - 1), min(int(y), h - 1)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    return float(
        depth[y0, x0] * (1 - dx) * (1 - dy)
        + depth[y0, x1] * dx * (1 - dy)
        + depth[y1, x0] * (1 - dx) * dy
        + depth[y1, x1] * dx * dy
    )


def sample_depth(depth: np.ndarray, valid: np.ndarray | None, u: float, v: float, radius: int = 2) -> tuple[float, float]:
    """Sample depth at a palm, returning ``(z, reliability)``.

    Bilinear sampling is used in the normal case.  A 5x5 median is used when
    the footprint crosses an invalid/discontinuous region, matching the
    challenge hand-to-depth contract.
    """

    h, w = depth.shape
    x, y = int(np.clip(round(u), 0, w - 1)), int(np.clip(round(v), 0, h - 1))
    mask = np.isfinite(depth) & (depth > 1e-6)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    if mask[y, x]:
        z = _bilinear(depth, u, v)
        if np.isfinite(z) and z > 1e-6:
            return z, 1.0
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    values = depth[y0:y1, x0:x1][mask[y0:y1, x0:x1]]
    if values.size == 0:
        return 0.0, 0.0
    return float(np.median(values)), float(min(1.0, values.size / ((2 * radius + 1) ** 2)))


def light_from_palm(
    geometry: GeometryState,
    palm_uv: tuple[float, float],
    confidence: float,
    *,
    color_rgb: tuple[float, float, float] = (1.0, 0.78, 0.48),
    intensity: float = 1.4,
    d_ref: float = 0.7,
    source_hand: int = 0,
    timestamp: float | None = None,
) -> LightState | None:
    """Back-project a detected palm using the *real* depth map."""

    z, sample_conf = sample_depth(geometry.depth, geometry.valid_mask, *palm_uv)
    if z <= 0:
        return None
    position = geometry.camera.unproject(palm_uv[0], palm_uv[1], z).astype(np.float32)
    distance = float(np.linalg.norm(position))
    falloff = 1.0 / (1.0 + (distance / max(d_ref, 1e-3)) ** 2)
    return LightState(
        position_camera_m=np.asarray(position, dtype=np.float32),
        intensity=float(intensity * falloff),
        color_rgb=np.asarray(color_rgb, dtype=np.float32),
        confidence=float(np.clip(confidence, 0, 1) * sample_conf),
        source_hand=source_hand,
        timestamp=time.time() if timestamp is None else float(timestamp),
    )


def _shadow_factor(geometry: GeometryState, light: LightState, *, steps: int = 4) -> np.ndarray:
    """Low-cost screen-space visibility test along the light-to-pixel segment."""

    h, w = geometry.depth.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    light_uv = geometry.camera.project(light.position_camera_m).astype(np.float32)
    depth = geometry.depth.astype(np.float32)
    visible = np.ones((h, w), dtype=np.float32)
    for fraction in np.linspace(0.2, 0.8, max(1, steps), dtype=np.float32):
        su = np.rint(xx + (light_uv[0] - xx) * fraction).astype(np.int32).clip(0, w - 1)
        sv = np.rint(yy + (light_uv[1] - yy) * fraction).astype(np.int32).clip(0, h - 1)
        sampled = depth[sv, su]
        expected = depth + (float(light.position_camera_m[2]) - depth) * fraction
        occluded = np.isfinite(sampled) & (sampled + 0.02 < expected)
        visible[occluded] *= 0.72
    return visible


def shade_geometry(
    rgb: np.ndarray,
    geometry: GeometryState,
    lights: list[LightState] | tuple[LightState, ...],
    *,
    ambient: float = 0.18,
    specular_strength: float = 0.28,
    shininess: float = 48.0,
    shadows: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply diffuse + Blinn-Phong specular lighting and screen-space shadows."""

    started = time.perf_counter()
    image = np.asarray(rgb, dtype=np.float32)[..., :3] / 255.0
    if not lights or geometry.normals is None:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8), {"lighting_ms": 0.0, "lights": 0.0}
    points = np.asarray(geometry.positions_3d, dtype=np.float32)
    normals = np.nan_to_num(np.asarray(geometry.normals, dtype=np.float32), nan=0.0)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-6)
    valid = np.asarray(geometry.valid_mask, dtype=bool)
    confidence = np.asarray(geometry.confidence if geometry.confidence is not None else valid, dtype=np.float32)
    view = -points
    view /= np.maximum(np.linalg.norm(view, axis=-1, keepdims=True), 1e-6)
    lit = image * float(ambient)
    for light in lights[:2]:
        delta = light.position_camera_m.reshape(1, 1, 3) - points
        distance = np.linalg.norm(delta, axis=-1, keepdims=True)
        direction = delta / np.maximum(distance, 1e-6)
        diffuse = np.maximum(np.sum(normals * direction, axis=-1), 0.0)
        halfway = direction + view
        halfway /= np.maximum(np.linalg.norm(halfway, axis=-1, keepdims=True), 1e-6)
        specular = np.maximum(np.sum(normals * halfway, axis=-1), 0.0) ** float(shininess)
        attenuation = float(light.intensity) / (1.0 + distance[..., 0] ** 2)
        visibility = _shadow_factor(geometry, light) if shadows else 1.0
        contribution = (diffuse * 0.92 + specular * specular_strength) * attenuation * visibility * float(light.confidence)
        lit += contribution[..., None] * light.color_rgb.reshape(1, 1, 3)
    lit = np.where(valid[..., None], lit, image * 0.1)
    lit *= np.clip(0.65 + 0.35 * confidence[..., None], 0.0, 1.0)
    elapsed = (time.perf_counter() - started) * 1000.0
    return np.clip(lit * 255.0, 0, 255).astype(np.uint8), {"lighting_ms": elapsed, "lights": float(min(len(lights), 2))}
