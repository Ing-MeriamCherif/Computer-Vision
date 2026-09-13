"""GPU-friendly (and CPU-safe) interactive lighting over ``GeometryState``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import time

import numpy as np

from .camera import CameraModel
from .state import GeometryState


@dataclass(slots=True)
class LightState:
    position_camera: np.ndarray | None = None
    intensity: float = 1.0
    color_rgb: np.ndarray | None = None
    confidence: float = 1.0
    source_hand: int = 0
    timestamp: float = 0.0
    enabled: bool = True
    light_id: int | str = 0
    position_camera_m: np.ndarray | None = None
    range_m: float = 1.0
    source_radius_m: float = 0.025

    def __post_init__(self) -> None:
        if self.position_camera is not None:
            pos = np.asarray(self.position_camera, dtype=np.float32)
            object.__setattr__(self, "position_camera", pos)
            object.__setattr__(self, "position_camera_m", pos)
        elif self.position_camera_m is not None:
            pos = np.asarray(self.position_camera_m, dtype=np.float32)
            object.__setattr__(self, "position_camera", pos)
            object.__setattr__(self, "position_camera_m", pos)
        else:
            pos = np.zeros(3, dtype=np.float32)
            object.__setattr__(self, "position_camera", pos)
            object.__setattr__(self, "position_camera_m", pos)
        if self.color_rgb is None:
            object.__setattr__(self, "color_rgb", np.array([1.0, 0.85, 0.6], dtype=np.float32))
        else:
            object.__setattr__(self, "color_rgb", np.asarray(self.color_rgb, dtype=np.float32))


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

    Bilinear sampling is used in the normal case. A 5x5 median is used when
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


def project_light_orb(
    camera: CameraModel,
    light: LightState,
    *,
    source_radius_m: float | None = None,
) -> tuple[float, float, float] | None:
    """Return the pinhole projection and screen radius for a rendered light."""
    position = light.position_camera
    if not light.enabled or position is None or not np.isfinite(position).all() or position[2] <= 0:
        return None
    uv = camera.project(position)
    if not np.isfinite(uv).all():
        return None
    radius_m = light.source_radius_m if source_radius_m is None else float(source_radius_m)
    radius_px = float(np.clip(camera.fx * radius_m / float(position[2]), 3.5, 14.0))
    return float(uv[0]), float(uv[1]), radius_px


@lru_cache(maxsize=24)
def _light_orb_masks(radius_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build reusable, small radial masks for the halo, inner glow, and core."""
    extent = max(5, int(math.ceil(radius_px * 3.0)))
    yy, xx = np.mgrid[-extent:extent + 1, -extent:extent + 1].astype(np.float32)
    distance = np.sqrt(xx * xx + yy * yy)
    broad_glow = np.exp(-0.5 * (distance / max(radius_px * 1.35, 1.0)) ** 2)
    inner = np.exp(-0.5 * (distance / max(radius_px * 0.58, 1.0)) ** 2)
    core = np.exp(-0.5 * (distance / max(radius_px * 0.23, 1.0)) ** 2)
    halo = np.maximum(broad_glow - inner, 0.0)
    return halo, inner, core


def render_light_orbs(
    rgb: np.ndarray,
    camera: CameraModel,
    lights: list[LightState] | tuple[LightState, ...],
    *,
    depth: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    depth_aware: bool = True,
) -> np.ndarray:
    """Composite visible emitters projected from the exact lighting states.

    Work is limited to each orb's small bounding box; reusable radial masks keep
    the per-frame overlay inexpensive. The white-hot core remains visible when
    depth indicates that the halo is behind a nearer surface.
    """
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("rgb must have shape (height, width, 3+)")
    height, width = image.shape[:2]
    result: np.ndarray | None = None

    for light in lights:
        projected = project_light_orb(camera, light)
        if projected is None or light.confidence <= 0 or light.intensity <= 0:
            continue
        u, v, radius = projected
        cx, cy = int(round(u)), int(round(v))
        radius_i = max(4, int(round(radius)))
        halo, inner, core = _light_orb_masks(radius_i)
        extent = halo.shape[0] // 2
        left, top = cx - extent, cy - extent
        right, bottom = cx + extent + 1, cy + extent + 1
        x0, y0 = max(0, left), max(0, top)
        x1, y1 = min(width, right), min(height, bottom)
        if x0 >= x1 or y0 >= y1:
            continue

        mask_x0, mask_y0 = x0 - left, y0 - top
        mask_x1, mask_y1 = mask_x0 + (x1 - x0), mask_y0 + (y1 - y0)
        halo_roi = halo[mask_y0:mask_y1, mask_x0:mask_x1]
        inner_roi = inner[mask_y0:mask_y1, mask_x0:mask_x1]
        core_roi = core[mask_y0:mask_y1, mask_x0:mask_x1]

        halo_visibility = 1.0
        if depth_aware and depth is not None and depth.shape == (height, width):
            scene_z, reliability = sample_depth(depth, valid, u, v)
            if reliability > 0 and scene_z + max(0.025, 0.04 * float(light.position_camera[2])) < float(light.position_camera[2]):
                halo_visibility = 0.28

        if result is None:
            result = image[..., :3].copy()
        roi = result[y0:y1, x0:x1].astype(np.float32)
        color = np.clip(np.asarray(light.color_rgb, dtype=np.float32), 0.0, 1.0)
        power = float(np.clip(light.intensity * light.confidence, 0.0, 2.0))
        glow = (halo_roi * 0.12 * halo_visibility + inner_roi * 0.28) * power
        roi += glow[..., None] * color[None, None, :] * 100.0

        core_alpha = np.clip(core_roi * min(0.62, 0.38 + power * 0.05), 0.0, 0.62)[..., None]
        hot_color = color * 0.35 + np.array([1.0, 0.98, 0.92], dtype=np.float32) * 0.65
        roi = roi * (1.0 - core_alpha) + hot_color[None, None, :] * (255.0 * core_alpha)
        result[y0:y1, x0:x1] = np.clip(roi, 0.0, 255.0).astype(np.uint8)

    return image if result is None else result


def light_from_palm(
    geometry: GeometryState,
    palm_uv: tuple[float, float],
    confidence: float,
    *,
    color_rgb: tuple[float, float, float] = (1.0, 0.78, 0.48),
    intensity: float = 1.4,
    d_ref: float = 0.7,
    range_m: float = 1.0,
    source_hand: int = 0,
    timestamp: float | None = None,
) -> LightState | None:
    """Back-project a detected palm using the *real* depth map without double attenuation."""
    z, sample_conf = sample_depth(geometry.depth, geometry.valid_mask, *palm_uv)
    if z <= 0:
        return None
    position = geometry.camera.unproject(palm_uv[0], palm_uv[1], z).astype(np.float32)
    # Physically coherent: intensity represents base emitted power I0.
    # No artificial attenuation from camera to hand (avoids double attenuation).
    return LightState(
        position_camera=position,
        intensity=float(intensity),
        color_rgb=np.asarray(color_rgb, dtype=np.float32),
        confidence=float(np.clip(confidence, 0.0, 1.0) * sample_conf),
        source_hand=int(source_hand),
        timestamp=time.time() if timestamp is None else float(timestamp),
        enabled=True,
        light_id=int(source_hand),
        range_m=float(range_m),
    )


def _shadow_factor(geometry: GeometryState, light: LightState, *, steps: int = 6) -> np.ndarray:
    """Camera-space screen-ray visibility; out-of-frame samples stay unknown."""
    h, w = geometry.depth.shape
    pos = light.position_camera if light.position_camera is not None else light.position_camera_m
    depth = np.asarray(geometry.depth, dtype=np.float32)
    valid = np.asarray(geometry.valid_mask, dtype=bool) & np.isfinite(depth) & (depth > 1e-5)
    points = np.asarray(geometry.positions_3d, dtype=np.float32)
    finite = np.isfinite(points).all(axis=-1) & (points[..., 2] > 1e-4)
    radius = max(float(getattr(light, "source_radius_m", 0.025)), 0.0)
    offsets = ((-0.65, -0.65), (0.65, -0.65), (-0.65, 0.65), (0.65, 0.65))
    visibility_sum = np.zeros((h, w), dtype=np.float32)

    for ox, oy in offsets:
        emitter = np.asarray(pos, dtype=np.float32).copy()
        emitter[0] += ox * radius
        emitter[1] += oy * radius
        ray = emitter.reshape(1, 1, 3) - points
        visibility = np.ones((h, w), dtype=np.float32)
        for fraction in np.linspace(0.07, 0.94, max(1, int(steps)), dtype=np.float32):
            sample = points + ray * fraction
            sample_z = sample[..., 2]
            safe_z = np.maximum(sample_z, 1e-4)
            sample_u = geometry.camera.fx * sample[..., 0] / safe_z + geometry.camera.cx
            sample_v = geometry.camera.fy * sample[..., 1] / safe_z + geometry.camera.cy
            inside = finite & (sample_z > 1e-4) & (sample_u >= 0.0) & (sample_u < w) & (sample_v >= 0.0) & (sample_v < h)
            ix = np.rint(np.nan_to_num(sample_u, nan=-1.0, posinf=-1.0, neginf=-1.0)).astype(np.int32).clip(0, w - 1)
            iy = np.rint(np.nan_to_num(sample_v, nan=-1.0, posinf=-1.0, neginf=-1.0)).astype(np.int32).clip(0, h - 1)
            scene_z = depth[iy, ix]
            bias = np.maximum(0.005, 0.012 * sample_z)
            blocked = inside & valid[iy, ix] & (scene_z < sample_z - bias)
            visibility[blocked] = np.minimum(visibility[blocked], 0.20)
        visibility_sum += visibility
    return visibility_sum / float(len(offsets))


def render_volumetric_scattering(
    geometry: GeometryState,
    lights: list[LightState] | tuple[LightState, ...],
    *,
    num_steps: int = 5,
    downsample_factor: int = 4,
    density: float = 0.08,
) -> tuple[np.ndarray, float]:
    """Ray-marched volumetric light shafts/haze.

    Evaluated at quarter-resolution for real-time responsiveness and upsampled
    back to full image resolution.
    """
    import cv2

    t0 = time.perf_counter()
    full_h, full_w = geometry.depth.shape
    h = max(2, full_h // downsample_factor)
    w = max(2, full_w // downsample_factor)

    depth_low = cv2.resize(geometry.depth, (w, h), interpolation=cv2.INTER_NEAREST)
    valid_low = np.isfinite(depth_low) & (depth_low > 1e-4)

    # Subsampled camera model
    scale_x = w / float(full_w)
    scale_y = h / float(full_h)
    fx_low = geometry.camera.fx * scale_x
    fy_low = geometry.camera.fy * scale_y
    cx_low = geometry.camera.cx * scale_x
    cy_low = geometry.camera.cy * scale_y

    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    ray_dir_x = (u_grid - cx_low) / fx_low
    ray_dir_y = (v_grid - cy_low) / fy_low
    ray_dir_z = np.ones((h, w), dtype=np.float32)
    ray_len = np.sqrt(ray_dir_x ** 2 + ray_dir_y ** 2 + 1.0)
    ray_dir_x /= ray_len
    ray_dir_y /= ray_len
    ray_dir_z /= ray_len

    haze = np.zeros((h, w, 3), dtype=np.float32)
    max_z = np.where(valid_low, depth_low, 3.0)
    t_min = 0.15

    for light in lights:
        if not light.enabled or light.confidence < 0.1:
            continue
        pos = light.position_camera if light.position_camera is not None else light.position_camera_m
        light_pos = pos.astype(np.float32)
        color = light.color_rgb.astype(np.float32)
        intensity = float(light.intensity) * float(light.confidence)

        for step in np.linspace(0.1, 0.9, num_steps, dtype=np.float32):
            sample_z = t_min + step * (max_z - t_min)
            t_dist = sample_z / np.maximum(ray_dir_z, 1e-4)
            px = ray_dir_x * t_dist
            py = ray_dir_y * t_dist
            pz = sample_z

            dx = light_pos[0] - px
            dy = light_pos[1] - py
            dz = light_pos[2] - pz
            dist_sq = dx ** 2 + dy ** 2 + dz ** 2 + 0.05
            range_m = max(float(getattr(light, "range_m", 1.0)), 1e-3)
            source_falloff = np.exp(-0.5 * dist_sq / (range_m * range_m))
            in_scatter = intensity / dist_sq * source_falloff

            # March from each haze sample toward its emitter. Off-screen
            # projections are unknown, not clamped to a border depth texel.
            visible = np.ones((h, w), dtype=np.float32)
            for shadow_t in np.linspace(0.12, 0.90, max(1, min(6, num_steps)), dtype=np.float32):
                sx = px + (light_pos[0] - px) * shadow_t
                sy = py + (light_pos[1] - py) * shadow_t
                sz = pz + (light_pos[2] - pz) * shadow_t
                safe_sz = np.maximum(sz, 1e-4)
                su = fx_low * sx / safe_sz + cx_low
                sv = fy_low * sy / safe_sz + cy_low
                inside = (sz > 1e-4) & (su >= 0.0) & (su < w) & (sv >= 0.0) & (sv < h)
                ix = np.rint(np.nan_to_num(su, nan=-1.0, posinf=-1.0, neginf=-1.0)).astype(np.int32).clip(0, w - 1)
                iy = np.rint(np.nan_to_num(sv, nan=-1.0, posinf=-1.0, neginf=-1.0)).astype(np.int32).clip(0, h - 1)
                blocker = depth_low[iy, ix]
                blocked = inside & valid_low[iy, ix] & (blocker < sz - np.maximum(0.008, 0.014 * sz))
                visible[blocked] = np.minimum(visible[blocked], 0.18)

            contribution = in_scatter * visible * density
            haze[..., 0] += contribution * color[0]
            haze[..., 1] += contribution * color[1]
            haze[..., 2] += contribution * color[2]

    # Upscale back to full resolution with smooth interpolation
    full_haze = cv2.resize(haze, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return full_haze, elapsed


def shade_geometry(
    rgb: np.ndarray,
    geometry: GeometryState,
    lights: list[LightState] | tuple[LightState, ...],
    *,
    ambient: float = 0.18,
    specular_strength: float = 0.28,
    shininess: float = 48.0,
    shadows: bool = True,
    volumetrics: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply diffuse + Blinn-Phong specular lighting, dynamic shadows, and optional volumetrics."""
    started = time.perf_counter()
    image = np.asarray(rgb, dtype=np.float32)[..., :3] / 255.0
    if not lights or geometry.normals is None:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8), {"lighting_ms": 0.0, "lights": 0.0, "volumetrics_ms": 0.0}

    points = np.asarray(geometry.positions_3d, dtype=np.float32)
    normals = np.nan_to_num(np.asarray(geometry.normals, dtype=np.float32), nan=0.0)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-6)
    valid = np.asarray(geometry.valid_mask, dtype=bool)
    confidence = np.asarray(geometry.confidence if geometry.confidence is not None else valid, dtype=np.float32)
    view = -points
    view /= np.maximum(np.linalg.norm(view, axis=-1, keepdims=True), 1e-6)

    lit = image * float(ambient)
    active_count = 0

    for light in lights:
        if not getattr(light, "enabled", True) or float(light.confidence) <= 0.0:
            continue
        active_count += 1
        pos = light.position_camera if light.position_camera is not None else light.position_camera_m
        delta = pos.reshape(1, 1, 3) - points
        distance = np.linalg.norm(delta, axis=-1, keepdims=True)
        direction = delta / np.maximum(distance, 1e-6)
        diffuse = np.maximum(np.sum(normals * direction, axis=-1), 0.0)
        halfway = direction + view
        halfway /= np.maximum(np.linalg.norm(halfway, axis=-1, keepdims=True), 1e-6)
        specular = np.maximum(np.sum(normals * halfway, axis=-1), 0.0) ** float(shininess)

        # Keep each hand light local while retaining smooth quadratic falloff.
        range_m = max(float(getattr(light, "range_m", 1.0)), 1e-3)
        attenuation = float(light.intensity) / (1.0 + (distance[..., 0] / range_m) ** 2)
        visibility = _shadow_factor(geometry, light) if shadows else 1.0
        contribution = (diffuse * 0.92 + specular * specular_strength) * attenuation * visibility * float(light.confidence)
        lit += contribution[..., None] * light.color_rgb.reshape(1, 1, 3)

    lit = np.where(valid[..., None], lit, image * 0.1)
    lit *= np.clip(0.65 + 0.35 * confidence[..., None], 0.0, 1.0)

    vol_ms = 0.0
    if volumetrics and active_count > 0:
        haze, vol_ms = render_volumetric_scattering(geometry, lights)
        lit += haze

    elapsed = (time.perf_counter() - started) * 1000.0
    return np.clip(lit * 255.0, 0, 255).astype(np.uint8), {
        "lighting_ms": elapsed,
        "lights": float(active_count),
        "volumetrics_ms": vol_ms,
    }
