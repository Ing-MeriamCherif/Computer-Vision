"""GPU-friendly (and CPU-safe) interactive lighting over ``GeometryState``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import time

import numpy as np

from .camera import CameraModel
from .depth_sampling import sample_depth as _sample_depth
from .state import GeometryState


DEFAULT_DIFFUSE_STRENGTH = 0.92
DEFAULT_SPECULAR_STRENGTH = 0.12
DEFAULT_SHININESS = 36.0
DEFAULT_AMBIENT = 0.40
DEFAULT_DIRECT_GAIN = 1.0 / DEFAULT_AMBIENT
LIGHTING_STAGES = {
    "l2_diffuse": {"diffuse": True, "specular": False, "shadows": False, "volumetrics": False},
    "l2_diffuse_specular": {"diffuse": True, "specular": True, "shadows": False, "volumetrics": False},
    "l4_shadows": {"diffuse": True, "specular": True, "shadows": True, "volumetrics": False},
    "full": {"diffuse": True, "specular": True, "shadows": True, "volumetrics": True},
}


def srgb_to_linear(color: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(color, dtype=np.float32), 0.0, 1.0)
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(color: np.ndarray) -> np.ndarray:
    value = np.maximum(np.nan_to_num(np.asarray(color, dtype=np.float32)), 0.0)
    return np.clip(np.where(value <= 0.0031308, value * 12.92, 1.055 * np.power(value, 1.0 / 2.4) - 0.055), 0.0, 1.0)


def normalize_lighting_stage(stage: str | None) -> str:
    value = str(stage or "full").strip().lower().replace("+", "_").replace(" ", "_")
    value = {"diffuse": "l2_diffuse", "l2": "l2_diffuse", "diffuse_specular": "l2_diffuse_specular", "shadows": "l4_shadows", "l4": "l4_shadows"}.get(value, value)
    if value not in LIGHTING_STAGES:
        raise ValueError(f"unknown lighting stage '{stage}'")
    return value


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
    visual_radius_m: float = 0.035
    orb_visibility: float = 1.0
    is_palm_attached: bool = False
    palm_center_camera: np.ndarray | None = None
    palm_normal_camera: np.ndarray | None = None
    palm_facing_score: float = 1.0
    orientation_confidence: float = 1.0
    self_intersection_epsilon_m: float = 0.002
    directionality: float = 0.78
    beam_inner_angle_deg: float = 32.0
    beam_outer_angle_deg: float = 78.0

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
        for name in ("palm_center_camera", "palm_normal_camera"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, np.asarray(value, dtype=np.float32).reshape(3))

    @property
    def effective_intensity(self) -> float:
        """Physical emitter power; camera-facing orb visibility is independent."""
        value = float(self.intensity) if self.enabled else 0.0
        return value if math.isfinite(value) and value > 0.0 else 0.0


def active_lights(lights: list[LightState] | tuple[LightState, ...]) -> list[LightState]:
    """Discard disabled, invalid, or palm-inactive sources before expensive passes."""
    return [
        light for light in lights
        if light.enabled and light.confidence > 0.0 and light.effective_intensity > 1e-4
        and np.isfinite(light.position_camera).all() and light.position_camera[2] > 0.0
    ]


def _emission_lobe(light: LightState, source_to_target: np.ndarray) -> np.ndarray | float:
    """Soft palm-directed power distribution with a constant omni floor."""
    normal = light.palm_normal_camera
    strength = float(np.clip(light.directionality, 0.0, 1.0))
    if not light.is_palm_attached or normal is None or strength <= 0.0:
        return 1.0
    direction = np.asarray(normal, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction).all() or norm <= 1e-6:
        return 1.0
    direction /= norm
    rays = np.asarray(source_to_target, dtype=np.float32)
    rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-6)
    cosine = np.sum(rays * direction, axis=-1)
    inner = math.cos(math.radians(float(np.clip(light.beam_inner_angle_deg, 1.0, 89.0))))
    outer_angle = float(np.clip(light.beam_outer_angle_deg, light.beam_inner_angle_deg + 1.0, 179.0))
    outer = math.cos(math.radians(outer_angle))
    t = np.clip((cosine - outer) / max(inner - outer, 1e-5), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return (1.0 - strength) + strength * smooth


def sample_depth(depth: np.ndarray, valid: np.ndarray | None, u: float, v: float, radius: int = 2) -> tuple[float, float]:
    """Sample depth at a palm, returning ``(z, reliability)``.

    Delegates to the shared local-neighborhood sampler so landmark queries do
    not scan the entire depth image for every hand point.
    """
    return _sample_depth(depth, valid, u, v, radius)


def project_light_orb(
    camera: CameraModel,
    light: LightState,
    *,
    source_radius_m: float | None = None,
) -> tuple[float, float, float] | None:
    """Return the pinhole projection and screen radius for a rendered light."""
    position = light.position_camera
    if not light.enabled or light.effective_intensity <= 1e-4 or float(light.orb_visibility) <= 1e-3 or position is None or not np.isfinite(position).all() or position[2] <= 0:
        return None
    uv = camera.project(position)
    if not np.isfinite(uv).all():
        return None
    radius_m = light.visual_radius_m if source_radius_m is None else float(source_radius_m)
    radius_px = float(np.clip(camera.fx * radius_m / float(position[2]), 6.0, 32.0))
    return float(uv[0]), float(uv[1]), radius_px


@lru_cache(maxsize=24)
def _light_orb_masks(radius_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build reusable masks for the local glow and shaded spherical emitter."""
    extent = max(5, int(math.ceil(radius_px * 1.8)))
    yy, xx = np.mgrid[-extent:extent + 1, -extent:extent + 1].astype(np.float32)
    distance = np.sqrt(xx * xx + yy * yy)
    broad_glow = np.exp(-0.5 * (distance / max(radius_px * 1.35, 1.0)) ** 2)
    inner = np.exp(-0.5 * (distance / max(radius_px * 0.88, 1.0)) ** 2)
    normalized = distance / max(float(radius_px), 1.0)
    sphere_depth = np.sqrt(np.maximum(1.0 - normalized * normalized, 0.0))
    highlight = np.exp(-0.5 * (
        ((xx + radius_px * 0.24) / max(radius_px * 0.12, 1.0)) ** 2
        + ((yy + radius_px * 0.30) / max(radius_px * 0.12, 1.0)) ** 2
    ))
    halo = np.maximum(broad_glow - inner, 0.0)
    return halo, inner, sphere_depth, highlight


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
        if projected is None or light.confidence <= 0 or light.effective_intensity <= 0:
            continue
        u, v, radius = projected
        cx, cy = int(round(u)), int(round(v))
        radius_i = max(4, int(round(radius)))
        halo, inner, sphere_depth, highlight = _light_orb_masks(radius_i)
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
        sphere_roi = sphere_depth[mask_y0:mask_y1, mask_x0:mask_x1]
        highlight_roi = highlight[mask_y0:mask_y1, mask_x0:mask_x1]

        orb_visibility = float(np.clip(light.orb_visibility, 0.0, 1.0))
        halo_visibility = 1.0
        if depth_aware and depth is not None and depth.shape == (height, width):
            scene_z, reliability = sample_depth(depth, valid, u, v)
            bias = max(0.006, 0.01 * float(light.position_camera[2])) if light.is_palm_attached else max(0.025, 0.04 * float(light.position_camera[2]))
            if reliability > 0 and scene_z + bias < float(light.position_camera[2]):
                if light.is_palm_attached:
                    continue
                halo_visibility = 0.28
        if halo_visibility <= 0.0:
            continue

        if result is None:
            result = image[..., :3].copy()
        roi = result[y0:y1, x0:x1].astype(np.float32)
        color = np.clip(np.asarray(light.color_rgb, dtype=np.float32), 0.0, 1.0)
        tracking_power = 1.0 if light.is_palm_attached else float(light.confidence)
        power = float(np.clip(light.effective_intensity * tracking_power, 0.0, 2.0))
        glow = (halo_roi * 0.05 * halo_visibility + inner_roi * 0.10) * power * orb_visibility
        roi += glow[..., None] * color[None, None, :] * 85.0

        normalized = np.sqrt(np.maximum(1.0 - sphere_roi * sphere_roi, 0.0))
        edge_t = np.clip((1.0 - normalized) * 8.0, 0.0, 1.0)
        edge_alpha = edge_t * edge_t * (3.0 - 2.0 * edge_t)
        sphere_alpha = (0.90 * edge_alpha * orb_visibility)[..., None]
        white_mix = np.clip(0.42 + 0.36 * sphere_roi + 0.55 * highlight_roi, 0.0, 1.0)[..., None]
        ball_color = color[None, None, :] * (1.0 - white_mix) + np.array([1.0, 0.98, 0.93], dtype=np.float32) * white_mix
        rim = ((1.0 - sphere_roi) ** 2 * 0.22)[..., None]
        ball_color = ball_color * (1.0 - rim) + color[None, None, :] * rim
        roi = roi * (1.0 - sphere_alpha) + ball_color * (255.0 * sphere_alpha)
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
    normals = np.zeros_like(points)
    if geometry.normals is not None:
        normals = np.nan_to_num(np.asarray(geometry.normals, dtype=np.float32), nan=0.0)
    epsilon = float(np.clip(light.self_intersection_epsilon_m, 0.0001, 0.02))

    for ox, oy in offsets:
        emitter = np.asarray(pos, dtype=np.float32).copy()
        emitter[0] += ox * radius
        emitter[1] += oy * radius
        start = points + normals * epsilon
        ray = emitter.reshape(1, 1, 3) - start
        visibility = np.ones((h, w), dtype=np.float32)
        for fraction in np.linspace(0.02, 0.98, max(1, int(steps)), dtype=np.float32):
            sample = start + ray * fraction
            sample_z = sample[..., 2]
            safe_z = np.maximum(sample_z, 1e-4)
            sample_u = geometry.camera.fx * sample[..., 0] / safe_z + geometry.camera.cx
            sample_v = geometry.camera.fy * sample[..., 1] / safe_z + geometry.camera.cy
            inside = finite & (sample_z > 1e-4) & (sample_u >= 0.0) & (sample_u < w) & (sample_v >= 0.0) & (sample_v < h)
            ix = np.rint(np.nan_to_num(sample_u, nan=-1.0, posinf=-1.0, neginf=-1.0)).astype(np.int32).clip(0, w - 1)
            iy = np.rint(np.nan_to_num(sample_v, nan=-1.0, posinf=-1.0, neginf=-1.0)).astype(np.int32).clip(0, h - 1)
            scene_z = depth[iy, ix]
            bias = np.maximum(0.005, 0.012 * sample_z)
            ray_min_z = np.minimum(start[..., 2], emitter[2])
            blocked = inside & valid[iy, ix] & (scene_z < sample_z - bias) & (scene_z > ray_min_z + bias)
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
    t0 = time.perf_counter()
    lights = active_lights(lights)
    full_h, full_w = geometry.depth.shape
    if not lights:
        return np.zeros((full_h, full_w, 3), dtype=np.float32), 0.0

    import cv2

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
        if light.confidence < 0.1:
            continue
        pos = light.position_camera if light.position_camera is not None else light.position_camera_m
        light_pos = pos.astype(np.float32)
        color = light.color_rgb.astype(np.float32)
        intensity = float(light.effective_intensity) * float(light.confidence)
        epsilon = float(np.clip(light.self_intersection_epsilon_m, 0.0001, 0.02))

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
            range_m = max(float(getattr(light, "range_m", 1.0)) * 0.78, 1e-3)
            source_falloff = np.exp(-0.5 * dist_sq / (range_m * range_m))
            source_to_sample = np.stack((px - light_pos[0], py - light_pos[1], pz - light_pos[2]), axis=-1)
            beam = _emission_lobe(light, source_to_sample)
            in_scatter = intensity / dist_sq * source_falloff * beam

            # March from each haze sample toward its emitter. Off-screen
            # projections are unknown, not clamped to a border depth texel.
            visible = np.ones((h, w), dtype=np.float32)
            for shadow_t in np.linspace(0.04, 0.96, max(1, min(6, num_steps)), dtype=np.float32):
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
                bias = np.maximum(epsilon, 0.008 * sz)
                blocked = (
                    inside & valid_low[iy, ix]
                    & (blocker < sz - bias)
                    & (blocker > np.minimum(pz, light_pos[2]) + bias)
                )
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
    diffuse_strength: float = DEFAULT_DIFFUSE_STRENGTH,
    specular_strength: float = 0.28,
    shininess: float = 48.0,
    direct_gain: float = DEFAULT_DIRECT_GAIN,
    specular_enabled: bool = True,
    shadows: bool = True,
    volumetrics: bool = False,
    shadows_enabled: bool | None = None,
    volumetrics_enabled: bool | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply diffuse + Blinn-Phong specular lighting, dynamic shadows, and optional volumetrics."""
    started = time.perf_counter()
    image = np.asarray(rgb, dtype=np.float32)[..., :3] / 255.0
    lights = active_lights(lights)
    if not lights or geometry.normals is None:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8), {"lighting_ms": 0.0, "raytrace_ms": 0.0, "lights": 0.0, "volumetrics_ms": 0.0}

    points = np.asarray(geometry.positions_3d, dtype=np.float32)
    normals = np.nan_to_num(np.asarray(geometry.normals, dtype=np.float32), nan=0.0)
    normal_length = np.linalg.norm(normals, axis=-1, keepdims=True)
    normal_valid = normal_length[..., 0] > 1e-4
    normals /= np.maximum(normal_length, 1e-6)
    valid = np.asarray(geometry.valid_mask, dtype=bool)
    confidence = np.asarray(geometry.confidence if geometry.confidence is not None else valid, dtype=np.float32)
    view = -points
    view /= np.maximum(np.linalg.norm(view, axis=-1, keepdims=True), 1e-6)

    if shadows_enabled is not None:
        shadows = bool(shadows_enabled)
    if volumetrics_enabled is not None:
        volumetrics = bool(volumetrics_enabled)
    lit = image.copy()
    active_count = 0
    raytrace_ms = 0.0

    for light in lights:
        active_count += 1
        pos = light.position_camera if light.position_camera is not None else light.position_camera_m
        delta = pos.reshape(1, 1, 3) - points
        distance = np.linalg.norm(delta, axis=-1, keepdims=True)
        direction = delta / np.maximum(distance, 1e-6)
        diffuse = np.maximum(np.sum(normals * direction, axis=-1), 0.0)
        halfway = direction + view
        halfway /= np.maximum(np.linalg.norm(halfway, axis=-1, keepdims=True), 1e-6)
        specular = np.maximum(np.sum(normals * halfway, axis=-1), 0.0) ** float(shininess)
        specular *= diffuse > 0.0
        if not specular_enabled:
            specular *= 0.0

        # Keep each hand light local while retaining smooth quadratic falloff.
        range_m = max(float(getattr(light, "range_m", 1.0)), 1e-3)
        attenuation = float(light.effective_intensity) / (1.0 + (distance[..., 0] / range_m) ** 2)
        attenuation *= _emission_lobe(light, -direction)
        if shadows:
            shadow_started = time.perf_counter()
            visibility = _shadow_factor(geometry, light)
            raytrace_ms += (time.perf_counter() - shadow_started) * 1000.0
        else:
            visibility = 1.0
        contribution = (diffuse * diffuse_strength + specular * specular_strength) * attenuation * visibility * float(light.confidence) * direct_gain
        lit += contribution[..., None] * light.color_rgb.reshape(1, 1, 3)

    usable = valid & normal_valid & (confidence > 0.0)
    lit = np.where(usable[..., None], lit, image)

    vol_ms = 0.0
    if volumetrics and active_count > 0:
        haze, vol_ms = render_volumetric_scattering(geometry, lights)
        lit += haze

    elapsed = (time.perf_counter() - started) * 1000.0
    return np.clip(lit * 255.0, 0, 255).astype(np.uint8), {
        "lighting_ms": elapsed,
        "raytrace_ms": raytrace_ms,
        "lights": float(active_count),
        "volumetrics_ms": vol_ms,
    }
