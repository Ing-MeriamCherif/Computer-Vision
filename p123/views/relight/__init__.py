"""Mode 7: live P123 HandXYZ-controlled GPU relighting with CPU reference fallback."""

from __future__ import annotations

import time
import threading
import math
from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from geometry.lighting import (
    DEFAULT_DIRECT_GAIN,
    DEFAULT_DIFFUSE_STRENGTH,
    DEFAULT_SHININESS,
    DEFAULT_SPECULAR_STRENGTH,
    LIGHTING_STAGES,
    LightState,
    normalize_lighting_stage,
    project_light_orb,
    render_light_orbs,
    shade_geometry,
)
from geometry.state import GeometryState


LIGHT_COLORS = ((0.44, 0.72, 0.82), (0.88, 0.63, 0.40))
COLOR_PRESETS = (("CYAN", (0.18, 0.78, 1.0)), ("AMBER", (1.0, 0.53, 0.14)), ("MAGENTA", (0.95, 0.24, 0.70)), ("WHITE", (1.0, 1.0, 0.92)))
DEFAULT_RANGE_M = 0.30
DEFAULT_INTENSITY = 0.70
MIN_INTENSITY, MAX_INTENSITY = 0.10, 2.00
MIN_RANGE_M, MAX_RANGE_M = 0.12, 1.50
FRESHNESS_LIMIT_MS = 250.0
FADE_START_MS = 150.0
HAND_LIGHT_HOLD_MS = 160.0
HAND_LIGHT_FADE_START_MS = 80.0


def _open_palm_state(hand: Any) -> bool | None:
    """Classify a detected palm; return None when landmarks are unavailable."""
    landmarks = getattr(hand, "landmarks_uv", None)
    if landmarks is None or len(landmarks) < 21:
        return None
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape[1:] != (2,) or not np.isfinite(points).all():
        return None
    wrist = points[0]
    # Four non-thumb fingertips should sit farther from the wrist than their
    # PIP joints for an open palm. Requiring three gives stable hysteresis
    # under mild perspective and partial finger occlusion.
    extended = sum(
        np.linalg.norm(points[tip] - wrist) > 1.12 * np.linalg.norm(points[pip] - wrist)
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
    )
    return extended >= 3


def lights_from_snapshot(
    snapshot: Any,
    *,
    freshness_limit_ms: float = FRESHNESS_LIMIT_MS,
) -> tuple[list[LightState], float | None]:
    """Adapt current tracked-hand and HandXYZ records without resampling depth."""
    tracked = {
        int(hand.hand_id): hand
        for hand in (snapshot.hand_state.hands if snapshot.hand_state is not None else ())
    }
    lights: list[LightState] = []
    ages: list[float] = []
    for xyz in getattr(snapshot, "xyz", ()):
        hand = tracked.get(int(xyz.hand_id))
        recorded_age = getattr(xyz, "source_age_ms", None)
        if recorded_age is None:
            recorded_age = getattr(xyz, "age_ms", None)
        try:
            recorded_age = max(0.0, float(recorded_age))
        except (TypeError, ValueError):
            recorded_age = 0.0
        try:
            timestamp = float(xyz.timestamp)
            capture_age = (time.monotonic() - timestamp) * 1000.0
            capture_age = max(0.0, capture_age) if math.isfinite(capture_age) else float("inf")
        except (TypeError, ValueError, AttributeError):
            capture_age = float("inf")
        # Stored worker age can only lag current reality. Capture time keeps a
        # frozen worker/camera from making an old HandXYZ look perpetually fresh.
        age_ms = max(recorded_age, capture_age) if math.isfinite(capture_age) else float("inf")
        ages.append(age_ms)
        position = xyz.xyz_camera
        # `TrackedHand.stale` means the current point came from the LK
        # between-detector path, not that the physical hand has expired.
        # HandXYZ source age and confidence are the authoritative freshness
        # signals for the light.
        if hand is None or position is None:
            continue
        position = np.asarray(position, dtype=np.float32)
        if position.shape != (3,) or not np.isfinite(position).all() or position[2] <= 0.0:
            continue
        if age_ms >= freshness_limit_ms or hand.confidence < 0.15 or xyz.confidence <= 0.0:
            continue
        fade_t = np.clip((freshness_limit_ms - age_ms) / max(freshness_limit_ms - FADE_START_MS, 1.0), 0.0, 1.0)
        fade = fade_t * fade_t * (3.0 - 2.0 * fade_t)
        confidence = float(np.clip(hand.confidence, 0.0, 1.0) * np.clip(xyz.confidence, 0.0, 1.0) * fade)
        if confidence < 0.04:
            continue
        hand_id = int(xyz.hand_id)
        lights.append(
            LightState(
                position_camera=position,
                intensity=DEFAULT_INTENSITY,
                color_rgb=np.asarray(LIGHT_COLORS[hand_id % len(LIGHT_COLORS)], dtype=np.float32),
                confidence=confidence,
                source_hand=hand_id,
                timestamp=float(xyz.timestamp),
                enabled=True,
                light_id=hand_id,
                range_m=DEFAULT_RANGE_M,
                source_radius_m=0.018,
                visual_radius_m=0.035,
            )
        )
        if len(lights) == 2:
            break
    return lights, max(ages) if ages else None


def _detail_preserving_composite(
    full_rgb: np.ndarray,
    small_rgb: np.ndarray,
    small_relit: np.ndarray,
    geometry: GeometryState,
    *,
    ambient: float,
) -> np.ndarray:
    """Upsample bounded CPU illumination while retaining full-resolution RGB detail."""
    base = np.asarray(full_rgb, dtype=np.float32)[..., :3]
    source = np.asarray(small_rgb, dtype=np.float32)[..., :3]
    relit = np.asarray(small_relit, dtype=np.float32)[..., :3]
    confidence = geometry.confidence
    if confidence is None:
        confidence = geometry.valid_mask.astype(np.float32)
    quality = np.clip(0.65 + 0.35 * np.nan_to_num(confidence, nan=0.0), 0.0, 1.0)
    baseline = source * max(float(ambient), 1e-3) * quality[..., None]
    gain = relit / np.maximum(baseline, 24.0)
    gain = np.clip(np.nan_to_num(gain, nan=1.0, posinf=1.8, neginf=1.0), 1.0, 1.8)
    gain[~geometry.valid_mask] = 1.0
    gain_full = cv2.resize(gain, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(base * gain_full, 0.0, 255.0).astype(np.uint8)


class RelightRenderer:
    """Latest-snapshot Mode 7 renderer with an explicit, testable CPU fallback."""

    def __init__(
        self,
        max_width: int = 192,
        max_height: int = 144,
        *,
        lighting_quality: str = "balanced",
        use_gpu: bool = True,
        backend: str = "auto",
    ) -> None:
        self.max_width = max(64, int(max_width))
        self.max_height = max(48, int(max_height))
        self.lighting_quality = lighting_quality
        self.use_gpu = bool(use_gpu)
        self.backend_requested = str(backend).lower()
        if self.backend_requested not in {"auto", "rtx", "raster"}:
            raise ValueError("backend must be auto, rtx, or raster")
        self.backend_name = "UNINITIALIZED"
        self.backend_reason: str | None = None
        self._rtx_renderer = None
        self._rtx_probe_done = False
        self._gpu_renderer = None
        self._gpu_error: str | None = None
        self._gpu_error_reported = False
        self._gpu_warmed = False
        self._gpu_warm_active = False
        self._gpu_warm_thread: threading.Thread | None = None
        self._cache_key: tuple[Any, ...] | None = None
        self._cache_image: np.ndarray | None = None
        self.last_render_ms = 0.0
        self.last_light_count = 0
        self.last_lighting_stats: dict[str, Any] = {}
        self.last_geometry_source_id: int | str | None = None
        self.last_geometry_age_ms: float | None = None
        self.last_xyz_source_age_ms: float | None = None
        self.light_intensity = DEFAULT_INTENSITY
        self.light_range_m = DEFAULT_RANGE_M
        self.light_color_index = 0
        self._gesture_enabled: dict[int, bool] = {}
        self._gesture_last_seen: dict[int, float] = {}
        self._recent_lights: dict[int, LightState] = {}
        self.lighting_stage = "full"

    def controlled_lights(self, lights: list[LightState]) -> list[LightState]:
        color = np.asarray(COLOR_PRESETS[self.light_color_index][1], dtype=np.float32)
        return [replace(light, intensity=self.light_intensity, range_m=self.light_range_m, color_rgb=color.copy()) for light in lights]

    def set_lighting_quality(self, quality: str) -> None:
        self.lighting_quality = str(quality).lower()
        if self._gpu_renderer is not None:
            self._gpu_renderer.set_quality(self.lighting_quality)
        self._cache_key = None

    def set_lighting_stage(self, stage: str) -> None:
        value = normalize_lighting_stage(stage)
        if value == self.lighting_stage:
            return
        self.lighting_stage = value
        self._cache_key = None
        if self._gpu_renderer is not None:
            self._gpu_renderer.reset_history()

    def cycle_lighting_stage(self) -> str:
        stages = ("l2_diffuse", "l2_diffuse_specular", "l4_shadows", "full")
        index = stages.index(self.lighting_stage)
        self.set_lighting_stage(stages[(index + 1) % len(stages)])
        return self.lighting_stage

    def set_backend(self, backend: str) -> None:
        value = str(backend).lower()
        if value not in {"auto", "rtx", "raster"}:
            raise ValueError("backend must be auto, rtx, or raster")
        self.backend_requested = value
        self._rtx_renderer = None
        self._rtx_probe_done = False
        self.backend_name = "UNINITIALIZED"
        self.backend_reason = None
        self.reset_for_source_change()

    def close(self) -> None:
        if self._gpu_warm_thread is not None and self._gpu_warm_thread.is_alive():
            self._gpu_warm_thread.join(timeout=5.0)
        if self._gpu_renderer is not None:
            self._gpu_renderer.close()
            self._gpu_renderer = None
        if self._rtx_renderer is not None:
            self._rtx_renderer.close()
            self._rtx_renderer = None
        self._gpu_warmed = False
        self._gesture_enabled.clear()
        self._gesture_last_seen.clear()
        self._recent_lights.clear()

    def reset_for_source_change(self) -> None:
        """Drop camera-dependent caches and history while retaining GL allocations."""
        if self._gpu_warm_thread is not None and self._gpu_warm_thread.is_alive():
            self._gpu_warm_thread.join(timeout=5.0)
        if self._gpu_renderer is not None:
            self._gpu_renderer.reset_history()
        self._cache_key = None
        self._cache_image = None
        self._gesture_enabled.clear()
        self._gesture_last_seen.clear()
        self._recent_lights.clear()
        self.last_geometry_source_id = None
        self.last_geometry_age_ms = None
        self.last_xyz_source_age_ms = None
        self.last_light_count = 0
        self.last_lighting_stats = {"renderer": "IDLE", "lights": 0.0}

    def _gesture_gated_lights(self, snapshot: Any, lights: list[LightState]) -> list[LightState]:
        hands = snapshot.hand_state.hands if snapshot.hand_state is not None else ()
        visible_ids = {int(hand.hand_id) for hand in hands}
        now = time.monotonic()
        for hand in hands:
            hand_id = int(hand.hand_id)
            try:
                seen_at = float(hand.timestamp)
                if not math.isfinite(seen_at):
                    seen_at = now
            except (TypeError, ValueError, AttributeError):
                seen_at = now
            self._gesture_last_seen[hand_id] = max(self._gesture_last_seen.get(hand_id, seen_at), seen_at)
            # Only detector frames change the latch. Optical-flow frames keep
            # the last deliberate open/closed state.
            if not getattr(hand, "stale", False):
                state = _open_palm_state(hand)
                if state is not None:
                    self._gesture_enabled[hand_id] = state
                    if not state:
                        self._recent_lights.pop(hand_id, None)

        current: dict[int, LightState] = {}
        for light in lights:
            hand_id = int(light.source_hand)
            if self._gesture_enabled.get(hand_id, True):
                current[hand_id] = light
                self._recent_lights[hand_id] = light

        result = list(current.values())
        for hand_id, light in tuple(self._recent_lights.items()):
            if self._gesture_enabled.get(hand_id, True) is False:
                self._recent_lights.pop(hand_id, None)
                continue
            age_ms = max(0.0, (now - float(light.timestamp)) * 1000.0)
            if age_ms >= HAND_LIGHT_HOLD_MS:
                self._recent_lights.pop(hand_id, None)
                if hand_id not in visible_ids:
                    self._gesture_enabled.pop(hand_id, None)
                    self._gesture_last_seen.pop(hand_id, None)
                continue
            if hand_id in current:
                continue

            fade_t = np.clip(
                (HAND_LIGHT_HOLD_MS - age_ms)
                / max(HAND_LIGHT_HOLD_MS - HAND_LIGHT_FADE_START_MS, 1.0),
                0.0,
                1.0,
            )
            fade = float(fade_t * fade_t * (3.0 - 2.0 * fade_t))
            if fade > 0.0:
                result.append(replace(light, confidence=light.confidence * fade))
                self.last_xyz_source_age_ms = max(self.last_xyz_source_age_ms or 0.0, age_ms)

        for hand_id, seen_at in tuple(self._gesture_last_seen.items()):
            if hand_id not in visible_ids and (now - seen_at) * 1000.0 >= HAND_LIGHT_HOLD_MS:
                self._gesture_last_seen.pop(hand_id, None)
                self._gesture_enabled.pop(hand_id, None)
                self._recent_lights.pop(hand_id, None)
        return sorted(result, key=lambda light: int(light.source_hand))

    def _geometry(self, snapshot: Any) -> GeometryState | None:
        return getattr(snapshot, "fast_geometry_state", None) or snapshot.geometry_state

    def _low_geometry(self, geometry: GeometryState) -> GeometryState:
        # Kept as a compatibility hook for callers/tests from the earlier
        # renderer.  L2 quality is never reduced by downsampling the surface
        # pass, so the production fallback uses the authoritative state.
        return geometry

    def _cache_key_for(self, snapshot: Any, geometry: GeometryState, lights: list[LightState]) -> tuple[Any, ...]:
        return (
            snapshot.rgb_capture_id,
            geometry.source_frame_id,
            tuple((light.light_id, tuple(np.round(light.position_camera, 3)), round(light.confidence, 2)) for light in lights),
            int((self.last_xyz_source_age_ms or 0.0) // 25),
            self.lighting_quality,
            self.lighting_stage,
        )

    def _cpu_render(self, frame: np.ndarray, geometry: GeometryState, lights: list[LightState]) -> np.ndarray:
        stage = LIGHTING_STAGES[self.lighting_stage]
        surface_geometry = self._low_geometry(geometry)
        relit, stats = shade_geometry(
            frame,
            surface_geometry,
            lights,
            ambient=0.40,
            diffuse_strength=DEFAULT_DIFFUSE_STRENGTH,
            specular_strength=DEFAULT_SPECULAR_STRENGTH,
            shininess=DEFAULT_SHININESS,
            direct_gain=DEFAULT_DIRECT_GAIN,
            specular_enabled=bool(stage["specular"]),
            shadows=bool(stage["shadows"]),
            volumetrics=bool(stage["volumetrics"]),
            shadows_enabled=bool(stage["shadows"]),
            volumetrics_enabled=bool(stage["volumetrics"]),
        )
        result = relit
        result = render_light_orbs(result, geometry.camera, lights, depth=geometry.depth, valid=geometry.valid_mask)
        stats.update({
            "renderer": "CPU_FALLBACK",
            "quality": self.lighting_quality,
            "shadow_quality": "CPU reference",
            "volumetric_quality": "CPU reference",
            "stage": self.lighting_stage,
            "fallback_reason": self._gpu_error or "GPU renderer unavailable",
        })
        self.last_lighting_stats = stats
        return result

    def _ensure_gpu(self):
        if not self.use_gpu:
            return None
        if self._gpu_error is not None:
            if self._gpu_renderer is not None:
                self._gpu_renderer.close()
                self._gpu_renderer = None
            if not self._gpu_error_reported:
                print(f"GPU RELIGHT INIT FAILED: {self._gpu_error}; falling back to CPU reference renderer")
                self._gpu_error_reported = True
            return None
        if self._gpu_renderer is None:
            try:
                from geometry.gpu_lighting import GPURelightRenderer

                self._gpu_renderer = GPURelightRenderer(self.lighting_quality)
            except Exception as exc:
                self._gpu_error = f"{type(exc).__name__}: {exc}"
                return self._ensure_gpu()
        return self._gpu_renderer

    def _ensure_rtx(self):
        if self.backend_requested == "raster":
            return None
        if self._rtx_probe_done:
            return self._rtx_renderer
        if self._rtx_renderer is not None:
            return self._rtx_renderer
        self._rtx_probe_done = True
        try:
            from geometry.rtx_lighting import create_optix_renderer

            self._rtx_renderer = create_optix_renderer(self.lighting_quality)
            self.backend_name = "RTX_OPTIX"
            return self._rtx_renderer
        except Exception as exc:  # noqa: BLE001
            self.backend_reason = f"{type(exc).__name__}: {exc}"
            if self.backend_requested == "rtx":
                self.backend_name = "RTX_UNAVAILABLE"
                return None
            self.backend_name = "OPENGL_RASTER"
            return None

    def _start_gpu_warmup(self, renderer, frame: np.ndarray, geometry: GeometryState, lights: list[LightState]) -> None:
        if self._gpu_warm_active:
            return
        self._gpu_warm_active = True
        warm_frame = np.ascontiguousarray(frame.copy())

        def warm() -> None:
            try:
                stage = LIGHTING_STAGES[self.lighting_stage]
                renderer.render(
                    warm_frame, geometry, lights, ambient=0.40, readback=False,
                    stage=self.lighting_stage, diffuse_strength=DEFAULT_DIFFUSE_STRENGTH,
                    specular_strength=DEFAULT_SPECULAR_STRENGTH,
                    shininess=DEFAULT_SHININESS, direct_gain=DEFAULT_DIRECT_GAIN,
                    shadows_enabled=bool(stage["shadows"]),
                    volumetrics_enabled=bool(stage["volumetrics"]),
                )
                self._gpu_warmed = True
            except Exception as exc:
                self._gpu_error = f"warmup {type(exc).__name__}: {exc}"
            finally:
                self._gpu_warm_active = False

        self._gpu_warm_thread = threading.Thread(target=warm, name="p123-gpu-relight-warmup", daemon=True)
        self._gpu_warm_thread.start()

    def render(self, snapshot: Any) -> tuple[np.ndarray | None, str, str | None]:
        frame = snapshot.rgb_frame
        title = "MODE 7 - HAND-HELD RELIGHT"
        if frame is None:
            return None, title, "waiting for camera"
        frame = np.asarray(frame)[..., :3]
        geometry = self._geometry(snapshot)
        if geometry is None or geometry.normals is None:
            self.last_light_count = 0
            self.last_lighting_stats = {"renderer": "WAITING"}
            self._cache_image = frame.copy()
            return self._cache_image, title, "waiting for surface geometry"

        self.last_geometry_source_id = geometry.source_frame_id
        self.last_geometry_age_ms = max(0.0, (time.monotonic() - geometry.timestamp) * 1000.0)
        lights, self.last_xyz_source_age_ms = lights_from_snapshot(snapshot)
        lights = self.controlled_lights(self._gesture_gated_lights(snapshot, lights))
        key = self._cache_key_for(snapshot, geometry, lights)
        if key == self._cache_key and self._cache_image is not None:
            return self._cache_image, title, None

        if not lights:
            started = time.perf_counter()
            self.last_light_count = 0
            self.last_lighting_stats = {
                "renderer": "IDLE",
                "quality": self.lighting_quality,
                "lights": 0.0,
                "geometry_age_ms": float(self.last_geometry_age_ms),
                "xyz_source_age_ms": float(self.last_xyz_source_age_ms or 0.0),
            }
            self.last_render_ms = (time.perf_counter() - started) * 1000.0
            self._cache_key = key
            self._cache_image = frame.copy()
            return self._cache_image, title, None

        started = time.perf_counter()
        rtx_renderer = self._ensure_rtx()
        if self.backend_requested == "rtx" and rtx_renderer is None:
            self.last_light_count = len(lights)
            self.last_lighting_stats = {
                "renderer": "RTX_UNAVAILABLE",
                "quality": self.lighting_quality,
                "lights": float(len(lights)),
                "fallback_reason": self.backend_reason or "RTX backend unavailable",
            }
            self.last_render_ms = (time.perf_counter() - started) * 1000.0
            self._cache_key = key
            self._cache_image = frame.copy()
            return self._cache_image, title, None
        if rtx_renderer is not None:
            try:
                image, stats = rtx_renderer.render(
                    frame, geometry, lights, ambient=0.40, stage=self.lighting_stage,
                    diffuse_strength=DEFAULT_DIFFUSE_STRENGTH,
                    specular_strength=DEFAULT_SPECULAR_STRENGTH,
                    shininess=DEFAULT_SHININESS, direct_gain=DEFAULT_DIRECT_GAIN,
                )
                self.last_lighting_stats = dict(stats)
                self.last_lighting_stats["renderer"] = "RTX_OPTIX"
            except Exception as exc:  # noqa: BLE001
                self.backend_reason = f"RTX render {type(exc).__name__}: {exc}"
                if self.backend_requested == "rtx":
                    self.last_lighting_stats = {"renderer": "RTX_UNAVAILABLE", "fallback_reason": self.backend_reason}
                    self._cache_image = frame.copy()
                    return self._cache_image, title, None
                self.backend_name = "OPENGL_RASTER"
                image = self._cpu_render(frame, geometry, lights)
        else:
            renderer = self._ensure_gpu()
            if renderer is not None:
                if not self._gpu_warmed:
                    self._start_gpu_warmup(renderer, frame, geometry, lights)
                    self.last_light_count = len(lights)
                    self.last_lighting_stats = {"renderer": "GPU_WARMUP", "quality": self.lighting_quality, "lights": float(len(lights))}
                    self.last_render_ms = (time.perf_counter() - started) * 1000.0
                    return frame.copy(), title, None
                try:
                    image, stats = renderer.render(
                        frame, geometry, lights, ambient=0.40, stage=self.lighting_stage,
                        diffuse_strength=DEFAULT_DIFFUSE_STRENGTH,
                        specular_strength=DEFAULT_SPECULAR_STRENGTH,
                        shininess=DEFAULT_SHININESS, direct_gain=DEFAULT_DIRECT_GAIN,
                    )
                    self.last_lighting_stats = dict(stats)
                    self.last_lighting_stats["renderer"] = "OPENGL_RASTER"
                except Exception as exc:
                    self._gpu_error = f"{type(exc).__name__}: {exc}"
                    self._gpu_error_reported = True
                    renderer.close()
                    self._gpu_renderer = None
                    image = self._cpu_render(frame, geometry, lights)
            else:
                image = self._cpu_render(frame, geometry, lights)

        self.last_light_count = len(lights)
        self.last_render_ms = (time.perf_counter() - started) * 1000.0
        self.last_lighting_stats.update({
            "lights": float(self.last_light_count),
            "geometry_age_ms": float(self.last_geometry_age_ms),
            "xyz_source_age_ms": float(self.last_xyz_source_age_ms or 0.0),
        })
        for index, light in enumerate(lights):
            projected = project_light_orb(geometry.camera, light)
            if projected is None:
                continue
            u, v, _ = projected
            position = light.position_camera
            label = f"H{light.source_hand} XYZ [{position[0]:+.2f}, {position[1]:+.2f}, {position[2]:.2f}]m"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            x = max(8, image.shape[1] - text_w - 18)
            y = 22 + 25 * index
            cv2.rectangle(image, (x - 5, y - text_h - 3), (x + text_w + 5, y + baseline + 2), (20, 26, 32), -1)
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (224, 231, 239), 1, cv2.LINE_AA)
        self._cache_key = key
        self._cache_image = image
        return image, title, None


_renderer = RelightRenderer()
_active_control: str | None = None


def _control_layout(viewport: tuple[int, int, int, int]) -> dict[str, tuple[int, int, int, int]]:
    vx, vy, vw, vh = viewport
    width = min(vw - 20, max(420, min(560, int(vw * 0.72))))
    if width < 300:
        return {}
    x, y = vx + (vw - width) // 2, vy + vh - 78
    scale = width / 560.0
    return {"panel": (x, y, width, 66), "intensity": (x + round(16*scale), y+46, round(148*scale), 18), "range": (x+round(190*scale), y+46, round(148*scale), 18), "color": (x+round(374*scale), y+12, round(170*scale), 42)}


def _set_slider(name: str, x: int, rect: tuple[int, int, int, int]) -> None:
    fraction = float(np.clip((x-rect[0])/max(rect[2], 1), 0.0, 1.0))
    if name == "intensity": _renderer.light_intensity = MIN_INTENSITY + fraction*(MAX_INTENSITY-MIN_INTENSITY)
    else: _renderer.light_range_m = MIN_RANGE_M + fraction*(MAX_RANGE_M-MIN_RANGE_M)
    _renderer._cache_key = None


def handle_control_mouse(event: int, x: int, y: int, flags: int, display_size: tuple[int, int]) -> bool:
    global _active_control
    from ..common import compute_layout
    layout = compute_layout(display_size); controls = _control_layout((layout["vx"], layout["vy"], layout["vw"], layout["vh"]))
    if not controls: return False
    if event == cv2.EVENT_LBUTTONDOWN:
        for name in ("intensity", "range"):
            rx, ry, rw, rh = controls[name]
            if rx-8 <= x <= rx+rw+8 and ry-12 <= y <= ry+rh+6: _active_control=name; _set_slider(name,x,controls[name]); return True
        rx,ry,rw,rh=controls["color"]
        if rx <= x <= rx+rw and ry <= y <= ry+rh: _active_control="color"; return True
    if event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON and _active_control in ("intensity","range"):
        _set_slider(_active_control,x,controls[_active_control]); return True
    if event == cv2.EVENT_LBUTTONUP:
        previous=_active_control; _active_control=None
        if previous in ("intensity","range"): _set_slider(previous,x,controls[previous]); return True
        if previous == "color": _renderer.light_color_index=(_renderer.light_color_index+1)%len(COLOR_PRESETS); _renderer._cache_key=None; return True
    return False


def draw_controls(canvas: np.ndarray, viewport: tuple[int, int, int, int]) -> None:
    controls = _control_layout(viewport)
    if not controls: return
    x,y,w,h=controls["panel"]; cv2.rectangle(canvas,(x,y),(x+w,y+h),(28,32,38),-1)
    for name,value,low,high,label in (("intensity",_renderer.light_intensity,MIN_INTENSITY,MAX_INTENSITY,"INTENSITY"),("range",_renderer.light_range_m,MIN_RANGE_M,MAX_RANGE_M,"RANGE")):
        rx,ry,rw,_=controls[name]; cv2.putText(canvas,f"{label} {value:.2f}",(rx,ry-14),cv2.FONT_HERSHEY_SIMPLEX,.34,(235,235,235),1,cv2.LINE_AA); filled=rx+round(rw*(value-low)/(high-low)); cv2.line(canvas,(rx,ry),(rx+rw,ry),(85,90,100),5); cv2.line(canvas,(rx,ry),(filled,ry),(240,130,40),5); cv2.circle(canvas,(filled,ry),6,(245,245,245),-1)
    rx,ry,_,_=controls["color"]; name,color=COLOR_PRESETS[_renderer.light_color_index]; cv2.putText(canvas,f"COLOR: {name}",(rx,ry+27),cv2.FONT_HERSHEY_SIMPLEX,.34,tuple(int(c*255) for c in color[::-1]),1,cv2.LINE_AA)


def configure(lighting_quality: str = "balanced", backend: str = "auto") -> None:
    _renderer.set_lighting_quality(lighting_quality)
    _renderer.set_backend(backend)


def render(snapshot: Any) -> tuple[np.ndarray | None, str, str | None]:
    return _renderer.render(snapshot)


__all__ = ["RelightRenderer", "lights_from_snapshot", "configure", "render"]
