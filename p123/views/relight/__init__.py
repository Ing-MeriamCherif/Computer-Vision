"""Mode 7: live P123 HandXYZ-controlled GPU relighting with CPU reference fallback."""

from __future__ import annotations

import time
import threading
import math
from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from geometry.lighting import LightState, project_light_orb, render_light_orbs, shade_geometry
from geometry.palm_light import PalmLightController
from geometry.state import GeometryState


LIGHT_COLORS = ((0.44, 0.72, 0.82), (0.88, 0.63, 0.40))
COLOR_PRESETS = (
    ("CYAN", (0.18, 0.78, 1.00)),
    ("AMBER", (1.00, 0.53, 0.14)),
    ("MAGENTA", (0.95, 0.24, 0.70)),
    ("VIOLET", (0.48, 0.30, 1.00)),
    ("GREEN", (0.20, 0.92, 0.42)),
    ("RED", (1.00, 0.20, 0.16)),
    ("BLUE", (0.22, 0.42, 1.00)),
    ("LIME", (0.78, 0.96, 0.16)),
    ("PINK", (1.00, 0.38, 0.58)),
    ("ICE", (0.72, 0.94, 1.00)),
    ("GOLD", (1.00, 0.78, 0.18)),
    ("WHITE", (1.00, 1.00, 0.92)),
)
DEFAULT_RANGE_M = 0.30
DEFAULT_INTENSITY = 0.70
MIN_INTENSITY, MAX_INTENSITY = 0.10, 2.00
MIN_RANGE_M, MAX_RANGE_M = 0.12, 1.50
FRESHNESS_LIMIT_MS = 250.0
FADE_START_MS = 150.0
HAND_LIGHT_HOLD_MS = 80.0
HAND_LIGHT_FADE_START_MS = 40.0


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
    geometry: GeometryState | None = None,
    palm_controller: PalmLightController | None = None,
    mirrored_input: bool = True,
) -> tuple[list[LightState], float | None]:
    """Adapt HandXYZ records, optionally attaching a depth-reconstructed palm pose."""
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
        if geometry is not None and palm_controller is not None:
            light = palm_controller.update(
                hand,
                geometry,
                fallback_position=position,
                mirrored_input=mirrored_input,
                intensity=DEFAULT_INTENSITY,
                color_rgb=LIGHT_COLORS[hand_id % len(LIGHT_COLORS)],
                range_m=DEFAULT_RANGE_M,
                source_hand=hand_id,
                light_id=hand_id,
                timestamp=float(xyz.timestamp),
            )
            light.confidence = confidence
            light.orb_visibility *= float(np.clip(confidence / max(float(hand.confidence), 1e-4), 0.0, 1.0))
        else:
            light = LightState(
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
        lights.append(light)
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
    ) -> None:
        self.max_width = max(64, int(max_width))
        self.max_height = max(48, int(max_height))
        self.lighting_quality = lighting_quality
        self.use_gpu = bool(use_gpu)
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
        self._gesture_enabled: dict[int, bool] = {}
        self._gesture_last_seen: dict[int, float] = {}
        self._recent_lights: dict[int, LightState] = {}
        self._palm_light_controller = PalmLightController()
        self.light_intensity = DEFAULT_INTENSITY
        self.light_range_m = DEFAULT_RANGE_M
        self.light_color_index = 0

    def controlled_lights(self, lights: list[LightState]) -> list[LightState]:
        color = np.asarray(COLOR_PRESETS[self.light_color_index][1], dtype=np.float32)
        return [
            replace(light, intensity=self.light_intensity, range_m=self.light_range_m, color_rgb=color.copy())
            for light in lights
        ]

    def set_lighting_quality(self, quality: str) -> None:
        self.lighting_quality = str(quality).lower()
        if self._gpu_renderer is not None:
            self._gpu_renderer.set_quality(self.lighting_quality)
        self._cache_key = None

    def close(self) -> None:
        if self._gpu_warm_thread is not None and self._gpu_warm_thread.is_alive():
            self._gpu_warm_thread.join(timeout=5.0)
        if self._gpu_renderer is not None:
            self._gpu_renderer.close()
            self._gpu_renderer = None
        self._gpu_warmed = False
        self._gesture_enabled.clear()
        self._gesture_last_seen.clear()
        self._recent_lights.clear()
        self._palm_light_controller.reset()

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
        self._palm_light_controller.reset()
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

    def _attach_tracking_stats(self, snapshot: Any) -> None:
        hand_state = getattr(snapshot, "hand_state", None)
        if hand_state is not None:
            self.last_lighting_stats.update({
                "hand_luminance": float(getattr(hand_state, "frame_luminance", 255.0)),
                "low_light_active": float(bool(getattr(hand_state, "low_light_active", False))),
                "clahe_active": float(bool(getattr(hand_state, "low_light_active", False))),
                "low_light_gamma": float(getattr(hand_state, "low_light_gamma", 1.0)),
                "low_light_preprocess_ms": float(getattr(hand_state, "preprocess_ms", 0.0)),
                "mediapipe_ms": float(getattr(hand_state, "detection_ms", 0.0)),
                "mediapipe_avg_ms": float(getattr(hand_state, "detection_avg_ms", 0.0)),
                "hand_filtering_ms": float(getattr(hand_state, "filtering_ms", 0.0)),
                "dropout_age_ms": float(getattr(hand_state, "dropout_age_ms", 0.0)),
                "hand_tracking_fps": float(getattr(getattr(snapshot, "metrics", None), "hand_hz", 0.0) or 0.0),
            })
        if self._palm_light_controller.last_stats:
            self.last_lighting_stats.update(self._palm_light_controller.last_stats)

    def _low_geometry(self, geometry: GeometryState) -> GeometryState:
        full_h, full_w = geometry.depth.shape
        scale = min(1.0, self.max_width / full_w, self.max_height / full_h)
        width = max(2, int(round(full_w * scale)))
        height = max(2, int(round(full_h * scale)))
        camera = geometry.camera.scaled_intrinsics(width, height)
        source_valid = geometry.valid_mask & np.isfinite(geometry.depth) & (geometry.depth > 1e-6)
        coverage = cv2.resize(source_valid.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
        weighted_depth = cv2.resize(
            np.where(source_valid, geometry.depth, 0.0).astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        depth = weighted_depth / np.maximum(coverage, 1e-6)
        valid = (coverage >= 0.35) & np.isfinite(depth) & (depth > 1e-6)
        depth = np.where(valid, depth, np.nan).astype(np.float32)
        normals = None
        if geometry.normals is not None:
            normals = cv2.resize(np.nan_to_num(geometry.normals, nan=0.0).astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
            normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-6)
            normals[~valid] = 0.0
        confidence = geometry.confidence
        if confidence is not None:
            confidence = cv2.resize(np.asarray(confidence, dtype=np.float32), (width, height), interpolation=cv2.INTER_AREA)
            confidence = np.clip(np.nan_to_num(confidence), 0.0, 1.0)
        else:
            confidence = valid.astype(np.float32)
        uu, vv = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        positions = camera.unproject(uu, vv, depth).astype(np.float32)
        return GeometryState(
            timestamp=geometry.timestamp,
            source_frame_id=geometry.source_frame_id,
            depth=depth,
            positions_3d=positions,
            valid_mask=valid,
            camera=camera,
            scale_mode=geometry.scale_mode,
            normals=normals,
            confidence=confidence,
        )

    def _cache_key_for(self, snapshot: Any, geometry: GeometryState, lights: list[LightState]) -> tuple[Any, ...]:
        return (
            snapshot.rgb_capture_id,
            geometry.source_frame_id,
            tuple((light.light_id, tuple(np.round(light.position_camera, 3)), round(light.confidence, 2), round(light.orb_visibility, 2), light.enabled, round(light.intensity, 3), round(light.range_m, 3), tuple(np.round(light.color_rgb, 3))) for light in lights),
            int((self.last_xyz_source_age_ms or 0.0) // 25),
            self.lighting_quality,
        )

    def _cpu_render(self, frame: np.ndarray, geometry: GeometryState, lights: list[LightState]) -> np.ndarray:
        small_geometry = self._low_geometry(geometry)
        small_rgb = cv2.resize(frame, (small_geometry.camera.width, small_geometry.camera.height), interpolation=cv2.INTER_AREA)
        ambient = 0.40
        relit, stats = shade_geometry(
            small_rgb,
            small_geometry,
            lights,
            ambient=ambient,
            specular_strength=0.12,
            shininess=36.0,
            shadows=True,
            volumetrics=True,
        )
        result = _detail_preserving_composite(frame, small_rgb, relit, small_geometry, ambient=ambient)
        result = render_light_orbs(result, geometry.camera, lights, depth=geometry.depth, valid=geometry.valid_mask)
        stats.update({
            "renderer": "CPU_FALLBACK",
            "quality": self.lighting_quality,
            "shadow_quality": "CPU reference",
            "volumetric_quality": "CPU reference",
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

    def _start_gpu_warmup(self, renderer, frame: np.ndarray, geometry: GeometryState, lights: list[LightState]) -> None:
        if self._gpu_warm_active:
            return
        self._gpu_warm_active = True
        warm_frame = np.ascontiguousarray(frame.copy())

        def warm() -> None:
            try:
                renderer.render(warm_frame, geometry, lights, ambient=0.40, readback=False)
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
            self.last_lighting_stats = {"renderer": "WAITING"}
            self._attach_tracking_stats(snapshot)
            return None, title, "waiting for camera"
        frame = np.asarray(frame)[..., :3]
        geometry = self._geometry(snapshot)
        if geometry is None or geometry.normals is None:
            self.last_light_count = 0
            self.last_lighting_stats = {"renderer": "WAITING"}
            self._cache_image = frame.copy()
            self._attach_tracking_stats(snapshot)
            return self._cache_image, title, "waiting for surface geometry"

        self.last_geometry_source_id = geometry.source_frame_id
        self.last_geometry_age_ms = max(0.0, (time.monotonic() - geometry.timestamp) * 1000.0)
        lights, self.last_xyz_source_age_ms = lights_from_snapshot(
            snapshot,
            geometry=geometry,
            palm_controller=self._palm_light_controller,
            mirrored_input=bool(getattr(snapshot, "mirrored_input", True)),
        )
        lights = self._gesture_gated_lights(snapshot, lights)
        lights = self.controlled_lights(lights)
        key = self._cache_key_for(snapshot, geometry, lights)
        if key == self._cache_key and self._cache_image is not None:
            self._attach_tracking_stats(snapshot)
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
            self._attach_tracking_stats(snapshot)
            return self._cache_image, title, None

        started = time.perf_counter()
        renderer = self._ensure_gpu()
        if renderer is not None:
            if not self._gpu_warmed:
                self._start_gpu_warmup(renderer, frame, geometry, lights)
                self.last_light_count = len(lights)
                self.last_lighting_stats = {
                    "renderer": "GPU_WARMUP",
                    "quality": self.lighting_quality,
                    "lights": float(len(lights)),
                }
                self.last_render_ms = (time.perf_counter() - started) * 1000.0
                return frame.copy(), title, None
            try:
                image, stats = renderer.render(frame, geometry, lights, ambient=0.40)
                self.last_lighting_stats = stats
            except Exception as exc:
                self._gpu_error = f"{type(exc).__name__}: {exc}"
                self._gpu_error_reported = True
                print(f"GPU RELIGHT RENDER FAILED: {self._gpu_error}; falling back to CPU reference renderer")
                renderer.close()
                self._gpu_renderer = None
                self._gpu_error = f"render {type(exc).__name__}: {exc}"
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
        self._attach_tracking_stats(snapshot)
        for index, light in enumerate(lights):
            if not light.is_palm_attached and project_light_orb(geometry.camera, light) is None:
                continue
            position = light.position_camera
            state = "EMIT" if light.enabled and light.effective_intensity > 0.01 else "OFF"
            label = (
                f"H{light.source_hand} {state} Orb:{light.orb_visibility:.2f} "
                f"F:{light.palm_facing_score:+.2f} XYZ[{position[0]:+.2f},{position[1]:+.2f},{position[2]:.2f}]"
                if light.is_palm_attached
                else f"H{light.source_hand} XYZ [{position[0]:+.2f}, {position[1]:+.2f}, {position[2]:.2f}]m"
            )
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
    height = 66
    x = vx + (vw - width) // 2
    y = vy + vh - height - 12
    scale = width / 560.0
    return {
        "panel": (x, y, width, height),
        "intensity": (x + round(16 * scale), y + 46, round(148 * scale), 18),
        "range": (x + round(190 * scale), y + 46, round(148 * scale), 18),
        "color": (x + round(374 * scale), y + 12, round(170 * scale), 42),
    }


def _set_slider(control: str, x: int, rect: tuple[int, int, int, int]) -> None:
    x0, _y, width, _height = rect
    value = float(np.clip((x - x0) / max(width, 1), 0.0, 1.0))
    if control == "intensity":
        _renderer.light_intensity = MIN_INTENSITY + value * (MAX_INTENSITY - MIN_INTENSITY)
    else:
        _renderer.light_range_m = MIN_RANGE_M + value * (MAX_RANGE_M - MIN_RANGE_M)
    _renderer._cache_key = None


def handle_control_mouse(event: int, x: int, y: int, flags: int, display_size: tuple[int, int]) -> bool:
    """Handle Mode 7 slider drags and color selection clicks."""
    global _active_control
    from ..common import compute_layout

    layout = compute_layout(display_size)
    controls = _control_layout((layout["vx"], layout["vy"], layout["vw"], layout["vh"]))
    if not controls:
        return False
    cv = cv2
    if event == cv.EVENT_LBUTTONDOWN:
        for name in ("intensity", "range"):
            rx, ry, rw, rh = controls[name]
            if rx - 8 <= x <= rx + rw + 8 and ry - 12 <= y <= ry + rh + 6:
                _active_control = name
                _set_slider(name, x, controls[name])
                return True
        rx, ry, rw, rh = controls["color"]
        if rx <= x <= rx + rw and ry <= y <= ry + rh:
            _active_control = "color"
            return True
    elif event == cv.EVENT_MOUSEMOVE and flags & cv.EVENT_FLAG_LBUTTON:
        if _active_control in ("intensity", "range"):
            _set_slider(_active_control, x, controls[_active_control])
            return True
    elif event == cv.EVENT_LBUTTONUP:
        previous = _active_control
        _active_control = None
        if previous in ("intensity", "range"):
            _set_slider(previous, x, controls[previous])
            return True
        if previous == "color":
            rx, ry, rw, rh = controls["color"]
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                _renderer.light_color_index = (_renderer.light_color_index + 1) % len(COLOR_PRESETS)
                _renderer._cache_key = None
            return True
    return False


def draw_controls(canvas: np.ndarray, viewport: tuple[int, int, int, int]) -> None:
    """Paint the compact Mode 7 power, range, and color controls."""
    from ..common import (
        COLOR_BORDER_SUBTLE,
        COLOR_CONTAINER_LOW,
        COLOR_PRIMARY_ACCENT,
        COLOR_TEXT_PRIMARY,
        COLOR_TEXT_SECONDARY,
        draw_rounded_rect_alpha,
    )

    controls = _control_layout(viewport)
    if not controls:
        return
    panel = controls["panel"]
    draw_rounded_rect_alpha(canvas, *panel, 10, COLOR_CONTAINER_LOW, 0.93, COLOR_BORDER_SUBTLE)
    scale = panel[2] / 560.0
    font = 0.34 * min(1.0, max(0.82, scale))
    for name, value, low, high, label in (
        ("intensity", _renderer.light_intensity, MIN_INTENSITY, MAX_INTENSITY,
         f"INTENSITY {_renderer.light_intensity:.2f}"),
        ("range", _renderer.light_range_m, MIN_RANGE_M, MAX_RANGE_M,
         f"RANGE {_renderer.light_range_m:.2f}m"),
    ):
        rx, ry, rw, _rh = controls[name]
        cv2.putText(canvas, label, (rx, ry - 14), cv2.FONT_HERSHEY_SIMPLEX, font, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
        cy = ry + 1
        cv2.line(canvas, (rx, cy), (rx + rw, cy), COLOR_BORDER_SUBTLE, 5, cv2.LINE_AA)
        fraction = float(np.clip((value - low) / (high - low), 0.0, 1.0))
        filled = rx + round(rw * fraction)
        cv2.line(canvas, (rx, cy), (filled, cy), COLOR_PRIMARY_ACCENT, 5, cv2.LINE_AA)
        cv2.circle(canvas, (filled, cy), max(5, round(6 * scale)), COLOR_TEXT_PRIMARY, -1, cv2.LINE_AA)
    rx, ry, rw, rh = controls["color"]
    name, color = COLOR_PRESETS[_renderer.light_color_index]
    swatch = tuple(int(round(channel * 255)) for channel in color[::-1])
    cv2.rectangle(canvas, (rx + 10, ry + 13), (rx + 27, ry + 30), swatch, -1)
    cv2.putText(canvas, f"COLOR: {name}", (rx + 36, ry + 27), cv2.FONT_HERSHEY_SIMPLEX,
                font, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)


def configure(lighting_quality: str = "balanced") -> None:
    _renderer.set_lighting_quality(lighting_quality)


def render(snapshot: Any) -> tuple[np.ndarray | None, str, str | None]:
    return _renderer.render(snapshot)


__all__ = ["RelightRenderer", "lights_from_snapshot", "configure", "render"]
