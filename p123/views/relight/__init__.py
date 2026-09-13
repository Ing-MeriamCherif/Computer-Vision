"""Mode 7: live P123 HandXYZ-controlled GPU relighting with CPU reference fallback."""

from __future__ import annotations

import time
import threading
import math
from typing import Any

import cv2
import numpy as np

from geometry.lighting import LightState, project_light_orb, render_light_orbs, shade_geometry
from geometry.state import GeometryState


LIGHT_COLORS = ((0.44, 0.72, 0.82), (0.88, 0.63, 0.40))
DEFAULT_RANGE_M = 0.30
DEFAULT_INTENSITY = 0.70
FRESHNESS_LIMIT_MS = 250.0
FADE_START_MS = 150.0


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
        age_value = xyz.source_age_ms if getattr(xyz, "source_age_ms", None) is not None else xyz.age_ms
        age_ms = float(age_value) if age_value is not None else float("inf")
        age_ms = max(0.0, age_ms) if math.isfinite(age_ms) else float("inf")
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

    def _gesture_gated_lights(self, snapshot: Any, lights: list[LightState]) -> list[LightState]:
        hands = snapshot.hand_state.hands if snapshot.hand_state is not None else ()
        visible_ids = {int(hand.hand_id) for hand in hands}
        for hand in hands:
            hand_id = int(hand.hand_id)
            # Only detector frames change the latch. Optical-flow frames keep
            # the last deliberate open/closed state.
            if not getattr(hand, "stale", False):
                state = _open_palm_state(hand)
                if state is not None:
                    self._gesture_enabled[hand_id] = state
        for hand_id in tuple(self._gesture_enabled):
            if hand_id not in visible_ids:
                self._gesture_enabled.pop(hand_id, None)
        return [light for light in lights if self._gesture_enabled.get(int(light.source_hand), True)]

    def _geometry(self, snapshot: Any) -> GeometryState | None:
        return getattr(snapshot, "fast_geometry_state", None) or snapshot.geometry_state

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
            tuple((light.light_id, tuple(np.round(light.position_camera, 3)), round(light.confidence, 2)) for light in lights),
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
        lights = self._gesture_gated_lights(snapshot, lights)
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


def configure(lighting_quality: str = "balanced") -> None:
    _renderer.set_lighting_quality(lighting_quality)


def render(snapshot: Any) -> tuple[np.ndarray | None, str, str | None]:
    return _renderer.render(snapshot)


__all__ = ["RelightRenderer", "lights_from_snapshot", "configure", "render"]
