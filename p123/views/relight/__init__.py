"""Mode 7: cached, hand-driven relighting within the P123 Material UI."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from geometry.lighting import LightState, light_from_palm, project_light_orb, render_light_orbs, shade_geometry
from geometry.state import GeometryState


def _detail_preserving_composite(
    full_rgb: np.ndarray,
    small_rgb: np.ndarray,
    small_relit: np.ndarray,
    geometry: GeometryState,
    *,
    ambient: float,
) -> np.ndarray:
    """Upsample only bounded illumination, preserving the camera's fine detail."""
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
    """Render one bounded-resolution P4 preview per new source snapshot."""

    def __init__(self, max_width: int = 192, max_height: int = 144) -> None:
        self.max_width = max(64, int(max_width))
        self.max_height = max(48, int(max_height))
        self._cache_key: tuple[Any, ...] | None = None
        self._cache_image: np.ndarray | None = None
        self.last_render_ms = 0.0
        self.last_light_count = 0
        self.last_lighting_stats: dict[str, float] = {}

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
            normals = cv2.resize(
                np.nan_to_num(geometry.normals, nan=0.0).astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_AREA,
            )
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

    def render(self, snapshot: Any) -> tuple[np.ndarray | None, str, str | None]:
        frame = snapshot.rgb_frame
        title = "MODE 7 - HAND-HELD RELIGHT"
        if frame is None:
            return None, title, "waiting for camera"

        geometry = snapshot.geometry_state
        hands = snapshot.hand_state.hands if snapshot.hand_state is not None else ()
        key = (
            id(frame),
            snapshot.rgb_capture_id,
            None if geometry is None else geometry.source_frame_id,
            None if snapshot.hand_state is None else snapshot.hand_state.source_frame_id,
        )
        if key == self._cache_key and self._cache_image is not None:
            waiting = None if self.last_light_count else "present a hand to control the light"
            return self._cache_image, title, waiting

        started = time.perf_counter()
        if geometry is None or geometry.normals is None:
            self._cache_image = np.asarray(frame).copy()
            self.last_light_count = 0
            self.last_lighting_stats = {}
            waiting = "waiting for surface geometry"
        else:
            colors = ((0.44, 0.72, 0.82), (0.88, 0.63, 0.40))
            lights: list[LightState] = []
            for index, hand in enumerate(hands[:2]):
                if hand.confidence < 0.15:
                    continue
                light = light_from_palm(
                    geometry,
                    hand.palm_uv,
                    hand.confidence,
                    color_rgb=colors[index % len(colors)],
                    intensity=0.70,
                    range_m=0.45,
                    source_hand=hand.hand_id,
                    timestamp=hand.timestamp,
                )
                if light is not None:
                    lights.append(light)

            small_geometry = self._low_geometry(geometry)
            small_rgb = cv2.resize(np.asarray(frame), (small_geometry.camera.width, small_geometry.camera.height), interpolation=cv2.INTER_AREA)
            ambient = 0.40
            relit, self.last_lighting_stats = shade_geometry(
                small_rgb,
                small_geometry,
                lights,
                ambient=ambient,
                specular_strength=0.12,
                shininess=36.0,
                shadows=True,
                volumetrics=True,
            )
            self._cache_image = _detail_preserving_composite(
                frame,
                small_rgb,
                relit,
                small_geometry,
                ambient=ambient,
            )
            relit = render_light_orbs(
                self._cache_image,
                geometry.camera,
                lights,
                depth=geometry.depth,
                valid=geometry.valid_mask,
            )
            self._cache_image = relit
            self.last_light_count = len(lights)
            for light_index, light in enumerate(lights):
                projected = project_light_orb(geometry.camera, light)
                if projected is None:
                    continue
                u, v, _ = projected
                position = light.position_camera
                label = f"H{light.source_hand} XYZ [{position[0]:+.2f}, {position[1]:+.2f}, {position[2]:.2f}]m  UV [{u:.0f}, {v:.0f}]"
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                x = max(8, self._cache_image.shape[1] - text_w - 18)
                y = 24 + 28 * light_index
                cv2.rectangle(self._cache_image, (x - 6, y - text_h - 4), (x + text_w + 6, y + baseline + 3), (20, 26, 32), -1)
                cv2.putText(self._cache_image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (224, 231, 239), 1, cv2.LINE_AA)
            waiting = None if lights else "present a hand to control the light"

        self._cache_key = key
        self.last_render_ms = (time.perf_counter() - started) * 1000.0
        return self._cache_image, title, waiting


_renderer = RelightRenderer()


def render(snapshot: Any) -> tuple[np.ndarray | None, str, str | None]:
    return _renderer.render(snapshot)
