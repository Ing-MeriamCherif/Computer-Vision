"""3D palm pose reconstruction and independently visible palm lights."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .depth_sampling import sample_depth
from .lighting import LightState
from .state import GeometryState


PALM_LIGHT_OFFSET_M = 0.035
SELF_INTERSECTION_EPSILON_M = 0.002
PALM_FACING_OFF = -0.20
PALM_FACING_ON = 0.08
PALM_FACING_FULL = 0.45
HAND_LOST_HOLD_SECONDS = 0.16


def _unit(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < 1e-6:
        return None
    return np.asarray(vector, dtype=np.float32) / length


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    t = float(np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _slerp(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    a = _unit(first)
    b = _unit(second)
    if a is None or b is None:
        return second
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    amount = float(np.clip(alpha, 0.0, 1.0))
    if dot < -0.999:
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(float(a[0])) < 0.8 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis = _unit(np.cross(a, basis))
        if axis is None:
            return b
        angle = math.pi * amount
        result = _unit(a * math.cos(angle) + axis * math.sin(angle))
        return b if result is None else result
    angle = math.acos(dot)
    if angle < 1e-4:
        result = _unit((1.0 - amount) * a + amount * b)
        return b if result is None else result
    scale = math.sin(angle)
    result = _unit(a * (math.sin((1.0 - amount) * angle) / scale) + b * (math.sin(amount * angle) / scale))
    return b if result is None else result


def _handedness_sign(handedness: str | None, mirrored_input: bool) -> float:
    """Orient cross(pinky-index, middle-wrist) outward for MediaPipe labels."""
    label = str(handedness or "").strip().lower()
    if label.startswith("right"):
        return -1.0 if mirrored_input else 1.0
    if label.startswith("left"):
        return 1.0 if mirrored_input else -1.0
    return 0.0


class _OneEuroVector:
    def __init__(self, min_cutoff: float, beta: float, derivative_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self.value: np.ndarray | None = None
        self.derivative = np.zeros(3, dtype=np.float32)
        self.timestamp: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / max(dt, 1e-3))

    def update(self, value: np.ndarray, timestamp: float, confidence: float) -> np.ndarray:
        current = np.asarray(value, dtype=np.float32).reshape(3)
        if self.value is None or self.timestamp is None or timestamp - self.timestamp > 0.35:
            self.value = current.copy()
            self.derivative.fill(0.0)
            self.timestamp = timestamp
            return self.value.copy()

        dt = max(timestamp - self.timestamp, 1e-3)
        derivative = (current - self.value) / dt
        derivative_alpha = self._alpha(self.derivative_cutoff, dt)
        self.derivative = derivative_alpha * derivative + (1.0 - derivative_alpha) * self.derivative
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.derivative))
        alpha = self._alpha(cutoff, dt) * float(np.clip(0.4 + 0.6 * confidence, 0.25, 1.0))
        self.value = alpha * current + (1.0 - alpha) * self.value
        self.timestamp = timestamp
        return self.value.copy()


class PalmLightController:
    """Build filtered light states; palm facing controls only the visible orb."""

    def __init__(self, *, offset_m: float = PALM_LIGHT_OFFSET_M) -> None:
        self.offset_m = max(0.0, float(offset_m))
        self._history: dict[int, dict[str, Any]] = {}
        self.last_stats: dict[str, float] = {"palm_pose_ms": 0.0, "filtering_ms": 0.0}

    def reset(self) -> None:
        self._history.clear()
        self.last_stats = {"palm_pose_ms": 0.0, "filtering_ms": 0.0}

    def update(
        self,
        hand: Any,
        geometry: GeometryState,
        *,
        fallback_position: np.ndarray | None = None,
        mirrored_input: bool = True,
        intensity: float = 0.72,
        color_rgb: tuple[float, float, float] = (1.0, 0.78, 0.48),
        range_m: float = 0.30,
        source_hand: int | None = None,
        light_id: int | str | None = None,
        timestamp: float | None = None,
    ) -> LightState:
        started = time.perf_counter()
        hand_id = int(getattr(hand, "hand_id", 0) if source_hand is None else source_hand)
        hand_timestamp = getattr(hand, "timestamp", None)
        now = float(timestamp if timestamp is not None else (hand_timestamp if hand_timestamp is not None else time.monotonic()))
        history = self._history.get(hand_id)
        pose_started = time.perf_counter()
        landmarks = getattr(hand, "landmarks_uv", None)
        points = np.asarray(landmarks, dtype=np.float32) if landmarks is not None else np.empty((0, 2), dtype=np.float32)
        samples: dict[int, tuple[np.ndarray, float]] = {}
        if points.ndim == 2 and points.shape[1] == 2 and len(points) >= 18:
            for index in (0, 5, 9, 13, 17):
                uv = points[index]
                if not np.isfinite(uv).all() or not (0 <= uv[0] < geometry.camera.width and 0 <= uv[1] < geometry.camera.height):
                    continue
                z, reliability = sample_depth(geometry.depth, geometry.valid_mask, float(uv[0]), float(uv[1]), radius=2)
                if z <= 0 or reliability <= 0 or not math.isfinite(z):
                    continue
                xyz = geometry.camera.unproject(float(uv[0]), float(uv[1]), z).astype(np.float32)
                if np.isfinite(xyz).all():
                    samples[index] = (xyz, float(reliability))

        mcp_xyz = [samples[index][0] for index in (5, 9, 13, 17) if index in samples]
        if len(mcp_xyz) >= 2:
            measured_center = np.median(np.stack(mcp_xyz), axis=0).astype(np.float32)
            center_confidence = float(np.mean([samples[i][1] for i in (5, 9, 13, 17) if i in samples]))
        elif fallback_position is not None:
            measured_center = np.asarray(fallback_position, dtype=np.float32).reshape(3).copy()
            center_confidence = 0.35
        elif history is not None and now - history["timestamp"] <= HAND_LOST_HOLD_SECONDS:
            measured_center = history["center"].copy()
            center_confidence = max(0.0, history["orientation_confidence"] * 0.6)
        else:
            uv = getattr(hand, "palm_uv", (geometry.camera.cx, geometry.camera.cy))
            z = getattr(hand, "depth_z", None)
            if z is None or not math.isfinite(float(z)) or float(z) <= 0:
                z = 0.7
            measured_center = geometry.camera.unproject(float(uv[0]), float(uv[1]), float(z)).astype(np.float32)
            center_confidence = 0.15

        normal: np.ndarray | None = None
        pose_confidence = 0.0
        sign = _handedness_sign(getattr(hand, "handedness", None), mirrored_input)
        if sign != 0 and all(index in samples for index in (0, 5, 9, 17)):
            across = samples[17][0] - samples[5][0]
            along = samples[9][0] - samples[0][0]
            normal = _unit(np.cross(across, along) * sign)
            if normal is not None and np.linalg.norm(across) > 1e-4 and np.linalg.norm(along) > 1e-4:
                span_quality = float(np.clip(np.linalg.norm(across) / 0.04, 0.2, 1.0))
                sample_quality = min(samples[i][1] for i in (0, 5, 9, 17))
                pose_confidence = float(np.clip(sample_quality * span_quality * center_confidence, 0.0, 1.0))
            else:
                normal = None
        pose_ms = (time.perf_counter() - pose_started) * 1000.0
        raw_center = measured_center.copy()
        raw_normal = None if normal is None else normal.copy()

        filter_started = time.perf_counter()
        hand_confidence = float(np.clip(getattr(hand, "confidence", 1.0), 0.0, 1.0))
        delta = max(0.0, now - history["timestamp"]) if history is not None else 1.0 / 30.0
        if history is not None and delta > HAND_LOST_HOLD_SECONDS:
            history = None
            self._history.pop(hand_id, None)

        if history is None:
            center_filter = _OneEuroVector(1.2, 1.2)
            filtered_center = center_filter.update(measured_center, now, center_confidence)
        else:
            center_filter = history["center_filter"]
            filtered_center = center_filter.update(measured_center, now, center_confidence)
            if HAND_LOST_HOLD_SECONDS * 0.5 < delta <= HAND_LOST_HOLD_SECONDS:
                filtered_center = history["center"] + 0.62 * (filtered_center - history["center"])
                center_filter.value = filtered_center.copy()

        pending_flip = None
        pending_flip_count = 0
        confirmed_flip = False
        if normal is not None and pose_confidence >= 0.08:
            if history is not None and delta <= 0.12:
                dot = float(np.dot(history["normal"], normal))
                if dot < -0.92 and pose_confidence >= 0.45:
                    candidate = history.get("pending_flip")
                    if candidate is not None and float(np.dot(candidate, normal)) > 0.92:
                        pending_flip_count = int(history.get("pending_flip_count", 0)) + 1
                        pending_flip = normal.copy()
                    else:
                        pending_flip_count = 1
                        pending_flip = normal.copy()
                    if pending_flip_count < 2:
                        normal = -normal
                        dot = -dot
                    else:
                        pending_flip = None
                        pending_flip_count = 0
                        confirmed_flip = True
                if dot > -0.80:
                    angle = math.acos(float(np.clip(dot, -1.0, 1.0)))
                    alpha = float(np.clip(0.30 + 0.08 * angle / max(delta, 1e-3), 0.30, 0.68))
                    alpha *= float(np.clip(0.45 + 0.55 * pose_confidence, 0.45, 1.0))
                    normal = _slerp(history["normal"], normal, alpha)
                elif confirmed_flip:
                    normal = _slerp(history["normal"], normal, 0.35)

            to_camera = _unit(-filtered_center)
            facing = float(np.dot(normal, to_camera)) if to_camera is not None else -1.0
            if not math.isfinite(facing):
                facing = -1.0
            was_facing = bool(history["active"]) if history is not None else False
            facing_latch = facing > (PALM_FACING_OFF if was_facing else PALM_FACING_ON)
            orientation_visibility = _smoothstep(PALM_FACING_OFF, PALM_FACING_FULL, facing) if facing_latch else 0.0
            if history is not None and delta <= HAND_LOST_HOLD_SECONDS:
                visibility_alpha = float(np.clip(delta / (0.05 + delta), 0.25, 0.82))
                orientation_visibility = (
                    (1.0 - visibility_alpha) * history["orientation_visibility"]
                    + visibility_alpha * orientation_visibility
                )
        else:
            facing_latch = bool(history["active"]) if history is not None else False
            facing = float(history["facing"]) if history is not None else -1.0
            if history is not None and delta <= HAND_LOST_HOLD_SECONDS:
                normal = history["normal"].copy()
                pose_confidence = history["orientation_confidence"] * max(0.0, 1.0 - delta / HAND_LOST_HOLD_SECONDS)
                orientation_visibility = history["orientation_visibility"] * max(0.0, 1.0 - delta / HAND_LOST_HOLD_SECONDS)
            else:
                orientation_visibility = 0.0

        tracking_visibility = float(np.clip(hand_confidence, 0.0, 1.0))
        orb_visibility = float(np.clip(orientation_visibility * tracking_visibility, 0.0, 1.0))
        light_position = filtered_center if normal is None else filtered_center + normal * self.offset_m
        valid_position = np.isfinite(light_position).all() and light_position[2] > 0.0
        self._history[hand_id] = {
            "center": filtered_center.copy(),
            "center_filter": center_filter,
            "normal": np.zeros(3, dtype=np.float32) if normal is None else normal.copy(),
            "orientation_confidence": pose_confidence,
            "facing": facing,
            "orientation_visibility": orientation_visibility,
            "active": facing_latch,
            "pending_flip": pending_flip if normal is not None else None,
            "pending_flip_count": pending_flip_count,
            "timestamp": now,
        }
        filtering_ms = (time.perf_counter() - filter_started) * 1000.0
        self.last_stats = {
            "palm_pose_ms": pose_ms,
            "filtering_ms": filtering_ms,
            "orb_visibility": orb_visibility,
            "tracking_confidence": hand_confidence,
            "orientation_confidence": float(np.clip(pose_confidence, 0.0, 1.0)),
            "raw_light_xyz": tuple(float(x) for x in (raw_center if raw_normal is None else raw_center + raw_normal * self.offset_m)),
            "filtered_light_xyz": tuple(float(x) for x in light_position),
        }

        return LightState(
            position_camera=light_position.astype(np.float32),
            intensity=float(intensity),
            color_rgb=np.asarray(color_rgb, dtype=np.float32),
            confidence=hand_confidence,
            source_hand=hand_id,
            timestamp=now,
            enabled=bool(valid_position and hand_confidence >= 0.15),
            light_id=hand_id if light_id is None else light_id,
            range_m=float(range_m),
            orb_visibility=orb_visibility,
            is_palm_attached=True,
            palm_center_camera=filtered_center.astype(np.float32),
            palm_normal_camera=None if normal is None else normal.astype(np.float32),
            palm_facing_score=float(facing),
            orientation_confidence=float(np.clip(pose_confidence, 0.0, 1.0)),
            self_intersection_epsilon_m=SELF_INTERSECTION_EPSILON_M,
        )


__all__ = [
    "PALM_LIGHT_OFFSET_M", "SELF_INTERSECTION_EPSILON_M", "PALM_FACING_OFF", "PALM_FACING_ON",
    "PALM_FACING_FULL", "HAND_LOST_HOLD_SECONDS", "PalmLightController",
]
