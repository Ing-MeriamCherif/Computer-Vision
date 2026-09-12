"""Live hand tracking and gesture state for the geometry/renderer boundary.

This module ports the useful parts of the colleague hand branch (MediaPipe
Tasks/legacy backends, adaptive detector cadence, One-Euro smoothing and a
short optical-flow coast) without importing its fixed-depth or placeholder
renderer.  The output is deliberately independent from any lighting model so
the real :class:`~geometry.state.GeometryState` can supply depth.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class HandObservation:
    """One tracked hand in pixel coordinates of the processed RGB frame."""

    palm_uv: tuple[float, float]
    landmarks_uv: np.ndarray | None
    confidence: float
    handedness: str | None = None
    palm_width_px: float | None = None
    velocity_px_s: float = 0.0
    stale: bool = False


@dataclass(slots=True)
class GestureState:
    """Timestamped hand state consumed by the lighting stage."""

    timestamp: float
    source_frame_id: int | str
    hands: tuple[HandObservation, ...]
    backend: str
    tracker_ms: float
    stale: bool = False

    @property
    def active(self) -> bool:
        return bool(self.hands)


class _Backend(Protocol):
    name: str

    def process(self, rgb: np.ndarray) -> list[HandObservation]: ...

    def close(self) -> None: ...


def _palm_width(points: np.ndarray | None) -> float | None:
    if points is None or len(points) < 18:
        return None
    return float(np.linalg.norm(points[5] - points[17]))


class _NullBackend:
    name = "unavailable"

    def process(self, rgb: np.ndarray) -> list[HandObservation]:
        return []

    def close(self) -> None:
        return None


class _TasksBackend:
    name = "mediapipe-tasks"

    def __init__(self, model_path: str, max_hands: int) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base,
            num_hands=max_hands,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._mp = mp
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def process(self, rgb: np.ndarray) -> list[HandObservation]:
        h, w = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        observations: list[HandObservation] = []
        for index, hand in enumerate(result.hand_landmarks):
            points = np.asarray([[p.x * w, p.y * h] for p in hand], dtype=np.float64)
            handedness = None
            confidence = 1.0
            try:
                category = result.handedness[index][0]
                handedness = str(category.category_name or "") or None
                confidence = float(category.score)
            except Exception:
                pass
            palm = points[9]
            observations.append(HandObservation(
                (float(palm[0]), float(palm[1])), points, confidence,
                handedness, _palm_width(points),
            ))
        return observations

    def close(self) -> None:
        self._landmarker.close()


class _LegacyBackend:
    name = "mediapipe-legacy"

    def __init__(self, max_hands: int) -> None:
        import mediapipe as mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )

    def process(self, rgb: np.ndarray) -> list[HandObservation]:
        h, w = rgb.shape[:2]
        result = self._hands.process(rgb)
        output: list[HandObservation] = []
        for index, hand in enumerate(result.multi_hand_landmarks or []):
            points = np.asarray([[p.x * w, p.y * h] for p in hand.landmark], dtype=np.float64)
            handedness = None
            confidence = 1.0
            try:
                category = result.multi_handedness[index].classification[0]
                handedness = str(category.label)
                confidence = float(category.score)
            except Exception:
                pass
            palm = points[9]
            output.append(HandObservation(
                (float(palm[0]), float(palm[1])), points, confidence,
                handedness, _palm_width(points),
            ))
        return output

    def close(self) -> None:
        self._hands.close()


def create_hand_tracker(
    *, model_path: str | None = None, max_hands: int = 2, backend: str = "auto"
) -> _Backend:
    """Select a real local backend, returning a safe no-op if unavailable."""

    requested = backend.lower()
    path = model_path or os.getenv("MP_HAND_BUNDLE", "models/hand_landmarker.task")
    if requested in ("auto", "tasks") and os.path.exists(path):
        try:
            return _TasksBackend(path, max_hands)
        except Exception:
            if requested == "tasks":
                return _NullBackend()
    if requested in ("auto", "legacy"):
        try:
            return _LegacyBackend(max_hands)
        except Exception:
            pass
    return _NullBackend()


class _OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.3) -> None:
        self.min_cutoff, self.beta = float(min_cutoff), float(beta)
        self.x: float | None = None
        self.dx = 0.0
        self.t: float | None = None

    def __call__(self, value: float, timestamp: float) -> float:
        if self.t is None or self.x is None:
            self.t, self.x = timestamp, float(value)
            return self.x
        dt = max(timestamp - self.t, 1e-3)
        derivative = (float(value) - self.x) / dt
        a_d = (2.0 * np.pi * 1.0 * dt) / (2.0 * np.pi * 1.0 * dt + 1.0)
        self.dx = a_d * derivative + (1.0 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        a = (2.0 * np.pi * cutoff * dt) / (2.0 * np.pi * cutoff * dt + 1.0)
        self.x = a * float(value) + (1.0 - a) * self.x
        self.t = timestamp
        return self.x


class HandControlEngine:
    """Adaptive, smoothed hand state with bounded dropout recovery."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        max_hands: int = 2,
        backend: str = "auto",
        detect_every_n: int = 2,
        max_coast_frames: int = 6,
        input_size: tuple[int, int] = (320, 240),
    ) -> None:
        self.backend = create_hand_tracker(model_path=model_path, max_hands=max_hands, backend=backend)
        self.detect_every_n = max(1, int(detect_every_n))
        self.max_coast_frames = max(0, int(max_coast_frames))
        self.input_size = input_size
        self._last: list[HandObservation] = []
        self._filters: dict[int, tuple[_OneEuro, _OneEuro]] = {}
        self._previous_gray: np.ndarray | None = None
        self._coast = 0
        self._last_timestamp: float | None = None

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def update(self, rgb: np.ndarray, timestamp: float, frame_id: int | str) -> GestureState:
        import cv2

        started = time.perf_counter()
        source = np.asarray(rgb, dtype=np.uint8)[..., :3]
        height, width = source.shape[:2]
        iw, ih = self.input_size
        scale_x, scale_y = width / iw, height / ih
        small = cv2.resize(source, (iw, ih), interpolation=cv2.INTER_AREA)
        # Do not force a detector pass on every frame merely because the last
        # pass found no hand; that was the main FPS regression in the original
        # branch.  Always anchor frame zero, then follow the configured cadence.
        should_detect = self._last_timestamp is None or (
            isinstance(frame_id, int) and frame_id % self.detect_every_n == 0
        )
        observations: list[HandObservation] = []
        if should_detect:
            observations = self.backend.process(small)
            for obs in observations:
                obs.palm_uv = (obs.palm_uv[0] * scale_x, obs.palm_uv[1] * scale_y)
                if obs.landmarks_uv is not None:
                    obs.landmarks_uv = obs.landmarks_uv * np.array([scale_x, scale_y])
                if obs.palm_width_px is not None:
                    obs.palm_width_px *= (scale_x + scale_y) * 0.5
        else:
            observations = self._last

        if observations:
            filtered: list[HandObservation] = []
            for index, obs in enumerate(observations):
                old = self._last[index] if index < len(self._last) else None
                uv = obs.palm_uv
                if index not in self._filters:
                    self._filters[index] = (_OneEuro(), _OneEuro())
                fx, fy = self._filters[index]
                smooth_uv = (fx(uv[0], timestamp), fy(uv[1], timestamp))
                velocity = 0.0
                if old is not None:
                    velocity = float(np.linalg.norm(np.subtract(smooth_uv, old.palm_uv)) / max(timestamp - getattr(self, "_last_timestamp", timestamp), 1e-3))
                obs.palm_uv = smooth_uv
                obs.velocity_px_s = velocity
                obs.stale = not should_detect
                filtered.append(obs)
            self._last = filtered
            self._coast = 0
        elif self._last and self._coast < self.max_coast_frames and self._previous_gray is not None:
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            prior_gray = self._previous_gray
            coasted: list[HandObservation] = []
            for obs in self._last:
                point = np.asarray([[obs.palm_uv[0] / scale_x, obs.palm_uv[1] / scale_y]], dtype=np.float32)
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(prior_gray, gray, point, None, winSize=(31, 31), maxLevel=3)
                if status is not None and bool(status[0, 0]):
                    u, v = float(nxt[0, 0, 0] * scale_x), float(nxt[0, 0, 1] * scale_y)
                    if 0 <= u < width and 0 <= v < height:
                        obs.palm_uv = (u, v)
                        obs.confidence *= 0.75
                        obs.stale = True
                        coasted.append(obs)
            observations = coasted
            self._last = coasted
            self._coast += 1
        else:
            self._last = []
            self._coast = 0
        self._previous_gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        self._last_timestamp = timestamp
        elapsed = (time.perf_counter() - started) * 1000.0
        return GestureState(timestamp, frame_id, tuple(observations), self.backend_name, elapsed, any(o.stale for o in observations))

    def reset(self) -> None:
        self._last = []
        self._filters.clear()
        self._previous_gray = None
        self._coast = 0
        self._last_timestamp = None

    def close(self) -> None:
        self.backend.close()
