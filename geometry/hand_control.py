"""Live hand tracking and gesture state for the geometry/renderer boundary.

Supports single-hand (L3) and true two-hand (L5) tracking with One-Euro / EMA
smoothing, continuous optical-flow coasting, and depth-fused hand Z coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Protocol

import numpy as np

from .depth_sampling import camera_uv_to_depth_uv, sample_depth
from .low_light import AdaptiveLowLightPreprocessor


@dataclass(slots=True)
class TrackedHand:
    """One tracked hand in pixel coordinates of the processed RGB frame."""

    hand_id: int
    landmarks_uv: np.ndarray | None
    palm_uv: tuple[float, float]
    confidence: float
    depth_z: float | None = None
    timestamp: float = 0.0
    velocity_px_s: float = 0.0
    handedness: str | None = None
    palm_width_px: float | None = None
    stale: bool = False
    depth_confidence: float = 0.0


# Backward-compatible alias for existing tests
HandObservation = TrackedHand


@dataclass(slots=True)
class GestureState:
    """Timestamped hand state consumed by the lighting stage."""

    timestamp: float
    source_frame_id: int | str
    hands: tuple[TrackedHand, ...]
    backend: str
    tracker_ms: float
    stale: bool = False
    frame_luminance: float = 255.0
    low_light_active: bool = False
    low_light_gamma: float = 1.0
    preprocess_ms: float = 0.0
    detection_ms: float = 0.0
    filtering_ms: float = 0.0
    dropout_age_ms: float = 0.0
    detection_avg_ms: float = 0.0

    @property
    def active(self) -> bool:
        return bool(self.hands)


def transform_hand_uv(
    uv: tuple[float, float],
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> tuple[float, float]:
    """Transform pixel coordinates between input, camera, and render resolutions."""
    scale_x = to_size[0] / max(from_size[0], 1)
    scale_y = to_size[1] / max(from_size[1], 1)
    return (float(uv[0] * scale_x), float(uv[1] * scale_y))


class _Backend(Protocol):
    name: str

    def process(self, rgb: np.ndarray) -> list[TrackedHand]: ...

    def close(self) -> None: ...


def _palm_width(points: np.ndarray | None) -> float | None:
    if points is None or len(points) < 18:
        return None
    return float(np.linalg.norm(points[5] - points[17]))


class _NullBackend:
    name = "unavailable"

    def process(self, rgb: np.ndarray) -> list[TrackedHand]:
        return []

    def close(self) -> None:
        return None


class _ColleagueBackend:
    """Adapter around the colleague repository's actual ``create_tracker``.

    The upstream tracker returns one ``HandResult`` (with pixel landmarks).
    Converting it here keeps the rest of our geometry state multi-hand ready
    while ensuring the live path executes the colleague implementation.
    """

    def __init__(self, model_path: str | None, max_hands: int) -> None:
        import os as _os
        from integrations.colleague_hand import hand_tracker as upstream

        if model_path and os.path.exists(model_path):
            self._old_bundle = _os.environ.get("MP_HAND_BUNDLE")
            _os.environ["MP_HAND_BUNDLE"] = model_path
        else:
            self._old_bundle = None
        upstream.config.HAND_BACKEND = "tasks"
        self._tracker = upstream.create_tracker(max_hands=max_hands)
        if self._tracker.backend_name == "mock":
            self._tracker.close()
            raise RuntimeError("colleague tracker fell back to synthetic MockTracker")
        self._upstream = upstream
        self.name = f"colleague-{self._tracker.backend_name}"

    def process(self, rgb: np.ndarray) -> list[TrackedHand]:
        result = self._tracker.process(rgb)
        if not result.found or result.palm_uv is None:
            return []
        return [TrackedHand(
            hand_id=0,
            palm_uv=(float(result.palm_uv[0]), float(result.palm_uv[1])),
            landmarks_uv=None if result.landmarks_uv is None else np.asarray(result.landmarks_uv, dtype=np.float64),
            confidence=float(result.confidence),
            handedness=getattr(result, "handedness", None),
            palm_width_px=result.palm_px,
        )]

    def close(self) -> None:
        self._tracker.close()
        if self._old_bundle is not None:
            os.environ["MP_HAND_BUNDLE"] = self._old_bundle
        else:
            os.environ.pop("MP_HAND_BUNDLE", None)


class _TasksBackend:
    name = "mediapipe-tasks"

    def __init__(self, model_path: str, max_hands: int) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base,
            num_hands=max(1, int(max_hands)),
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp = mp
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def process(self, rgb: np.ndarray) -> list[TrackedHand]:
        h, w = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        observations: list[TrackedHand] = []
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
            if confidence < 0.45:
                continue
            observations.append(TrackedHand(
                hand_id=index,
                landmarks_uv=points,
                palm_uv=(float(palm[0]), float(palm[1])),
                confidence=confidence,
                handedness=handedness,
                palm_width_px=_palm_width(points),
            ))
        return observations

    def close(self) -> None:
        if hasattr(self, "_landmarker") and self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None


class _LegacyBackend:
    name = "mediapipe-legacy"

    def __init__(self, max_hands: int) -> None:
        import mediapipe as mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max(1, int(max_hands)),
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )

    def process(self, rgb: np.ndarray) -> list[TrackedHand]:
        h, w = rgb.shape[:2]
        result = self._hands.process(rgb)
        output: list[TrackedHand] = []
        for index, hand in enumerate(result.multi_hand_landmarks or []):
            pts = np.array([[p.x * w, p.y * h] for p in hand.landmark], dtype=np.float64)
            palm = pts[9]
            conf = 1.0
            handedness = None
            if result.multi_handedness and index < len(result.multi_handedness):
                try:
                    cat = result.multi_handedness[index].classification[0]
                    conf = float(cat.score)
                    handedness = str(cat.label)
                except Exception:
                    pass
            output.append(TrackedHand(
                hand_id=index,
                landmarks_uv=pts,
                palm_uv=(float(palm[0]), float(palm[1])),
                confidence=conf,
                handedness=handedness,
                palm_width_px=_palm_width(pts),
            ))
        return output

    def close(self) -> None:
        self._hands.close()


def create_hand_tracker(
    *, model_path: str | None = None, max_hands: int = 2, backend: str = "auto"
) -> _Backend:
    """Select a real local backend, supporting 1 or 2 simultaneous hands."""
    requested = backend.lower()
    path = model_path or os.getenv("MP_HAND_BUNDLE", "models/hand_landmarker.task")
    if requested in ("auto", "tasks") and os.path.exists(path):
        try:
            return _TasksBackend(path, max_hands)
        except Exception:
            if requested == "tasks":
                return _NullBackend()
    if requested in ("auto", "colleague"):
        try:
            return _ColleagueBackend(path, max_hands)
        except Exception:
            if requested == "colleague":
                return _NullBackend()
    if requested in ("auto", "legacy"):
        try:
            return _LegacyBackend(max_hands)
        except Exception:
            pass
    return _NullBackend()


class _OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.3) -> None:
        try:
            from integrations.colleague_hand.filters import OneEuroFilter
            self._filter = OneEuroFilter(mincutoff=min_cutoff, beta=beta)
        except Exception:
            self._filter = None
        self.min_cutoff, self.beta = float(min_cutoff), float(beta)
        self.x: float | None = None
        self.dx = 0.0
        self.t: float | None = None

    def __call__(self, value: float, timestamp: float) -> float:
        if self._filter is not None:
            return float(self._filter(float(value), float(timestamp)))
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
    """Adaptive, smoothed multi-hand state with bounded dropout recovery and depth fusion."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        max_hands: int = 2,
        backend: str = "auto",
        detect_every_n: int = 2,
        max_coast_frames: int = 6,
        input_size: tuple[int, int] = (320, 240),
        filter_mode: str = "oneeuro",
        dropout_grace_ms: float = 120.0,
    ) -> None:
        self.max_hands = max(1, int(max_hands))
        self.backend = create_hand_tracker(model_path=model_path, max_hands=self.max_hands, backend=backend)
        self.detect_every_n = max(1, int(detect_every_n))
        self.max_coast_frames = max(0, int(max_coast_frames))
        self.input_size = input_size
        self.filter_mode = str(filter_mode).lower()
        self.dropout_grace_seconds = max(0.0, float(dropout_grace_ms)) / 1000.0
        self.preprocessor = AdaptiveLowLightPreprocessor()
        self._last: list[TrackedHand] = []
        self._filters: dict[int, tuple[_OneEuro, _OneEuro, _OneEuro]] = {}
        self._ema_filters: dict[int, object] = {}
        self._previous_gray: np.ndarray | None = None
        self._coast = 0
        self._last_timestamp: float | None = None
        self._last_detected_at: dict[int, float] = {}
        self._reacquire_pending: set[int] = set()
        self._reacquire_remaining: dict[int, int] = {}
        self._last_detection_ms = 0.0
        self._detection_ms_total = 0.0
        self._detection_samples = 0
        self.last_detection_ran = False
        self.lk_updates = 0
        self.lk_motion_px = 0.0

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def _assign_stable_ids(self, raw_obs: list[TrackedHand]) -> list[TrackedHand]:
        """Greedy matching of new detections to previous hands to maintain stable hand_ids."""
        if not self._last:
            for idx, obs in enumerate(raw_obs[:self.max_hands]):
                obs.hand_id = idx
            return raw_obs[:self.max_hands]

        assigned: list[TrackedHand] = []
        available_ids = list(range(self.max_hands))
        used_new = set()

        # Prioritize matching existing hands
        for old in self._last:
            best_idx = None
            best_dist = float("inf")
            for n_idx, n_obs in enumerate(raw_obs):
                if n_idx in used_new:
                    continue
                dist = float(np.hypot(n_obs.palm_uv[0] - old.palm_uv[0], n_obs.palm_uv[1] - old.palm_uv[1]))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = n_idx

            if best_idx is not None and best_dist < 200.0:
                n_obs = raw_obs[best_idx]
                n_obs.hand_id = old.hand_id
                if old.hand_id in available_ids:
                    available_ids.remove(old.hand_id)
                assigned.append(n_obs)
                used_new.add(best_idx)

        # Assign remaining detections to available IDs
        for n_idx, n_obs in enumerate(raw_obs):
            if n_idx not in used_new and available_ids:
                n_obs.hand_id = available_ids.pop(0)
                assigned.append(n_obs)

        return assigned

    def update(
        self,
        rgb: np.ndarray,
        timestamp: float,
        frame_id: int | str,
        *,
        depth_map: np.ndarray | None = None,
    ) -> GestureState:
        import cv2

        started = time.perf_counter()
        source = np.asarray(rgb, dtype=np.uint8)[..., :3]
        height, width = source.shape[:2]
        iw, ih = self.input_size
        scale_x, scale_y = width / iw, height / ih
        small = cv2.resize(source, (iw, ih), interpolation=cv2.INTER_AREA)
        tracking = self.preprocessor.process(small)
        tracker_frame = tracking.rgb

        should_detect = self._last_timestamp is None or (
            isinstance(frame_id, int) and frame_id % self.detect_every_n == 0
        )
        self.last_detection_ran = bool(should_detect)
        observations: list[TrackedHand] = []
        detection_ms = 0.0

        if should_detect:
            detection_started = time.perf_counter()
            raw_observations = self.backend.process(tracker_frame)
            detection_ms = (time.perf_counter() - detection_started) * 1000.0
            self._last_detection_ms = detection_ms
            self._detection_ms_total += detection_ms
            self._detection_samples += 1
            for obs in raw_observations:
                obs.palm_uv = (obs.palm_uv[0] * scale_x, obs.palm_uv[1] * scale_y)
                if obs.landmarks_uv is not None:
                    obs.landmarks_uv = obs.landmarks_uv * np.array([scale_x, scale_y])
                if obs.palm_width_px is not None:
                    obs.palm_width_px *= (scale_x + scale_y) * 0.5
            observations = self._assign_stable_ids([obs for obs in raw_observations if obs.confidence >= 0.45])
            for obs in observations:
                if obs.hand_id in self._reacquire_pending:
                    self._reacquire_remaining[obs.hand_id] = 3
                    self._reacquire_pending.discard(obs.hand_id)
                self._last_detected_at[obs.hand_id] = timestamp
        else:
            detection_ms = self._last_detection_ms
            # Detector-skip frames still update coordinates using LK optical
            # flow. Reusing the previous UV was a false 30 Hz claim and made
            # fast hands appear frozen between detector passes.
            observations = []
            if self._last and self._previous_gray is not None:
                gray = cv2.cvtColor(tracker_frame, cv2.COLOR_RGB2GRAY)
                for prior in self._last:
                    point = np.asarray([[prior.palm_uv[0] / scale_x, prior.palm_uv[1] / scale_y]], dtype=np.float32)
                    nxt, status, _ = cv2.calcOpticalFlowPyrLK(self._previous_gray, gray, point, None, winSize=(31, 31), maxLevel=3)
                    if status is not None and bool(status[0, 0]):
                        tracked = np.asarray(nxt).reshape(-1, 2)[0]
                        u, v = float(tracked[0] * scale_x), float(tracked[1] * scale_y)
                        if 0 <= u < width and 0 <= v < height:
                            delta = float(np.hypot(u - prior.palm_uv[0], v - prior.palm_uv[1]))
                            self.lk_updates += 1
                            self.lk_motion_px += delta
                            observations.append(TrackedHand(
                                hand_id=prior.hand_id,
                                landmarks_uv=None if prior.landmarks_uv is None else prior.landmarks_uv.copy(),
                                palm_uv=(u, v), confidence=prior.confidence * 0.95,
                                depth_z=prior.depth_z, timestamp=timestamp,
                                velocity_px_s=prior.velocity_px_s, handedness=prior.handedness,
                                palm_width_px=prior.palm_width_px, stale=True,
                                depth_confidence=prior.depth_confidence,
                            ))

        filtering_started = time.perf_counter()
        if observations:
            filtered: list[TrackedHand] = []
            for obs in observations:
                hid = obs.hand_id
                old = next((h for h in self._last if h.hand_id == hid), None)
                uv = obs.palm_uv

                # Depth estimation from depth map if available
                hand_z = obs.depth_z
                if depth_map is not None:
                    depth_arr = np.asarray(depth_map)
                    if depth_arr.ndim != 2:
                        raise ValueError("depth_map must be a 2D array")
                    depth_uv = camera_uv_to_depth_uv(uv, (width, height), (depth_arr.shape[1], depth_arr.shape[0]))
                    z_val, z_conf = sample_depth(depth_arr, None, depth_uv[0], depth_uv[1])
                    if z_val > 0.0:
                        hand_z = z_val
                        obs.depth_confidence = float(z_conf)

                if self.filter_mode == "ema":
                    try:
                        from integrations.colleague_hand.filters import EMAFilter
                        ema = self._ema_filters.setdefault(hid, EMAFilter(alpha=0.4))
                        smooth_arr = ema.update_dynamic(np.asarray(uv), 0.4)
                        smooth_uv = (float(smooth_arr[0]), float(smooth_arr[1]))
                    except Exception:
                        smooth_uv = uv
                    smooth_z = hand_z
                else:
                    if hid not in self._filters:
                        self._filters[hid] = (_OneEuro(), _OneEuro(), _OneEuro())
                    fx, fy, fz = self._filters[hid]
                    smooth_uv = (fx(uv[0], timestamp), fy(uv[1], timestamp))
                    smooth_z = fz(hand_z, timestamp) if hand_z is not None else None

                reacquire_remaining = self._reacquire_remaining.get(hid, 0)
                if old is not None and reacquire_remaining > 0:
                    alpha = (0.45, 0.62, 0.80)[3 - reacquire_remaining]
                    smooth_uv = (
                        old.palm_uv[0] + alpha * (smooth_uv[0] - old.palm_uv[0]),
                        old.palm_uv[1] + alpha * (smooth_uv[1] - old.palm_uv[1]),
                    )
                    if smooth_z is not None and old.depth_z is not None:
                        smooth_z = old.depth_z + alpha * (smooth_z - old.depth_z)
                    reacquire_remaining -= 1
                    if reacquire_remaining:
                        self._reacquire_remaining[hid] = reacquire_remaining
                    else:
                        self._reacquire_remaining.pop(hid, None)

                velocity = 0.0
                if old is not None:
                    velocity = float(
                        np.linalg.norm(np.subtract(smooth_uv, old.palm_uv))
                        / max(timestamp - getattr(self, "_last_timestamp", timestamp), 1e-3)
                    )

                obs.palm_uv = smooth_uv
                obs.depth_z = smooth_z
                obs.velocity_px_s = velocity
                obs.timestamp = timestamp
                obs.stale = not should_detect
                filtered.append(obs)

            self._last = filtered
            self._coast = 0
        elif self._last and self._coast < self.max_coast_frames and self._previous_gray is not None:
            previous_hands = tuple(self._last)
            gray = cv2.cvtColor(tracker_frame, cv2.COLOR_RGB2GRAY)
            prior_gray = self._previous_gray
            coasted: list[TrackedHand] = []
            for obs in previous_hands:
                point = np.asarray([[obs.palm_uv[0] / scale_x, obs.palm_uv[1] / scale_y]], dtype=np.float32)
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(prior_gray, gray, point, None, winSize=(31, 31), maxLevel=3)
                if status is not None and bool(status[0, 0]):
                    tracked = np.asarray(nxt).reshape(-1, 2)[0]
                    u, v = float(tracked[0] * scale_x), float(tracked[1] * scale_y)
                    if 0 <= u < width and 0 <= v < height:
                        delta = float(np.hypot(u - obs.palm_uv[0], v - obs.palm_uv[1]))
                        self.lk_updates += 1
                        self.lk_motion_px += delta
                        coasted.append(TrackedHand(
                            hand_id=obs.hand_id,
                            landmarks_uv=None if obs.landmarks_uv is None else obs.landmarks_uv.copy(),
                            palm_uv=(u, v), confidence=obs.confidence * 0.85,
                            depth_z=obs.depth_z, timestamp=timestamp,
                            velocity_px_s=obs.velocity_px_s, handedness=obs.handedness,
                            palm_width_px=obs.palm_width_px, stale=True,
                            depth_confidence=obs.depth_confidence,
                        ))
            observations = coasted
            if coasted:
                self._last = coasted
                self._coast += 1
            else:
                observations = self._hold_dropped(previous_hands, timestamp)
                self._last = observations
                self._coast = 0
        else:
            observations = self._hold_dropped(tuple(self._last), timestamp)
            self._last = observations
            self._coast = 0

        filtering_ms = (time.perf_counter() - filtering_started) * 1000.0
        self._previous_gray = cv2.cvtColor(tracker_frame, cv2.COLOR_RGB2GRAY)
        self._last_timestamp = timestamp
        elapsed = (time.perf_counter() - started) * 1000.0
        dropout_age_ms = max(
            (max(0.0, timestamp - self._last_detected_at.get(hand.hand_id, timestamp)) * 1000.0 for hand in observations if hand.stale),
            default=0.0,
        )
        return GestureState(
            timestamp,
            frame_id,
            tuple(observations),
            self.backend_name,
            elapsed,
            any(o.stale for o in observations),
            tracking.luminance,
            tracking.clahe_active,
            tracking.gamma,
            tracking.elapsed_ms,
            detection_ms,
            filtering_ms,
            dropout_age_ms,
            self._detection_ms_total / max(self._detection_samples, 1),
        )

    def _hold_dropped(self, previous: tuple[TrackedHand, ...], timestamp: float) -> list[TrackedHand]:
        held: list[TrackedHand] = []
        grace = self.dropout_grace_seconds
        for old in previous:
            seen_at = self._last_detected_at.get(old.hand_id, old.timestamp)
            age = max(0.0, timestamp - seen_at)
            if grace <= 0.0 or age >= grace:
                self._last_detected_at.pop(old.hand_id, None)
                self._filters.pop(old.hand_id, None)
                self._ema_filters.pop(old.hand_id, None)
                self._reacquire_pending.discard(old.hand_id)
                self._reacquire_remaining.pop(old.hand_id, None)
                continue
            fade = 1.0 - 0.45 * age / grace
            held.append(TrackedHand(
                hand_id=old.hand_id,
                landmarks_uv=None if old.landmarks_uv is None else old.landmarks_uv.copy(),
                palm_uv=old.palm_uv,
                confidence=float(max(0.15, old.confidence * fade)),
                depth_z=old.depth_z,
                timestamp=timestamp,
                velocity_px_s=0.0,
                handedness=old.handedness,
                palm_width_px=old.palm_width_px,
                stale=True,
                depth_confidence=old.depth_confidence,
            ))
            self._reacquire_pending.add(old.hand_id)
        return held

    def reset(self) -> None:
        self._last = []
        self._filters.clear()
        self._ema_filters.clear()
        self._previous_gray = None
        self._coast = 0
        self._last_timestamp = None
        self._last_detected_at.clear()
        self._reacquire_pending.clear()
        self._reacquire_remaining.clear()
        self._last_detection_ms = 0.0
        self._detection_ms_total = 0.0
        self._detection_samples = 0
        self.last_detection_ran = False
        self.lk_updates = 0
        self.lk_motion_px = 0.0

    def close(self) -> None:
        self.backend.close()
