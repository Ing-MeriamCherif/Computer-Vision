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


PALM_MCP_INDICES = (5, 9, 13, 17)


def palm_center(points: np.ndarray | None) -> tuple[float, float] | None:
    """Return a robust center of the four palm MCP landmarks."""
    if points is None:
        return None
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) <= max(PALM_MCP_INDICES):
        return None
    mcp = points[list(PALM_MCP_INDICES), :2]
    if not np.isfinite(mcp).all():
        return None
    center = np.median(mcp, axis=0)
    # With four samples, an extreme outlier can still influence the average
    # of the two middle values. Remove only a clearly isolated MCP, retaining
    # the requested median behavior for normal hand geometry.
    distances = np.linalg.norm(mcp - center, axis=1)
    baseline = float(np.median(distances))
    if baseline > 1e-6:
        outlier = int(np.argmax(distances))
        if distances[outlier] > 3.0 * baseline:
            center = np.median(np.delete(mcp, outlier, axis=0), axis=0)
    return float(center[0]), float(center[1])


def choose_tracker_size(source_size: tuple[int, int], target_size: tuple[int, int] = (640, 360)) -> tuple[int, int]:
    """Choose a direct aspect-matched tracker resolution (width, height)."""
    sw, sh = max(1, int(source_size[0])), max(1, int(source_size[1]))
    tw = max(14, int(target_size[0]))
    # The width is the processing budget. Height follows the physical aspect
    # ratio (640x360 for 16:9, 640x480 for 4:3).
    scale = min(1.0, tw / sw)
    return max(14, int(round(sw * scale))), max(14, int(round(sh * scale)))


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
    dropout_age_ms: float = 0.0

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

    def process(self, rgb: np.ndarray, timestamp: float | None = None) -> list[TrackedHand]: ...

    def close(self) -> None: ...


def _palm_width(points: np.ndarray | None) -> float | None:
    if points is None or len(points) < 18:
        return None
    points = np.asarray(points, dtype=np.float64)
    if not np.isfinite(points[[5, 17], :2]).all():
        return None
    return float(np.linalg.norm(points[5, :2] - points[17, :2]))


class _NullBackend:
    name = "unavailable"

    def process(self, rgb: np.ndarray, timestamp: float | None = None) -> list[TrackedHand]:
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

    def process(self, rgb: np.ndarray, timestamp: float | None = None) -> list[TrackedHand]:
        result = self._tracker.process(rgb)
        if not result.found or result.palm_uv is None:
            return []
        points = None if result.landmarks_uv is None else np.asarray(result.landmarks_uv, dtype=np.float64)
        center = palm_center(points) or (float(result.palm_uv[0]), float(result.palm_uv[1]))
        return [TrackedHand(
            hand_id=0,
            palm_uv=center,
            landmarks_uv=points,
            confidence=float(result.confidence),
            handedness=getattr(result, "handedness", None),
            palm_width_px=_palm_width(points) if points is not None else result.palm_px,
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
            running_mode=mp_vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp = mp
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def process(self, rgb: np.ndarray, timestamp: float | None = None) -> list[TrackedHand]:
        h, w = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if timestamp is None:
            timestamp = time.monotonic()
        timestamp_ms = max(self._last_timestamp_ms + 1, int(round(float(timestamp) * 1000.0)))
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
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
            palm = palm_center(points)
            if palm is None:
                continue
            if confidence < 0.45:
                continue
            observations.append(TrackedHand(
                hand_id=index,
                landmarks_uv=points,
                palm_uv=palm,
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

    def reset(self) -> None:
        self._last_timestamp_ms = -1


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

    def process(self, rgb: np.ndarray, timestamp: float | None = None) -> list[TrackedHand]:
        h, w = rgb.shape[:2]
        result = self._hands.process(rgb)
        output: list[TrackedHand] = []
        for index, hand in enumerate(result.multi_hand_landmarks or []):
            pts = np.array([[p.x * w, p.y * h] for p in hand.landmark], dtype=np.float64)
            palm = palm_center(pts)
            if palm is None:
                continue
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
                palm_uv=palm,
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
        detect_every_n: int = 1,
        max_coast_frames: int = 6,
        input_size: tuple[int, int] | None = None,
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
        self._last_detected_at: dict[int, float] = {}
        self._last: list[TrackedHand] = []
        self._filters: dict[int, tuple[_OneEuro, _OneEuro, _OneEuro]] = {}
        self._ema_filters: dict[int, object] = {}
        self._previous_gray: np.ndarray | None = None
        self._coast = 0
        self._last_timestamp: float | None = None
        self._update_index = 0
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

    def _coast_with_lk(
        self, small: np.ndarray, width: int, height: int,
        scale_x: float, scale_y: float, timestamp: float,
    ) -> list[TrackedHand]:
        """Coast only a genuine detector dropout and update palm scale too."""
        import cv2
        if not self._last or self._previous_gray is None or self._coast >= self.max_coast_frames:
            return []
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        coasted: list[TrackedHand] = []
        for prior in self._last:
            if prior.landmarks_uv is not None and len(prior.landmarks_uv) >= 18:
                points = np.asarray(prior.landmarks_uv, dtype=np.float32)
                tracker_points = points / np.asarray([scale_x, scale_y], dtype=np.float32)
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                    self._previous_gray, gray, tracker_points, None,
                    winSize=(31, 31), maxLevel=3,
                )
                if nxt is None or status is None:
                    continue
                valid = np.asarray(status).reshape(-1).astype(bool)
                required = np.asarray(PALM_MCP_INDICES)
                if len(valid) <= int(required.max()) or not bool(valid[required].all()):
                    continue
                mapped = np.asarray(nxt).reshape(-1, 2) * np.asarray([scale_x, scale_y])
                if not np.isfinite(mapped).all():
                    continue
                center = palm_center(mapped)
                width_px = _palm_width(mapped)
                if center is None or width_px is None:
                    continue
                u, v = center
                if not (0 <= u < width and 0 <= v < height):
                    continue
                delta = float(np.linalg.norm(np.subtract(center, prior.palm_uv)))
                self.lk_updates += 1
                self.lk_motion_px += delta
                coasted.append(TrackedHand(
                    hand_id=prior.hand_id, landmarks_uv=mapped.astype(np.float64),
                    palm_uv=center, confidence=prior.confidence * 0.85,
                    depth_z=prior.depth_z, timestamp=timestamp,
                    velocity_px_s=prior.velocity_px_s, handedness=prior.handedness,
                    palm_width_px=width_px, stale=True,
                    depth_confidence=prior.depth_confidence,
                ))
            else:
                point = np.asarray([[prior.palm_uv[0] / scale_x, prior.palm_uv[1] / scale_y]], dtype=np.float32)
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                    self._previous_gray, gray, point, None,
                    winSize=(31, 31), maxLevel=3,
                )
                if nxt is None or status is None or not bool(status[0, 0]):
                    continue
                tracked = np.asarray(nxt).reshape(-1, 2)[0]
                center = (float(tracked[0] * scale_x), float(tracked[1] * scale_y))
                if not (0 <= center[0] < width and 0 <= center[1] < height):
                    continue
                delta = float(np.linalg.norm(np.subtract(center, prior.palm_uv)))
                self.lk_updates += 1
                self.lk_motion_px += delta
                coasted.append(TrackedHand(
                    hand_id=prior.hand_id, landmarks_uv=None, palm_uv=center,
                    confidence=prior.confidence * 0.85, depth_z=prior.depth_z,
                    timestamp=timestamp, velocity_px_s=prior.velocity_px_s,
                    handedness=prior.handedness, palm_width_px=prior.palm_width_px,
                    stale=True, depth_confidence=prior.depth_confidence,
                ))
        if coasted:
            self._coast += 1
        return coasted

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
        iw, ih = choose_tracker_size((width, height), self.input_size or (640, 360))
        scale_x, scale_y = width / iw, height / ih
        small = cv2.resize(source, (iw, ih), interpolation=cv2.INTER_AREA)
        tracking = self.preprocessor.process(small)
        small = tracking.rgb

        should_detect = (
            self._last_timestamp is None
            or self.detect_every_n == 1
            or self._update_index % self.detect_every_n == 0
        )
        self._update_index += 1
        self.last_detection_ran = bool(should_detect)
        observations: list[TrackedHand] = []

        if should_detect:
            try:
                raw_observations = self.backend.process(small, timestamp)
            except TypeError:
                # Preserve compatibility with simple test/colleague adapters.
                raw_observations = self.backend.process(small)
            for obs in raw_observations:
                if obs.landmarks_uv is not None:
                    obs.landmarks_uv = obs.landmarks_uv * np.array([scale_x, scale_y])
                    center = palm_center(obs.landmarks_uv)
                    if center is not None:
                        obs.palm_uv = center
                    obs.palm_width_px = _palm_width(obs.landmarks_uv)
                else:
                    obs.palm_uv = (obs.palm_uv[0] * scale_x, obs.palm_uv[1] * scale_y)
            observations = self._assign_stable_ids([obs for obs in raw_observations if obs.confidence >= 0.45])
            for obs in observations:
                self._last_detected_at[obs.hand_id] = timestamp
        if not observations and self._last and self._previous_gray is not None:
            observations = self._coast_with_lk(small, width, height, scale_x, scale_y, timestamp)

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
                    smooth_width = obs.palm_width_px
                    smooth_z = hand_z

                else:
                    if hid not in self._filters:
                        self._filters[hid] = (_OneEuro(), _OneEuro(), _OneEuro())
                    fx, fy, fw = self._filters[hid]
                    smooth_uv = (fx(uv[0], timestamp), fy(uv[1], timestamp))
                    width_value = obs.palm_width_px
                    previous_width = old.palm_width_px if old is not None else None
                    if width_value is not None and previous_width is not None:
                        ratio = width_value / max(previous_width, 1e-6)
                        if ratio < 0.35 or ratio > 1.65:
                            width_value = None
                            obs.confidence *= 0.7
                    smooth_width = fw(width_value, timestamp) if width_value is not None else None
                    smooth_z = hand_z

                if old is not None and old.stale and not obs.stale:
                    smooth_uv = ((old.palm_uv[0] + smooth_uv[0]) * 0.5, (old.palm_uv[1] + smooth_uv[1]) * 0.5)

                velocity = 0.0
                if old is not None:
                    velocity = float(
                        np.linalg.norm(np.subtract(smooth_uv, old.palm_uv))
                        / max(timestamp - getattr(self, "_last_timestamp", timestamp), 1e-3)
                    )

                obs.palm_uv = smooth_uv
                obs.palm_width_px = smooth_width
                obs.depth_z = smooth_z
                obs.velocity_px_s = velocity
                obs.timestamp = timestamp
                obs.stale = bool(obs.stale or not should_detect or self._coast > 0)
                filtered.append(obs)

            self._last = filtered
            self._coast = 0
        else:
            held = []
            for old in self._last:
                age = max(0.0, timestamp - self._last_detected_at.get(old.hand_id, old.timestamp))
                if age < self.dropout_grace_seconds:
                    old.stale = True
                    old.timestamp = timestamp
                    held.append(old)
            observations = held
            self._last = held
            self._coast = 0
            self._filters.clear()
            self._ema_filters.clear()

        self._previous_gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        self._last_timestamp = timestamp
        elapsed = (time.perf_counter() - started) * 1000.0
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
            max((max(0.0, timestamp-self._last_detected_at.get(o.hand_id,timestamp))*1000.0 for o in observations if o.stale), default=0.0),
        )

    def reset(self) -> None:
        self._last = []
        self._filters.clear()
        self._ema_filters.clear()
        self._previous_gray = None
        self._coast = 0
        self._last_timestamp = None
        self._update_index = 0
        self._last_detected_at.clear()
        self.last_detection_ran = False
        self.lk_updates = 0
        self.lk_motion_px = 0.0
        reset_backend = getattr(self.backend, "reset", None)
        if callable(reset_backend):
            reset_backend()

    def close(self) -> None:
        self.backend.close()
