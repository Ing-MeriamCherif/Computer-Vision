"""Person 1 hand tracker — MediaPipe local inference, swappable backend.

Priority: MediaPipe Tasks vision API (maintained).
Fallbacks (auto): legacy mp.solutions.hands -> MockTracker (so pipeline
still runs for FPS/vector testing when mediapipe is not installed).

Output: pixel-space landmarks for palm back-projection. Only direction
+ intensity are needed downstream, so keep this lightweight.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class HandResult:
    found: bool
    palm_uv: tuple[float, float] | None      # pixel coords of PALM_IDX
    landmarks_uv: np.ndarray | None          # (21,2) pixel coords
    confidence: float
    backend: str
    palm_px: float | None = None             # palm width (5-17) in px -> depth proxy


class HandTrackerBase:
    backend_name = "base"

    def process(self, rgb: np.ndarray) -> HandResult:
        raise NotImplementedError

    def close(self) -> None:
        pass


def palm_width_px(landmarks_uv: np.ndarray | None) -> float | None:
    """Palm width = dist(landmark 5, landmark 17). None if unavailable."""
    try:
        if landmarks_uv is None or len(landmarks_uv) < 18:
            return None
        return float(np.linalg.norm(landmarks_uv[5] - landmarks_uv[17]))
    except Exception:
        return None


class MockTracker(HandTrackerBase):
    """Deterministic stub: circle in center. Lets vector/FPS run w/o model."""
    backend_name = "mock"

    def __init__(self) -> None:
        self.t0 = time.time()

    def process(self, rgb: np.ndarray) -> HandResult:
        h, w = rgb.shape[:2]
        t = time.time() - self.t0
        cx = w / 2 + 0.2 * w * np.sin(t * 1.2)
        cy = h / 2 + 0.15 * h * np.cos(t * 0.9)
        lm = np.zeros((21, 2), dtype=np.float64)
        lm[:, 0], lm[:, 1] = cx, cy
        lm[config.PALM_IDX] = [cx, cy]
        lm[5] = [cx - 30, cy]
        lm[17] = [cx + 30, cy]
        return HandResult(True, (float(cx), float(cy)), lm, 0.0, self.backend_name,
                          palm_width_px(lm))


class LegacyMPTracker(HandTrackerBase):
    """mp.solutions.hands (deprecated but widely available)."""
    backend_name = "mediapipe-legacy"

    def __init__(self, max_hands: int = 1) -> None:
        import mediapipe as mp
        self._mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=config.MIN_DETECTION_CONF,
            min_tracking_confidence=config.MIN_TRACKING_CONF,
        )

    def process(self, rgb: np.ndarray) -> HandResult:
        h, w = rgb.shape[:2]
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return HandResult(False, None, None, 0.0, self.backend_name)
        lm = res.multi_hand_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float64)
        palm = (float(pts[config.PALM_IDX, 0]), float(pts[config.PALM_IDX, 1]))
        conf = 1.0
        if res.multi_handedness:
            try:
                conf = float(res.multi_handedness[0].classification[0].score)
            except Exception:
                pass
        return HandResult(True, palm, pts, conf, self.backend_name, palm_width_px(pts))

    def close(self) -> None:
        try:
            self.hands.close()
        except Exception:
            pass


def _try_tasks_tracker(max_hands: int) -> HandTrackerBase | None:
    """MediaPipe Tasks HandLandmarker (needs .task bundle file)."""
    import os
    bundle = os.getenv("MP_HAND_BUNDLE", "hand_landmarker.task")
    if not os.path.exists(bundle):
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        base = mp_python.BaseOptions(model_asset_path=bundle)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base,
            num_hands=max_hands,
            min_hand_detection_confidence=config.MIN_DETECTION_CONF,
            min_hand_presence_confidence=config.MIN_TRACKING_CONF,
            min_tracking_confidence=config.MIN_TRACKING_CONF,
        )
        landmarker = mp_vision.HandLandmarker.create_from_options(opts)

        class TasksTracker(HandTrackerBase):
            backend_name = "mediapipe-tasks"

            def process(self, rgb: np.ndarray) -> HandResult:
                h, w = rgb.shape[:2]
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = landmarker.detect(mp_img)
                if not res.hand_landmarks:
                    return HandResult(False, None, None, 0.0, self.backend_name)
                pts = np.array(
                    [[p.x * w, p.y * h] for p in res.hand_landmarks[0]], dtype=np.float64
                )
                palm = (float(pts[config.PALM_IDX, 0]), float(pts[config.PALM_IDX, 1]))
                conf = 1.0
                try:
                    conf = float(res.handedness[0][0].score)
                except Exception:
                    pass
                return HandResult(True, palm, pts, conf, self.backend_name,
                                  palm_width_px(pts))

            def close(self) -> None:
                try:
                    landmarker.close()
                except Exception:
                    pass

        return TasksTracker()
    except Exception:
        return None


def create_tracker(max_hands: int | None = None) -> HandTrackerBase:
    """Auto-select best available backend. Never crashes: falls back to mock."""
    mh = max_hands or config.MAX_HANDS
    forced = __import__("os").getenv("HAND_BACKEND", "auto").lower()
    if forced == "mock":
        return MockTracker()
    if forced in ("legacy", "mediapipe-legacy"):
        try:
            return LegacyMPTracker(mh)
        except Exception:
            return MockTracker()
    # auto: tasks bundle -> legacy -> mock
    t = _try_tasks_tracker(mh)
    if t is not None:
        return t
    try:
        return LegacyMPTracker(mh)
    except Exception as e:
        print(f"[hand_tracker] mediapipe unavailable ({e}); using MockTracker.")
        return MockTracker()
