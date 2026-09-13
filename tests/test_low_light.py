from __future__ import annotations

import cv2
import numpy as np

from geometry.hand_control import HandControlEngine, TrackedHand
from geometry.low_light import AdaptiveLowLightPreprocessor


def test_bright_tracking_frame_is_unchanged():
    preprocessor = AdaptiveLowLightPreprocessor(threshold=88.0)
    source = np.full((48, 64, 3), 120, dtype=np.uint8)
    result = preprocessor.process(source)

    assert not result.clahe_active
    assert result.gamma == 1.0
    np.testing.assert_array_equal(result.rgb, source)
    np.testing.assert_array_equal(source, 120)


def test_dark_tracking_frame_is_lifted_without_mutating_source():
    preprocessor = AdaptiveLowLightPreprocessor(threshold=88.0)
    source = np.full((48, 64, 3), 24, dtype=np.uint8)
    original = source.copy()
    result = preprocessor.process(source)

    assert result.clahe_active
    assert result.gamma > 1.0
    assert float(result.rgb.mean()) > float(source.mean())
    np.testing.assert_array_equal(source, original)


def test_gamma_transition_is_smooth_near_threshold():
    preprocessor = AdaptiveLowLightPreprocessor(threshold=88.0)
    just_dim = preprocessor.process(np.full((48, 64, 3), 84, dtype=np.uint8))
    normal = preprocessor.process(np.full((48, 64, 3), 88, dtype=np.uint8))

    assert just_dim.gamma < 1.03
    assert float(just_dim.rgb.mean()) - float(normal.rgb.mean()) < 8.0


def test_hand_engine_preprocesses_tracker_input_not_camera_frame(monkeypatch):
    import geometry.hand_control as hand_control

    class RecordingBackend:
        name = "recording"

        def __init__(self):
            self.frames = []

        def process(self, rgb):
            self.frames.append(rgb.copy())
            return []

        def close(self):
            pass

    backend = RecordingBackend()
    monkeypatch.setattr(hand_control, "create_hand_tracker", lambda **_kwargs: backend)
    engine = HandControlEngine(input_size=(64, 48), detect_every_n=1, backend="tasks")
    dark = np.full((96, 128, 3), 24, dtype=np.uint8)
    original = dark.copy()
    state = engine.update(dark, 1.0, 0)
    assert state.low_light_active
    assert state.frame_luminance < 88.0
    assert backend.frames[0].mean() > cv2.resize(original, (64, 48)).mean()
    np.testing.assert_array_equal(dark, original)

    bright = np.full((96, 128, 3), 120, dtype=np.uint8)
    state = engine.update(bright, 1.033, 1)
    assert not state.low_light_active
    np.testing.assert_array_equal(backend.frames[1], cv2.resize(bright, (64, 48)))
    engine.close()


def test_hand_engine_holds_dropout_reacquires_smoothly_and_expires(monkeypatch):
    import geometry.hand_control as hand_control

    class SequenceBackend:
        name = "sequence"

        def __init__(self):
            self.results = [
                [TrackedHand(0, np.tile([[24.0, 18.0]], (21, 1)), (24.0, 18.0), 0.9, depth_z=1.0)],
                [],
                [TrackedHand(0, np.tile([[48.0, 18.0]], (21, 1)), (48.0, 18.0), 0.9, depth_z=1.0)],
                [],
            ]

        def process(self, _rgb):
            return self.results.pop(0)

        def close(self):
            pass

    backend = SequenceBackend()
    monkeypatch.setattr(hand_control, "create_hand_tracker", lambda **_kwargs: backend)
    engine = HandControlEngine(
        input_size=(64, 48), detect_every_n=1, max_coast_frames=0,
        dropout_grace_ms=120.0, backend="tasks",
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    detected = engine.update(frame, 1.0, 0)
    held = engine.update(frame, 1.06, 1)
    reacquired = engine.update(frame, 1.13, 2)
    expired = engine.update(frame, 1.4, 3)

    assert detected.active and not detected.stale
    assert held.active and held.stale
    assert 59.0 <= held.dropout_age_ms <= 61.0
    assert reacquired.active and not reacquired.stale
    assert reacquired.hands[0].hand_id == 0
    assert 24.0 < reacquired.hands[0].palm_uv[0] < 48.0
    assert not expired.active
    engine.close()
