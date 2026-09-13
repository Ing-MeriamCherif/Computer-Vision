from __future__ import annotations

import numpy as np

from geometry.hand_control import HandControlEngine, TrackedHand, choose_tracker_size, palm_center


def _landmarks(center=(100.0, 80.0), span=40.0):
    pts = np.zeros((21, 2), dtype=np.float64)
    pts[:] = center
    cx, cy = center
    pts[5] = (cx - span / 2.0, cy)
    pts[9] = (cx - span / 6.0, cy)
    pts[13] = (cx + span / 6.0, cy)
    pts[17] = (cx + span / 2.0, cy)
    return pts


def test_tracker_resolution_preserves_common_source_aspects():
    assert choose_tracker_size((1920, 1080)) == (640, 360)
    assert choose_tracker_size((1280, 720)) == (640, 360)
    assert choose_tracker_size((640, 480)) == (640, 480)


def test_palm_center_median_rejects_one_noisy_mcp():
    points = _landmarks()
    expected = palm_center(points)
    points[9] = (900.0, -500.0)
    actual = palm_center(points)
    assert expected is not None and actual is not None
    assert np.linalg.norm(np.subtract(actual, expected)) < 8.0


def test_string_frame_ids_do_not_disable_detector_cadence():
    class Backend:
        name = "test"

        def __init__(self):
            self.calls = 0

        def process(self, rgb, timestamp=None):
            self.calls += 1
            pts = _landmarks()
            return [TrackedHand(0, pts, (0.0, 0.0), 1.0)]

        def close(self):
            pass

    engine = HandControlEngine(backend="missing", input_size=(64, 48), detect_every_n=2)
    backend = Backend()
    engine.backend = backend
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    engine.update(frame, 1.0, "camA-1")
    engine.update(frame, 1.033, "camA-2")
    engine.update(frame, 1.066, "camA-3")
    assert backend.calls == 2
    engine.close()


def test_projection_uses_filtered_palm_and_fresh_width():
    class Backend:
        name = "test"

        def __init__(self):
            self.calls = 0

        def process(self, rgb, timestamp=None):
            self.calls += 1
            pts = _landmarks((80.0 + self.calls * 3.0, 60.0), 30.0 + self.calls * 5.0)
            return [TrackedHand(0, pts, (0.0, 0.0), 1.0)]

        def close(self):
            pass

    engine = HandControlEngine(backend="missing", input_size=(160, 120), detect_every_n=1)
    backend = Backend()
    engine.backend = backend
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    first = engine.update(frame, 1.0, "a")
    second = engine.update(frame, 1.033, "b")
    assert first.hands[0].palm_width_px is not None
    assert second.hands[0].palm_width_px is not None
    assert second.hands[0].palm_width_px > first.hands[0].palm_width_px
    engine.close()


def test_lk_dropout_recomputes_palm_width(monkeypatch):
    class Backend:
        name = "test"

        def __init__(self):
            self.calls = 0

        def process(self, rgb, timestamp=None):
            self.calls += 1
            return [TrackedHand(0, _landmarks(center=(30.0, 24.0), span=30.0), (0.0, 0.0), 1.0)] if self.calls == 1 else []

        def close(self):
            pass

    engine = HandControlEngine(backend="missing", input_size=(64, 48), detect_every_n=1, max_coast_frames=2)
    engine.backend = Backend()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    first = engine.update(frame, 1.0, "a")
    old = first.hands[0].landmarks_uv
    assert old is not None
    expanded = old.copy()
    expanded[5, 0] -= 5.0
    expanded[17, 0] += 5.0

    def fake_lk(_prev, _cur, points, *_args, **_kwargs):
        return expanded.astype(np.float32), np.ones((len(points), 1), np.uint8), None

    monkeypatch.setattr("cv2.calcOpticalFlowPyrLK", fake_lk)
    coast = engine.update(frame, 1.033, "b")
    assert coast.hands and coast.hands[0].stale
    assert coast.hands[0].palm_width_px is not None
    assert coast.hands[0].palm_width_px > first.hands[0].palm_width_px
    engine.close()


def test_talel_xyz_projects_back_to_filtered_palm():
    from geometry.camera import CameraModel
    from geometry.p123_live_runtime import _talel_hand_xyz

    camera = CameraModel(640, 360, 560.0, 560.0, 319.5, 179.5)
    uv = (410.25, 142.75)
    xyz, valid = _talel_hand_xyz(camera, uv, 48.0)
    assert valid
    assert np.allclose(camera.project(xyz), uv, atol=1e-6)


def test_palm_scale_z_is_forward_monotonic_and_half_scale_corrected():
    from geometry.p123_live_runtime import _talel_depth_from_palm_size

    z30, _ = _talel_depth_from_palm_size(525.0, 30.0)
    z40, _ = _talel_depth_from_palm_size(525.0, 40.0)
    z50, _ = _talel_depth_from_palm_size(525.0, 50.0)
    assert z30 > z40 > z50
    assert np.isclose(z40, 525.0 * 0.0425 / 40.0)
