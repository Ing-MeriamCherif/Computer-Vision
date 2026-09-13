from __future__ import annotations

import time

import cv2
import numpy as np


def test_latest_frame_slot_distinguishes_overwrites_and_consumption():
    from geometry.camera_worker import LatestFrameSlot

    slot = LatestFrameSlot()
    frame = np.zeros((4, 4, 3), np.uint8)
    slot.put(frame, 10, time.monotonic())
    slot.put(frame, 11, time.monotonic())
    assert slot.dropped_count == 1
    assert slot.get(timeout=0.1)[1] == 11
    assert slot.total_arrived == 2


def test_geometry_preserves_depth_source_identity():
    from geometry import CameraModel, DepthState
    from geometry.normals import geometry_from_depth_state

    cam = CameraModel(8, 6, 7, 7, 3.5, 2.5)
    depth = np.ones((6, 8), np.float32)
    state = DepthState(depth, 12.5, 42, "relative", valid_mask=np.ones_like(depth, bool))
    geometry = geometry_from_depth_state(state, cam)
    assert geometry.source_frame_id == 42
    assert geometry.timestamp == 12.5


def test_detector_skip_uses_lk_motion_update():
    from geometry.hand_control import HandControlEngine, TrackedHand

    class FakeBackend:
        name = "test-real-backend"

        def process(self, rgb):
            return [TrackedHand(0, None, (20.0, 20.0), 1.0)]

        def close(self):
            pass

    engine = HandControlEngine(backend="missing", input_size=(64, 48), detect_every_n=2, max_coast_frames=4)
    engine.backend = FakeBackend()
    first = np.zeros((48, 64, 3), np.uint8)
    cv2.circle(first, (20, 20), 5, (255, 255, 255), -1)
    second = np.zeros_like(first)
    cv2.circle(second, (27, 20), 5, (255, 255, 255), -1)
    a = engine.update(first, 1.0, 0)
    x0 = a.hands[0].palm_uv[0]
    b = engine.update(second, 1.033, 1)
    assert b.hands and b.hands[0].stale
    assert b.hands[0].palm_uv[0] != x0
    engine.close()


def test_hand_depth_resizes_explicitly_between_domains():
    from geometry.hand_control import HandControlEngine, TrackedHand

    class FakeBackend:
        name = "test-real-backend"

        def process(self, rgb):
            return [TrackedHand(0, None, (4.0, 4.0), 1.0)]

        def close(self):
            pass

    engine = HandControlEngine(backend="missing", input_size=(8, 8), detect_every_n=1)
    engine.backend = FakeBackend()
    state = engine.update(np.zeros((8, 8, 3), np.uint8), 1.0, 0, depth_map=np.ones((4, 4), np.float32))
    assert state.hands and state.hands[0].depth_z is not None
    assert state.hands[0].depth_confidence > 0
    engine.close()


def test_physical_gate_has_no_synthetic_flag(capsys):
    from tools.physical_camera_gate import main
    import sys

    old = sys.argv
    try:
        sys.argv = ["physical_camera_gate", "--help"]
        try:
            main()
        except SystemExit as exc:
            assert exc.code == 0
        assert "--synthetic" not in capsys.readouterr().out
    finally:
        sys.argv = old


def test_temporal_processing_ids_survive_skipped_capture_ids():
    from geometry import CameraModel, DepthState
    from geometry.temporal import TemporalGeometryEngine

    cam = CameraModel(8, 6, 7, 7, 3.5, 2.5)
    depth = np.ones((6, 8), np.float32)
    engine = TemporalGeometryEngine(cam)
    first = engine.update(np.zeros((6, 8, 3), np.uint8), cam, 100, 1.0, DepthState(depth, 1.0, 100, "relative"), processing_frame_id=0)
    second = engine.update(np.zeros((6, 8, 3), np.uint8), cam, 104, 1.033, DepthState(depth, 1.033, 104, "relative"), processing_frame_id=1)
    assert first.processing_frame_id == 0
    assert second.processing_frame_id == 1
    assert engine.last_diagnostics is None or engine.last_diagnostics.reset_reason != "frame_discontinuity"


def test_depth_uv_mapping_and_warm_near_palette():
    from geometry.depth_sampling import camera_uv_to_depth_uv
    from geometry.visualization import depth_to_rgb

    assert camera_uv_to_depth_uv((0.0, 0.0), (640, 480), (256, 192)) == (0.0, 0.0)
    assert camera_uv_to_depth_uv((320.0, 240.0), (640, 480), (256, 192)) == (128.0, 96.0)
    image = depth_to_rgb(np.asarray([[1.0, 3.0]], np.float32))
    assert tuple(image[0, 0])[:2] > tuple(image[0, 1])[:2]


def test_depth_gate_rejects_reversed_forward_z():
    from tools.physical_camera_gate import _depth_results
    from geometry import DepthState

    records = [(DepthState(np.ones((2, 2), np.float32), 1.0, 0, "relative"), 1.0)] * 4
    result = _depth_results(records, [1.0, 2.0], [1.0], [2.0], [3.0, 3.0, 2.0, 2.0])
    assert result["convention_physically_verified"] is False
    assert result["result"] == "FAIL"


def test_hand_control_has_no_p4_depth_sampling_dependency():
    from pathlib import Path

    source = Path("geometry/hand_control.py").read_text(encoding="utf-8")
    assert "from .lighting import sample_depth" not in source
