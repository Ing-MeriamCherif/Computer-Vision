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


def test_source_discontinuity_resets_geometry_history_before_new_depth():
    from geometry.p123_live_runtime import _reset_history_on_source_discontinuity

    class FakeBackend:
        def __init__(self):
            self.reset_calls = 0

        def reset_history(self):
            self.reset_calls += 1

    backend = FakeBackend()
    assert not _reset_history_on_source_discontinuity(backend, 50, 51)
    assert backend.reset_calls == 0
    assert _reset_history_on_source_discontinuity(backend, 51, 3)
    assert backend.reset_calls == 1


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


def test_fast_temporal_confidence_is_native_and_scale_invariant():
    from geometry.p123_live_runtime import _fast_temporal_confidence

    previous = np.full((2, 3), 2.0, np.float32)
    current = np.full((2, 3), 4.0, np.float32)
    confidence = _fast_temporal_confidence(previous, current, None, None)
    assert confidence.shape == previous.shape
    assert np.all(confidence > 0.99)

    changed = current.copy()
    changed[0, 0] = 8.0
    confidence = _fast_temporal_confidence(previous, changed, None, None)
    assert confidence[0, 0] < confidence[1, 1]


def test_latest_buffers_wait_for_versions_without_polling():
    from geometry.async_pipeline import LatestFrameBuffer, LatestDepthBuffer
    from geometry import DepthState

    frames = LatestFrameBuffer()
    packet, version = frames.wait_for_new(timeout=0.0)
    assert packet is None and version == 0
    frames.put(np.zeros((2, 2, 3), np.uint8), 7, 1.0)
    packet, version = frames.wait_for_new(version, timeout=0.01)
    assert packet is not None and packet.frame_id == 7 and version == 1
    depths = LatestDepthBuffer()
    depths.put(DepthState(np.ones((2, 2), np.float32), 1.0, 7, "relative"))
    state, version = depths.wait_for_new(timeout=0.01)
    assert state is not None and state.source_frame_id == 7 and version == 1


def test_depth_size_parser_accepts_rectangular_inputs():
    from tools.p123_live_app import _parse_depth_size

    assert _parse_depth_size("336x448", (480, 640)) == (336, 448)
    assert _parse_depth_size("native", (480, 640)) == (480, 640)


def test_p123_live_cli_defaults_to_local_production(monkeypatch):
    from tools import p123_live_app

    monkeypatch.setattr("sys.argv", ["p123_live_app"])
    args = p123_live_app.parse_args()
    assert args.depth_backend == "local"
    assert args.depth_size == "336x448"


def test_explicit_calibration_validates_dimensions_before_mirror_adjustment():
    import pytest
    from geometry.camera import CameraModel
    from geometry.p123_live_runtime import _camera_for_capture

    calibration = CameraModel(640, 480, 500.0, 501.0, 280.0, 240.0)
    mirrored = _camera_for_capture(calibration, 640, 480, mirror=True)
    assert mirrored.cx == 359.0
    assert mirrored.cy == calibration.cy
    with pytest.raises(RuntimeError, match="mismatches negotiated camera"):
        _camera_for_capture(calibration, 1280, 720, mirror=True)


def test_p123_views_are_separate_display_modules():
    import importlib

    for name in ("rgb", "depth", "normals", "temporal", "hands", "xyz"):
        module = importlib.import_module(f"p123.views.{name}")
        assert callable(module.render)


def test_cuda_normals_include_radius_four_and_unit_vectors():
    import pytest
    pytest.importorskip("torch")
    from geometry import CameraModel
    from geometry.cuda_backend import TorchGeometryBackend

    camera = CameraModel(24, 16, 20, 20, 12, 8)
    backend = TorchGeometryBackend("cpu")
    geometry = backend.process_depth(np.ones((16, 24), np.float32), camera)
    assert 4 in np.unique(geometry.selected_radius)
    vectors = geometry.normals[geometry.normal_valid_mask]
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)


def test_talel_xyz_depth_proxy_stays_in_metric_working_volume():
    from geometry import CameraModel
    from geometry.p123_live_runtime import _talel_depth_from_palm_size, _talel_hand_xyz

    z, estimated = _talel_depth_from_palm_size(525.0, 45.0)
    assert estimated is True
    assert 0.20 <= z <= 3.0
    fallback, estimated = _talel_depth_from_palm_size(525.0, None)
    assert (fallback, estimated) == (0.50, False)
    xyz, estimated = _talel_hand_xyz(CameraModel(640, 480, 525, 525, 320, 240), (320, 240), 45)
    assert estimated is True and np.allclose(xyz[:2], 0.0, atol=1e-6) and 0.9 < xyz[2] < 1.1


def test_xyz_worker_tracks_hands_without_waiting_for_depth_geometry():
    import threading
    from collections import deque
    from geometry import CameraModel
    from geometry.hand_control import GestureState, TrackedHand
    from geometry.p123_live_runtime import P123LiveRuntime

    runtime = P123LiveRuntime.__new__(P123LiveRuntime)
    runtime._running = True
    runtime._state_lock = threading.Lock()
    runtime.max_state_age_ms = 200.0
    runtime._hands = GestureState(
        timestamp=time.monotonic(), source_frame_id=7,
        hands=(TrackedHand(0, None, (320.0, 240.0), 0.95, palm_width_px=45.0),),
        backend="test", tracker_ms=1.0,
    )
    runtime.camera = CameraModel(640, 480, 525, 525, 320, 240)
    runtime._xyz = ()
    runtime._xyz_smooth = {}
    runtime._hand_times = deque(maxlen=8)
    runtime._xyz_times = deque(maxlen=8)
    runtime._xyz_ages = deque(maxlen=8)
    runtime._xyz_completion_ages = deque(maxlen=8)
    worker = threading.Thread(target=runtime._xyz_loop)
    worker.start()
    time.sleep(0.05)
    runtime._running = False
    worker.join(timeout=1.0)

    assert runtime._xyz and runtime._xyz[0].xyz_camera is not None
    assert runtime._xyz[0].source_frame_id == 7


def test_xyz_worker_never_reuses_a_previous_position_when_palm_size_is_invalid():
    import threading
    from collections import deque
    from geometry import CameraModel
    from geometry.hand_control import GestureState, TrackedHand
    from geometry.p123_live_runtime import P123LiveRuntime

    runtime = P123LiveRuntime.__new__(P123LiveRuntime)
    runtime._running = True
    runtime._state_lock = threading.Lock()
    runtime.max_state_age_ms = 200.0
    runtime._hands = GestureState(
        timestamp=time.monotonic(), source_frame_id=8,
        hands=(TrackedHand(0, None, (320.0, 240.0), 0.95, palm_width_px=None),),
        backend="test", tracker_ms=1.0,
    )
    runtime.camera = CameraModel(640, 480, 525, 525, 320, 240)
    runtime._xyz = ()
    runtime._xyz_smooth = {0: np.asarray((0.0, 0.0, 1.0), dtype=np.float32)}
    runtime._hand_times = deque(maxlen=8)
    runtime._xyz_times = deque(maxlen=8)
    runtime._xyz_ages = deque(maxlen=8)
    runtime._xyz_completion_ages = deque(maxlen=8)
    worker = threading.Thread(target=runtime._xyz_loop)
    worker.start()
    time.sleep(0.05)
    runtime._running = False
    worker.join(timeout=1.0)

    assert runtime._xyz and runtime._xyz[0].xyz_camera is None
    assert runtime._xyz_smooth == {}
