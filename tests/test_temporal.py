import numpy as np
import pytest

from geometry import CameraModel, DepthState, backproject_depth
from geometry.motion import MotionState
from geometry.normals import estimate_normals
from geometry.temporal import TemporalConfig, TemporalGeometryEngine, photometric_error


def _camera(width: int = 32, height: int = 24) -> CameraModel:
    return CameraModel(width, height, 35.0, 35.0, (width - 1) / 2, (height - 1) / 2)


def _motion(frame_a: int, frame_b: int, width: int, height: int, dx: float = 0.0) -> MotionState:
    forward = np.zeros((height, width, 2), dtype=np.float32)
    backward = np.zeros_like(forward)
    forward[..., 0] = dx
    backward[..., 0] = -dx
    return MotionState(frame_a, frame_b, float(frame_b), forward, backward)


def test_static_noisy_depth_is_stabilized() -> None:
    cam = _camera()
    base = np.tile(np.linspace(1.5, 2.5, cam.width, dtype=np.float32), (cam.height, 1))
    rng = np.random.default_rng(4)
    engine = TemporalGeometryEngine(cam, TemporalConfig(history_weight_max=0.9, history_decay=0.98))
    raws = []
    stabilized = []
    for frame_id in range(12):
        depth = base + rng.normal(0.0, 0.03, base.shape).astype(np.float32)
        raws.append(float(depth[10, 10]))
        state = engine.update(None, cam, frame_id, frame_id / 30.0, DepthState(depth, frame_id / 30.0, frame_id, "relative"), _motion(frame_id - 1, frame_id, cam.width, cam.height) if frame_id else None)
        stabilized.append(float(state.depth[10, 10]))
    assert np.std(stabilized[2:]) < np.std(raws[2:])


def test_stabilized_normals_have_lower_frame_to_frame_jitter() -> None:
    cam = _camera()
    base = np.full((cam.height, cam.width), 2.0, dtype=np.float32)
    rng = np.random.default_rng(9)
    engine = TemporalGeometryEngine(cam, TemporalConfig(history_weight_max=0.9, history_decay=0.98))
    raw_normals = []
    stable_normals = []
    flow = np.zeros((cam.height, cam.width, 2), dtype=np.float32)
    for frame_id in range(12):
        depth = base + rng.normal(0.0, 0.03, base.shape).astype(np.float32)
        positions, valid = backproject_depth(depth, cam)
        raw_normals.append(estimate_normals(positions, valid, depth).normals[10, 14])
        motion = None if frame_id == 0 else MotionState(frame_id - 1, frame_id, frame_id / 30.0, flow, flow)
        state = engine.update(None, cam, frame_id, frame_id / 30.0, DepthState(depth, frame_id / 30.0, frame_id, "relative"), motion)
        stable_normals.append(state.normals[10, 14])
    raw_jitter = np.linalg.norm(np.diff(np.asarray(raw_normals), axis=0), axis=1).mean()
    stable_jitter = np.linalg.norm(np.diff(np.asarray(stable_normals), axis=0), axis=1).mean()
    assert stable_jitter < raw_jitter


def test_translation_propagates_history_to_new_pixel_location() -> None:
    cam = _camera(12, 8)
    depth = np.tile(np.arange(cam.width, dtype=np.float32) * 0.02 + 2.0, (cam.height, 1))
    engine = TemporalGeometryEngine(cam)
    engine.update(None, cam, 0, 0.0, DepthState(depth, 0.0, 0, "relative"))
    state = engine.update(None, cam, 1, 1 / 30, None, _motion(0, 1, cam.width, cam.height, dx=1.0))
    assert not state.valid_mask[:, 0].any()
    np.testing.assert_allclose(state.depth[:, 1:], depth[:, :-1], atol=1e-5)
    assert state.temporal_age is not None and state.temporal_age[3, 5] == 1


def test_disoccluded_region_prefers_current_depth() -> None:
    cam = _camera(16, 8)
    previous_depth = np.full((cam.height, cam.width), 3.0, dtype=np.float32)
    previous_depth[:, :10] = 1.0
    current_depth = np.full_like(previous_depth, 3.0)
    current_depth[:, :6] = 1.0
    engine = TemporalGeometryEngine(cam)
    engine.update(None, cam, 0, 0.0, DepthState(previous_depth, 0.0, 0, "relative"))
    state = engine.update(None, cam, 1, 1 / 30, DepthState(current_depth, 1 / 30, 1, "relative"), _motion(0, 1, cam.width, cam.height))
    assert state.occlusion_mask is not None
    assert state.occlusion_mask[:, 7:10].all()
    np.testing.assert_allclose(state.depth[:, 7:10], 3.0, atol=1e-5)
    assert state.history_valid is not None and not state.history_valid[:, 7:10].any()


def test_missing_depth_decays_confidence_and_expires_history() -> None:
    cam = _camera()
    depth = np.full((cam.height, cam.width), 2.0, dtype=np.float32)
    config = TemporalConfig(history_decay=0.5, max_history_age=2)
    engine = TemporalGeometryEngine(cam, config)
    engine.update(None, cam, 0, 0.0, DepthState(depth, 0.0, 0, "relative"))
    first = engine.update(None, cam, 1, 0.1, None, _motion(0, 1, cam.width, cam.height))
    second = engine.update(None, cam, 2, 0.2, None, _motion(1, 2, cam.width, cam.height))
    third = engine.update(None, cam, 3, 0.3, None, _motion(2, 3, cam.width, cam.height))
    assert first.temporal_age is not None and first.temporal_age.max() == 1
    assert second.temporal_age is not None and second.temporal_age.max() == 2
    assert first.temporal_confidence is not None and second.temporal_confidence is not None
    assert second.temporal_confidence.mean() < first.temporal_confidence.mean()
    assert not third.valid_mask.any()


def test_camera_change_and_timestamp_gap_reset_history() -> None:
    cam = _camera()
    depth = np.full((cam.height, cam.width), 2.0, dtype=np.float32)
    engine = TemporalGeometryEngine(cam, TemporalConfig(timestamp_gap_reset=0.2))
    engine.update(None, cam, 0, 0.0, DepthState(depth, 0.0, 0, "relative"))
    changed = engine.update(None, cam, 5, 0.1, None, None)
    assert not changed.valid_mask.any()
    engine.update(None, cam, 6, 0.2, DepthState(depth, 0.2, 6, "relative"))
    gap = engine.update(None, cam, 7, 1.0, None, None)
    assert not gap.valid_mask.any()


def test_first_update_without_depth_fails_safely() -> None:
    cam = _camera()
    with pytest.raises(ValueError, match="first temporal update"):
        TemporalGeometryEngine(cam).update(None, cam, 0, 0.0, None)


def test_photometric_error_detects_local_change_but_ignores_global_offset() -> None:
    previous = np.tile(np.linspace(0, 255, 16, dtype=np.uint8), (8, 1))
    current = np.clip(previous.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    current[2:6, 6:10] = 255
    flow = np.zeros((8, 16, 2), dtype=np.float32)
    error, valid = photometric_error(previous, current, flow)
    assert valid.all()
    assert float(error[3:5, 7:9].mean()) > float(error[0:2, 0:4].mean()) + 0.2
