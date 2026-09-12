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
    assert state.disocclusion_mask is not None
    assert state.disocclusion_mask[:, 7:10].all()
    assert state.history_rejection_mask is not None
    assert state.history_rejection_mask[:, 7:10].all()
    assert state.occlusion_mask is not None
    assert not state.occlusion_mask[:, 7:10].any()
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


def test_subpixel_moving_step_edge_history_only() -> None:
    """Target 17: Subpixel moving step edge with depth_state=None on second frame."""
    cam = _camera(16, 8)
    fg, bg = 1.0, 3.0
    depth0 = np.full((cam.height, cam.width), bg, dtype=np.float32)
    depth0[:, :8] = fg

    engine = TemporalGeometryEngine(cam)
    engine.update(None, cam, 0, 0.0, DepthState(depth0, 0.0, 0, "relative"))

    # Backward flow shifts sampling by subpixel amount (e.g. -0.5, moving history left)
    motion = np.zeros((cam.height, cam.width, 2), dtype=np.float32)
    motion[..., 0] = -0.5
    fwd = np.zeros_like(motion)
    fwd[..., 0] = 0.5
    mstate = MotionState(0, 1, 1 / 30, fwd, motion)

    # Second frame has NO fresh depth (history-only propagation)
    state = engine.update(None, cam, 1, 1 / 30, depth_state=None, motion_state=mstate)

    # Valid region should not produce any phantom intermediate surfaces
    assert state.valid_mask[:, 1:-1].all()
    valid_depths = state.depth[state.valid_mask]
    intermediate_min = fg + 0.15 * (bg - fg)
    intermediate_max = bg - 0.15 * (bg - fg)
    phantom = (valid_depths > intermediate_min) & (valid_depths < intermediate_max)
    assert not phantom.any(), f"Found intermediate phantom depths in history-only warp: {valid_depths[phantom]}"

    # All valid pixels must be close to fg or bg
    close_fg = np.isclose(valid_depths, fg, atol=1e-3)
    close_bg = np.isclose(valid_depths, bg, atol=1e-3)
    assert (close_fg | close_bg).all()

    # Temporal age should be 1 for propagated history
    assert (state.temporal_age[state.valid_mask] == 1).all()


def test_100_consecutive_frames_with_fresh_depth_never_expires() -> None:
    """Target 19: 100 consecutive frames with fresh valid depth -> verify age is 0 at all 100 frames."""
    cam = _camera(16, 12)
    depth = np.full((cam.height, cam.width), 2.5, dtype=np.float32)
    engine = TemporalGeometryEngine(cam, TemporalConfig(max_history_age=8))

    for frame_id in range(100):
        motion = None if frame_id == 0 else _motion(frame_id - 1, frame_id, cam.width, cam.height)
        state = engine.update(
            None,
            cam,
            frame_id,
            frame_id / 30.0,
            depth_state=DepthState(depth, frame_id / 30.0, frame_id, "relative"),
            motion_state=motion,
        )
        assert state.valid_mask.all(), f"Frame {frame_id} lost validity"
        assert state.temporal_age is not None
        assert (state.temporal_age == 0).all(), f"Frame {frame_id} has non-zero age: max {state.temporal_age.max()}"


def test_distinct_rejection_and_mask_semantics() -> None:
    """Target 23: Separate mask semantics for bad flow, photometric, occlusion, disocclusion."""
    cam = _camera(20, 10)
    base_depth = np.full((cam.height, cam.width), 2.0, dtype=np.float32)
    fwd = np.zeros((cam.height, cam.width, 2), dtype=np.float32)
    bwd = np.zeros((cam.height, cam.width, 2), dtype=np.float32)

    # 1. Bad flow only: history rejected, NOT occlusion
    engine = TemporalGeometryEngine(cam)
    engine.update(None, cam, 0, 0.0, DepthState(base_depth, 0.0, 0, "relative"))
    bad_flow_err = np.zeros((cam.height, cam.width), dtype=np.float32)
    bad_flow_err[:, 5:10] = 50.0
    bad_flow_conf = np.ones((cam.height, cam.width), dtype=np.float32)
    bad_flow_conf[:, 5:10] = 0.0
    motion_bad_flow = MotionState(0, 1, 1 / 30, fwd, bwd, forward_backward_error=bad_flow_err, flow_confidence=bad_flow_conf)
    state = engine.update(None, cam, 1, 1 / 30, DepthState(base_depth, 1 / 30, 1, "relative"), motion_bad_flow)

    assert state.history_rejection_mask[:, 5:10].all()
    assert not state.history_valid[:, 5:10].any()
    assert not state.occlusion_mask[:, 5:10].any()
    assert not state.disocclusion_mask[:, 5:10].any()

    # 2. Photometric failure only: history rejected, NOT occlusion
    rgb_prev = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
    rgb_cur = np.zeros_like(rgb_prev)
    rgb_cur[:, 5:10] = 255
    engine = TemporalGeometryEngine(cam)
    engine.update(rgb_prev, cam, 0, 0.0, DepthState(base_depth, 0.0, 0, "relative"))
    state = engine.update(rgb_cur, cam, 1, 1 / 30, DepthState(base_depth, 1 / 30, 1, "relative"), MotionState(0, 1, 1 / 30, fwd, bwd))

    assert state.history_rejection_mask[:, 5:10].all()
    assert not state.history_valid[:, 5:10].any()
    assert not state.occlusion_mask[:, 5:10].any()
    assert not state.disocclusion_mask[:, 5:10].any()

    # 3. Geometric occlusion: current surface significantly closer than history
    engine = TemporalGeometryEngine(cam)
    engine.update(None, cam, 0, 0.0, DepthState(np.full((cam.height, cam.width), 3.0, dtype=np.float32), 0.0, 0, "relative"))
    cur_occluded = np.full((cam.height, cam.width), 3.0, dtype=np.float32)
    cur_occluded[:, 5:10] = 1.0
    state = engine.update(None, cam, 1, 1 / 30, DepthState(cur_occluded, 1 / 30, 1, "relative"), MotionState(0, 1, 1 / 30, fwd, bwd))

    assert state.occlusion_mask[:, 5:10].all()
    assert not state.disocclusion_mask[:, 5:10].any()
    assert state.history_rejection_mask[:, 5:10].all()
    assert not state.history_valid[:, 5:10].any()

    # 4. Geometric disocclusion: current surface significantly farther than history
    engine = TemporalGeometryEngine(cam)
    engine.update(None, cam, 0, 0.0, DepthState(np.full((cam.height, cam.width), 1.0, dtype=np.float32), 0.0, 0, "relative"))
    cur_disoccluded = np.full((cam.height, cam.width), 1.0, dtype=np.float32)
    cur_disoccluded[:, 5:10] = 3.0
    state = engine.update(None, cam, 1, 1 / 30, DepthState(cur_disoccluded, 1 / 30, 1, "relative"), MotionState(0, 1, 1 / 30, fwd, bwd))

    assert state.disocclusion_mask[:, 5:10].all()
    assert not state.occlusion_mask[:, 5:10].any()
    assert state.history_rejection_mask[:, 5:10].all()
    assert not state.history_valid[:, 5:10].any()


def test_bad_flow_local_patch_uses_fresh_depth() -> None:
    """Target 27: Bad-flow local patch uses fresh depth with age=0; adjacent good flow uses fused depth."""
    cam = _camera(16, 16)
    depth0 = np.full((cam.height, cam.width), 2.0, dtype=np.float32)
    depth1 = np.full((cam.height, cam.width), 2.2, dtype=np.float32)

    engine = TemporalGeometryEngine(cam, TemporalConfig(history_weight_max=0.5))
    engine.update(None, cam, 0, 0.0, DepthState(depth0, 0.0, 0, "metric"))

    fwd = np.zeros((cam.height, cam.width, 2), dtype=np.float32)
    bwd = np.zeros((cam.height, cam.width, 2), dtype=np.float32)
    err = np.zeros((cam.height, cam.width), dtype=np.float32)
    conf = np.ones((cam.height, cam.width), dtype=np.float32)

    # Patch in center [4:12, 4:12] has catastrophic flow error
    err[4:12, 4:12] = 20.0
    conf[4:12, 4:12] = 0.0
    motion = MotionState(0, 1, 1 / 30, fwd, bwd, forward_backward_error=err, flow_confidence=conf)

    state = engine.update(None, cam, 1, 1 / 30, DepthState(depth1, 1 / 30, 1, "metric"), motion)

    # Center patch: history rejected, pure fresh depth, age=0
    assert not state.history_valid[4:12, 4:12].any()
    assert state.history_rejection_mask[4:12, 4:12].all()
    np.testing.assert_allclose(state.depth[4:12, 4:12], 2.2, atol=1e-5)
    assert (state.temporal_age[4:12, 4:12] == 0).all()

    # Border regions: history accepted, depth fused between 2.0 and 2.2
    assert state.history_valid[:3, :3].all()
    assert not state.history_rejection_mask[:3, :3].any()
    assert ((state.depth[:3, :3] > 2.0) & (state.depth[:3, :3] < 2.2)).all()


def test_moving_step_edge_follows_current_observation_without_lag() -> None:
    """Moving step edge follows current observation without smeared edge lag."""
    cam = _camera(20, 8)
    # Frame 0: edge at column 10 (0..9 fg=1.0, 10..19 bg=3.0)
    d0 = np.full((cam.height, cam.width), 3.0, dtype=np.float32)
    d0[:, :10] = 1.0
    # Frame 1: edge shifts to column 12 (0..11 fg=1.0, 12..19 bg=3.0)
    d1 = np.full((cam.height, cam.width), 3.0, dtype=np.float32)
    d1[:, :12] = 1.0

    engine = TemporalGeometryEngine(cam)
    engine.update(None, cam, 0, 0.0, DepthState(d0, 0.0, 0, "relative"))

    # Zero motion (e.g. flow failed to capture edge movement or camera is static)
    fwd = np.zeros((cam.height, cam.width, 2), dtype=np.float32)
    bwd = np.zeros((cam.height, cam.width, 2), dtype=np.float32)
    state = engine.update(None, cam, 1, 1 / 30, DepthState(d1, 1 / 30, 1, "relative"), MotionState(0, 1, 1 / 30, fwd, bwd))

    # At columns 10 and 11, history had 3.0, but current observation has 1.0 (occlusion).
    # Disagreement must reject history and use current depth 1.0 without lag or intermediate smearing!
    np.testing.assert_allclose(state.depth[:, 10:12], 1.0, atol=1e-4)
    assert state.occlusion_mask[:, 10:12].all()
    assert not state.history_valid[:, 10:12].any()

