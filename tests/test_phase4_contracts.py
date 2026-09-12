import numpy as np
import pytest

from geometry import CameraModel, DepthState, TemporalGeometryEngine, validate_renderer_geometry
from geometry.normals import NormalConfig, NormalMode


def _camera() -> CameraModel:
    return CameraModel(20, 12, 24.0, 26.0, 8.5, 5.0)


def test_renderer_handoff_validator_accepts_consistent_state() -> None:
    cam = _camera()
    depth = np.full((cam.height, cam.width), 2.0, dtype=np.float32)
    state = TemporalGeometryEngine(cam).update(None, cam, 0, 0.0, DepthState(depth, 0.0, 0, "relative"))
    report = validate_renderer_geometry(state, normal_config=NormalConfig(), max_history_age=8)
    assert report.valid, report.errors


def test_renderer_handoff_validator_reports_invalid_geometry() -> None:
    cam = _camera()
    depth = np.full((cam.height, cam.width), 2.0, dtype=np.float32)
    state = TemporalGeometryEngine(cam).update(None, cam, 0, 0.0, DepthState(depth, 0.0, 0, "relative"))
    state.positions_3d[2, 2, 2] = 99.0
    report = validate_renderer_geometry(state)
    assert not report.valid
    assert any("P_z" in error for error in report.errors)


def test_temporal_engine_rejects_stale_depth_identity_and_timestamp() -> None:
    cam = _camera()
    depth = np.full((cam.height, cam.width), 2.0, dtype=np.float32)
    engine = TemporalGeometryEngine(cam)
    with pytest.raises(ValueError, match="source_frame_id"):
        engine.update(None, cam, 10, 1.0, DepthState(depth, 1.0, 8, "relative"))
    with pytest.raises(ValueError, match="stale"):
        engine.update(None, cam, 10, 1.0, DepthState(depth, 0.0, 10, "relative"))


@pytest.mark.parametrize("period", [1, 2, 3, 4])
def test_temporal_multi_rate_schedule_keeps_contract(period: int) -> None:
    cam = _camera()
    depth = np.full((cam.height, cam.width), 2.0, dtype=np.float32)
    engine = TemporalGeometryEngine(cam)
    for frame_id in range(12):
        fresh = frame_id % period == 0
        state = engine.update(
            None,
            cam,
            frame_id,
            frame_id / 30.0,
            DepthState(depth, frame_id / 30.0, frame_id, "relative") if fresh else None,
            None if frame_id == 0 else _motion(cam, frame_id - 1, frame_id),
        )
        assert np.all((state.confidence >= 0) & (state.confidence <= 1))
        assert np.all(np.isfinite(state.positions_3d[state.valid_mask]))


def _motion(cam: CameraModel, source: int, target: int):
    from geometry import MotionState

    flow = np.zeros((cam.height, cam.width, 2), dtype=np.float32)
    return MotionState(source, target, target / 30.0, flow, flow)
