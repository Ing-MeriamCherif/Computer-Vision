import numpy as np
import pytest

from geometry import CameraModel, DepthScaleMode, DepthState, GeometryState


def _make_sample_camera_and_depth(h: int = 10, w: int = 12):
    cam = CameraModel(w, h, float(w), float(h), (w - 1) / 2.0, (h - 1) / 2.0)
    depth = np.ones((h, w), dtype=np.float32) * 2.0
    return cam, depth


def test_depth_state_validation() -> None:
    depth = np.ones((10, 12), dtype=np.float32)
    state = DepthState(
        depth=depth,
        timestamp=1.0,
        source_frame_id=0,
        scale_mode="relative",
        valid_mask=np.ones((10, 12), dtype=int),
        confidence=np.ones((10, 12), dtype=np.float32),
    )
    assert state.scale_mode == DepthScaleMode.RELATIVE
    assert state.valid_mask.dtype == bool
    assert state.valid_mask.shape == (10, 12)

    # 1D or 3D depth rejected
    with pytest.raises(ValueError, match="must be a 2D array"):
        DepthState(depth.ravel(), 1.0, 0, "relative")

    # Mismatched valid_mask shape
    with pytest.raises(ValueError, match="valid_mask shape .* must match depth"):
        DepthState(depth, 1.0, 0, "relative", valid_mask=np.ones((10, 11)))

    # Mismatched confidence shape
    with pytest.raises(ValueError, match="confidence shape .* must match depth"):
        DepthState(depth, 1.0, 0, "relative", confidence=np.ones((9, 12)))


def test_geometry_state_valid_all_fields() -> None:
    cam, depth = _make_sample_camera_and_depth(10, 12)
    h, w = depth.shape
    positions = np.zeros((h, w, 3), dtype=np.float32)
    valid_mask = np.ones((h, w), dtype=int)  # integers converted to bool
    normals = np.zeros((h, w, 3), dtype=np.float32)
    confidence = np.ones((h, w), dtype=np.float32)
    temporal_age = np.zeros((h, w), dtype=np.float32)
    history_valid = np.ones((h, w), dtype=int)
    occlusion_mask = np.zeros((h, w), dtype=float)

    state = GeometryState(
        timestamp=100.0,
        source_frame_id="frame_001",
        depth=depth,
        positions_3d=positions,
        valid_mask=valid_mask,
        camera=cam,
        scale_mode="metric",
        normals=normals,
        confidence=confidence,
        temporal_age=temporal_age,
        history_valid=history_valid,
        occlusion_mask=occlusion_mask,
    )
    assert state.scale_mode == DepthScaleMode.METRIC
    assert state.valid_mask.dtype == bool
    assert state.history_valid.dtype == bool
    assert state.occlusion_mask.dtype == bool
    assert state.normals.shape == (10, 12, 3)
    assert state.confidence.shape == (10, 12)
    assert state.temporal_age.shape == (10, 12)


def test_geometry_state_rejects_shape_mismatches() -> None:
    cam, depth = _make_sample_camera_and_depth(10, 12)
    h, w = depth.shape
    valid_pos = np.zeros((h, w, 3), dtype=np.float32)
    valid_mask = np.ones((h, w), dtype=bool)

    # Depth not 2D
    with pytest.raises(ValueError, match="depth must be a 2D array"):
        GeometryState(1.0, 0, depth.ravel(), valid_pos, valid_mask, cam, "relative")

    # valid_mask shape mismatch
    with pytest.raises(ValueError, match="valid_mask shape .* must match depth"):
        GeometryState(1.0, 0, depth, valid_pos, np.ones((h + 1, w), dtype=bool), cam, "relative")

    # positions_3d wrong dimensions
    with pytest.raises(ValueError, match="positions_3d must have shape"):
        GeometryState(1.0, 0, depth, np.zeros((h, w, 2)), valid_mask, cam, "relative")
    with pytest.raises(ValueError, match="positions_3d must have shape"):
        GeometryState(1.0, 0, depth, np.zeros((h + 1, w, 3)), valid_mask, cam, "relative")

    # camera resolution mismatch
    wrong_cam = CameraModel(w + 1, h, float(w), float(h), 0, 0)
    with pytest.raises(ValueError, match="camera resolution .* does not match depth shape"):
        GeometryState(1.0, 0, depth, valid_pos, valid_mask, wrong_cam, "relative")

    # normals shape mismatch
    with pytest.raises(ValueError, match="normals must have shape"):
        GeometryState(1.0, 0, depth, valid_pos, valid_mask, cam, "relative", normals=np.zeros((h, w)))
    with pytest.raises(ValueError, match="normals must have shape"):
        GeometryState(1.0, 0, depth, valid_pos, valid_mask, cam, "relative", normals=np.zeros((h, w, 4)))

    # confidence shape mismatch
    with pytest.raises(ValueError, match="confidence must have shape"):
        GeometryState(1.0, 0, depth, valid_pos, valid_mask, cam, "relative", confidence=np.ones((h, w + 1)))

    # temporal_age shape mismatch
    with pytest.raises(ValueError, match="temporal_age must have shape"):
        GeometryState(1.0, 0, depth, valid_pos, valid_mask, cam, "relative", temporal_age=np.ones((h + 1, w)))

    # history_valid shape mismatch
    with pytest.raises(ValueError, match="history_valid must have shape"):
        GeometryState(1.0, 0, depth, valid_pos, valid_mask, cam, "relative", history_valid=np.ones((h, w + 1)))

    # occlusion_mask shape mismatch
    with pytest.raises(ValueError, match="occlusion_mask must have shape"):
        GeometryState(1.0, 0, depth, valid_pos, valid_mask, cam, "relative", occlusion_mask=np.ones((h + 1, w)))
