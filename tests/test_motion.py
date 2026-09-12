import numpy as np
import pytest

from geometry.motion import MotionState, OpenCVFlowProvider, flow_consistency


def test_identity_flow_has_zero_error_and_full_confidence() -> None:
    flow = np.zeros((5, 7, 2), dtype=np.float32)
    error, confidence, valid = flow_consistency(flow, flow)
    assert valid.all()
    np.testing.assert_allclose(error, 0)
    np.testing.assert_allclose(confidence, 1)


def test_inconsistent_flow_lowers_confidence() -> None:
    forward = np.zeros((5, 7, 2), dtype=np.float32)
    backward = np.zeros_like(forward)
    backward[..., 0] = 3.0
    error, confidence, valid = flow_consistency(forward, backward, sigma=1.0)
    assert valid.all()
    np.testing.assert_allclose(error, 3.0)
    assert np.all(confidence < 0.01)


def test_single_flow_confidence_and_monotonic_sigma() -> None:
    """Verify single source of truth for flow confidence and monotonic sigma behavior."""
    forward = np.zeros((4, 4, 2), dtype=np.float32)
    backward = np.zeros_like(forward)
    backward[..., 0] = 1.0  # error = 1.0 px

    # sigma = 1.0 => conf = exp(-(1/1)^2) = exp(-1) ~ 0.3679
    _, conf_1, _ = flow_consistency(forward, backward, sigma=1.0)
    np.testing.assert_allclose(conf_1, np.exp(-1.0), atol=1e-5)

    # sigma = 2.0 => conf = exp(-(1/2)^2) = exp(-0.25) ~ 0.7788
    _, conf_2, _ = flow_consistency(forward, backward, sigma=2.0)
    np.testing.assert_allclose(conf_2, np.exp(-0.25), atol=1e-5)

    assert np.all(conf_2 > conf_1)

    # Large error (5 px) -> confidence near 0
    backward[..., 0] = 5.0
    _, conf_large, _ = flow_consistency(forward, backward, sigma=1.0)
    assert np.all(conf_large < 1e-10)


def test_motion_state_validates_shape_and_frame_contract() -> None:
    flow = np.zeros((5, 7, 2), dtype=np.float32)
    state = MotionState(1, 2, 0.1, flow, flow)
    assert state.valid_mask is not None and state.valid_mask.all()
    assert state.flow_confidence is not None
    with pytest.raises(ValueError, match="backward_flow"):
        MotionState(1, 2, 0.1, flow, np.zeros((5, 6, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="valid_mask"):
        MotionState(1, 2, 0.1, flow, flow, valid_mask=np.ones((5, 6), dtype=bool))
    with pytest.raises(ValueError, match="fb_sigma must be positive"):
        MotionState(1, 2, 0.1, flow, flow, fb_sigma=0.0)


def test_motion_state_hardened_validation() -> None:
    """Ensure invalid flow pixels have zero confidence, NaNs are sanitized, and shapes are checked."""
    flow = np.zeros((6, 8, 2), dtype=np.float32)

    # Provided flow_confidence with out of bounds and NaNs
    conf = np.ones((6, 8), dtype=np.float32) * 1.5
    conf[0, 0] = np.nan
    conf[1, 1] = -0.5

    # valid_mask with invalid pixel at (2, 2)
    valid = np.ones((6, 8), dtype=bool)
    valid[2, 2] = False

    state = MotionState(1, 2, 0.1, flow, flow, valid_mask=valid, flow_confidence=conf)
    assert state.flow_confidence[0, 0] == 0.0
    assert state.flow_confidence[1, 1] == 0.0
    assert state.flow_confidence[2, 2] == 0.0
    assert state.flow_confidence[3, 3] == 1.0  # clamped from 1.5 to 1.0
    assert not np.isnan(state.flow_confidence).any()
    assert np.all((state.flow_confidence >= 0.0) & (state.flow_confidence <= 1.0))

    # Incompatible shapes for flow diagnostics raise ValueError
    with pytest.raises(ValueError, match="flow_confidence"):
        MotionState(1, 2, 0.1, flow, flow, flow_confidence=np.ones((4, 4)))
    with pytest.raises(ValueError, match="forward_backward_error"):
        MotionState(1, 2, 0.1, flow, flow, forward_backward_error=np.ones((4, 4)))
    with pytest.raises(ValueError, match="photometric_error"):
        MotionState(1, 2, 0.1, flow, flow, photometric_error=np.ones((4, 4)))
    with pytest.raises(ValueError, match="occlusion_mask"):
        MotionState(1, 2, 0.1, flow, flow, occlusion_mask=np.ones((4, 4), dtype=bool))

    # flow_consistency validates valid_mask shape
    with pytest.raises(ValueError, match="valid_mask must have shape"):
        flow_consistency(flow, flow, valid_mask=np.ones((4, 4), dtype=bool))


def _synthetic_texture(width: int = 64, height: int = 48, shift_x: float = 0.0) -> np.ndarray:
    xx, yy = np.meshgrid(np.arange(width) - shift_x, np.arange(height))
    pattern = 0.5 + 0.25 * np.sin(xx * 0.4) * np.cos(yy * 0.4) + 0.25 * np.sin((xx + yy) * 0.3)
    return np.clip(pattern, 0.0, 1.0).astype(np.float32)


def test_float_and_uint8_image_handling_in_opencv_flow() -> None:
    """Verify float [0, 1] images are not collapsed to 0/1 and match uint8 flow direction."""
    try:
        provider = OpenCVFlowProvider(method="farneback")
    except RuntimeError:
        pytest.skip("OpenCV not available")

    # Synthetic translated texture (dx = 2.0 px)
    prev_float = _synthetic_texture(64, 48, shift_x=0.0)
    curr_float = _synthetic_texture(64, 48, shift_x=2.0)

    prev_uint8 = np.clip(prev_float * 255.0, 0, 255).astype(np.uint8)
    curr_uint8 = np.clip(curr_float * 255.0, 0, 255).astype(np.uint8)

    motion_float = provider.compute(prev_float, curr_float, 0, 1, 0.1)
    motion_uint8 = provider.compute(prev_uint8, curr_uint8, 0, 1, 0.1)

    # Flow should be non-degenerate in interior
    mean_flow_float = motion_float.forward_flow[10:38, 10:54, 0].mean()
    mean_flow_uint8 = motion_uint8.forward_flow[10:38, 10:54, 0].mean()

    # Both should estimate motion in approximately positive x direction (~ 2 px)
    assert mean_flow_float > 0.5
    assert mean_flow_uint8 > 0.5
    assert np.isclose(mean_flow_float, mean_flow_uint8, atol=0.5)


@pytest.mark.parametrize("method", ["dis", "farneback"])
def test_opencv_flow_smoke(method: str) -> None:
    """Smoke test for DIS and Farneback providers."""
    try:
        provider = OpenCVFlowProvider(method=method, fb_sigma=1.8)
    except RuntimeError:
        pytest.skip("OpenCV not available")

    prev = _synthetic_texture(64, 48, shift_x=0.0)
    curr = _synthetic_texture(64, 48, shift_x=1.0)

    motion = provider.compute(prev, curr, 10, 11, 0.33)
    assert motion.forward_flow.shape == (48, 64, 2)
    assert motion.backward_flow.shape == (48, 64, 2)
    assert motion.valid_mask is not None
    assert motion.flow_confidence is not None
    assert motion.forward_backward_error is not None
    assert motion.source_frame_id == 10
    assert motion.target_frame_id == 11
    assert np.all((motion.flow_confidence >= 0.0) & (motion.flow_confidence <= 1.0))

    # Resolution mismatch rejected
    mismatched = _synthetic_texture(32, 24)
    with pytest.raises(ValueError, match="same resolution"):
        provider.compute(prev, mismatched, 10, 11, 0.33)
