import numpy as np
import pytest

from geometry.motion import MotionState, flow_consistency


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


def test_motion_state_validates_shape_and_frame_contract() -> None:
    flow = np.zeros((5, 7, 2), dtype=np.float32)
    state = MotionState(1, 2, 0.1, flow, flow)
    assert state.valid_mask is not None and state.valid_mask.all()
    assert state.flow_confidence is not None
    with pytest.raises(ValueError, match="backward_flow"):
        MotionState(1, 2, 0.1, flow, np.zeros((5, 6, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="valid_mask"):
        MotionState(1, 2, 0.1, flow, flow, valid_mask=np.ones((5, 6), dtype=bool))
