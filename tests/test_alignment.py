import numpy as np

from geometry.alignment import align_history_depth, align_inverse_depth


def test_inverse_depth_alignment_recovers_affine_scale_and_shift() -> None:
    history = np.linspace(1.0, 4.0, 400, dtype=np.float64).reshape(20, 20)
    q_current = 1.15 / history - 0.08
    current = 1.0 / q_current
    result = align_inverse_depth(current, history, np.ones_like(history, dtype=bool), min_samples=64)
    assert result.fit_success
    assert abs(result.scale - 1.15) < 1e-6
    assert abs(result.shift + 0.08) < 1e-6
    aligned, valid = align_history_depth(history, result, "relative")
    np.testing.assert_allclose(aligned, current, atol=1e-5)
    assert valid.all()


def test_alignment_rejects_outliers_robustly() -> None:
    history = np.linspace(1.0, 4.0, 400, dtype=np.float64).reshape(20, 20)
    q_current = 1.15 / history - 0.08
    current = 1.0 / q_current
    current.flat[::5] = 0.6
    result = align_inverse_depth(current, history, np.ones_like(history, dtype=bool), min_samples=64)
    assert result.fit_success
    assert abs(result.scale - 1.15) < 0.03
    assert abs(result.shift + 0.08) < 0.03
    assert result.sample_count < 400


def test_alignment_fails_safely_with_insufficient_samples() -> None:
    depth = np.ones((4, 4), dtype=np.float32)
    result = align_inverse_depth(depth, depth, np.ones_like(depth, dtype=bool), min_samples=32)
    assert not result.fit_success
    assert result.sample_count == 16
    assert np.isinf(result.fit_residual)
