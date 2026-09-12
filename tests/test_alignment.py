import numpy as np
import pytest

from geometry.alignment import align_history_depth, align_inverse_depth


def test_inverse_depth_alignment_recovers_affine_scale_and_shift() -> None:
    history = np.linspace(1.0, 4.0, 400, dtype=np.float64).reshape(20, 20)
    q_current = 1.15 / history - 0.08
    current = 1.0 / q_current
    result = align_inverse_depth(current, history, np.ones_like(history, dtype=bool), min_samples=64)
    assert result.fit_success
    assert result.model_used == "affine"
    assert abs(result.scale - 1.15) < 1e-5
    assert abs(result.shift + 0.08) < 1e-5
    assert result.normalized_fit_residual < 0.01
    aligned, valid = align_history_depth(history, result, "relative")
    np.testing.assert_allclose(aligned, current, atol=1e-4)
    assert valid.all()


def test_alignment_rejects_outliers_robustly() -> None:
    history = np.linspace(1.0, 4.0, 400, dtype=np.float64).reshape(20, 20)
    q_current = 1.15 / history - 0.08
    current = 1.0 / q_current
    current.flat[::5] = 0.6
    result = align_inverse_depth(current, history, np.ones_like(history, dtype=bool), min_samples=64)
    assert result.fit_success
    assert result.model_used == "affine"
    assert abs(result.scale - 1.15) < 0.03
    assert abs(result.shift + 0.08) < 0.03
    assert result.sample_count < 400
    assert result.inlier_count < 400
    assert result.input_sample_count == 400


def test_alignment_fails_safely_with_insufficient_samples() -> None:
    depth = np.ones((4, 4), dtype=np.float32)
    result = align_inverse_depth(depth, depth, np.ones_like(depth, dtype=bool), min_samples=32)
    assert not result.fit_success
    assert result.model_used == "none"
    assert result.sample_count == 16
    assert result.input_sample_count == 16
    assert np.isinf(result.fit_residual)
    assert np.isinf(result.normalized_fit_residual)


@pytest.mark.parametrize("scale_factor", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
def test_alignment_scale_invariance_clean(scale_factor: float) -> None:
    """Uniform scaling D -> k*D preserves recovered scale, normalized residual, and fit success."""
    base_history = np.linspace(1.0, 4.0, 400, dtype=np.float64).reshape(20, 20)
    base_q_current = 1.20 / base_history - 0.05
    base_current = 1.0 / base_q_current

    # Reference fit
    ref_res = align_inverse_depth(base_current, base_history, np.ones_like(base_history, dtype=bool), min_samples=64)
    assert ref_res.fit_success

    # Scaled fit: Z' = k * Z => q' = q / k
    scaled_history = base_history * scale_factor
    scaled_q_current = base_q_current / scale_factor
    scaled_current = 1.0 / scaled_q_current

    scaled_res = align_inverse_depth(scaled_current, scaled_history, np.ones_like(scaled_history, dtype=bool), min_samples=64)

    assert scaled_res.fit_success == ref_res.fit_success
    assert scaled_res.model_used == ref_res.model_used
    assert np.isclose(scaled_res.scale, ref_res.scale, rtol=1e-4)
    assert np.isclose(scaled_res.shift * scale_factor, ref_res.shift, rtol=1e-3, atol=1e-5)
    assert np.isclose(scaled_res.normalized_fit_residual, ref_res.normalized_fit_residual, atol=1e-4)

    # Reconstructed depth is geometrically equivalent under scaling
    aligned, valid = align_history_depth(scaled_history, scaled_res, "relative")
    np.testing.assert_allclose(aligned, scaled_current, rtol=1e-4)
    assert valid.all()


@pytest.mark.parametrize("scale_factor", [0.01, 1.0, 100.0])
def test_alignment_scale_invariance_noisy_and_outliers(scale_factor: float) -> None:
    """Scale-invariance holds in the presence of noise and gross outliers."""
    rng = np.random.default_rng(42)
    base_history = np.linspace(1.0, 4.0, 400, dtype=np.float64).reshape(20, 20)
    base_q_current = 1.15 / base_history - 0.06 + rng.normal(0, 0.005, (20, 20))
    base_current = 1.0 / base_q_current
    base_current.flat[::8] = 0.4  # 12.5% outliers

    ref_res = align_inverse_depth(base_current, base_history, np.ones_like(base_history, dtype=bool), min_samples=64)
    assert ref_res.fit_success

    scaled_history = base_history * scale_factor
    scaled_current = base_current * scale_factor

    scaled_res = align_inverse_depth(scaled_current, scaled_history, np.ones_like(scaled_history, dtype=bool), min_samples=64)
    assert scaled_res.fit_success == ref_res.fit_success
    assert scaled_res.model_used == ref_res.model_used
    assert np.isclose(scaled_res.scale, ref_res.scale, rtol=1e-3)
    assert np.isclose(scaled_res.normalized_fit_residual, ref_res.normalized_fit_residual, atol=2e-3)
    assert scaled_res.inlier_count == ref_res.inlier_count


def test_alignment_low_variation_scene_cases() -> None:
    """Validate degeneracy detection and fallback across Cases A, B, C, D, E."""
    # Case A: Textured / varied depth -> affine recovered
    h_varied = np.linspace(1.0, 5.0, 100, dtype=np.float64).reshape(10, 10)
    c_varied = 1.0 / (1.1 / h_varied - 0.04)
    res_a = align_inverse_depth(c_varied, h_varied, np.ones_like(h_varied, dtype=bool), min_samples=32)
    assert res_a.fit_success
    assert res_a.model_used == "affine"

    # Case B: Nearly constant-depth plane -> affine degeneracy detected -> scale_only fallback
    h_plane = np.full((10, 10), 2.5, dtype=np.float64)
    c_plane = np.full((10, 10), 2.5, dtype=np.float64)
    res_b = align_inverse_depth(c_plane, h_plane, np.ones_like(h_plane, dtype=bool), min_samples=32)
    assert res_b.fit_success
    assert res_b.model_used == "scale_only"
    assert np.isclose(res_b.scale, 1.0, atol=1e-5)
    assert res_b.shift == 0.0

    # Case C: Same plane with global depth-scale flicker (e.g. scale 1.25)
    c_flicker = h_plane / 1.25  # q_current = 1.25 * q_history
    res_c = align_inverse_depth(c_flicker, h_plane, np.ones_like(h_plane, dtype=bool), min_samples=32)
    assert res_c.fit_success
    assert res_c.model_used == "scale_only"
    assert np.isclose(res_c.scale, 1.25, atol=1e-4)
    assert res_c.shift == 0.0
    aligned, valid = align_history_depth(h_plane, res_c, "relative")
    np.testing.assert_allclose(aligned, c_flicker, atol=1e-5)

    # Case D: Completely unusable data -> safe failure
    h_bad = np.full((10, 10), 2.0, dtype=np.float64)
    rng = np.random.default_rng(99)
    c_bad = rng.uniform(0.1, 10.0, (10, 10))  # totally uncorrelated
    res_d = align_inverse_depth(c_bad, h_bad, np.ones_like(h_bad, dtype=bool), min_samples=32, residual_threshold=0.01)
    assert not res_d.fit_success
    assert res_d.model_used == "none"

    # Case E: Insufficient samples
    mask_small = np.zeros((10, 10), dtype=bool)
    mask_small[:2, :2] = True
    res_e = align_inverse_depth(c_varied, h_varied, mask_small, min_samples=32)
    assert not res_e.fit_success
    assert res_e.model_used == "none"
    assert res_e.sample_count == 4


def test_alignment_moving_object_outliers() -> None:
    """75% static background with known affine instability, 25% moving foreground."""
    history = np.linspace(1.5, 4.5, 400, dtype=np.float64).reshape(20, 20)
    q_current = 1.20 / history - 0.05
    current = 1.0 / q_current

    # 25% moving foreground object in lower-right quadrant
    # In history it was background depth, now it moves to depth 0.8
    current[10:, 10:] = 0.8

    result = align_inverse_depth(current, history, np.ones_like(history, dtype=bool), min_samples=64)
    assert result.fit_success
    assert result.model_used == "affine"
    assert np.isclose(result.scale, 1.20, rtol=0.03)
    assert np.isclose(result.shift, -0.05, atol=0.03)
    # The moving object pixels (100 pixels) must be excluded from inliers
    assert result.inlier_count <= 310
    assert result.input_sample_count == 400
