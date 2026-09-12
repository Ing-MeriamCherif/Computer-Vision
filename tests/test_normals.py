import pytest
import numpy as np

from geometry import CameraModel, DepthState, backproject_depth, geometry_from_depth_state
from geometry.debug import exact_plane_depth, fronto_parallel_plane, sphere_depth, sphere_normals, step_depth
from geometry.normals import (
    NormalConfig,
    NormalMode,
    NormalResult,
    _select_multiscale,
    angular_metrics,
    estimate_normals,
    normals_to_rgb,
)


def _camera(width: int = 96, height: int = 72) -> CameraModel:
    return CameraModel(width, height, 90.0, 92.0, (width - 1) / 2, (height - 1) / 2)


def test_frontoparallel_normals_face_camera_and_are_unit() -> None:
    cam = _camera()
    depth = fronto_parallel_plane(cam)
    points, valid = backproject_depth(depth, cam)
    result = estimate_normals(points, valid, depth, NormalMode.BASELINE)
    assert result.normal_valid_mask.all()
    np.testing.assert_allclose(result.normals[..., 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(result.normals[..., 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(result.normals[..., 2], -1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(result.normals, axis=-1), 1.0, atol=1e-6)
    assert np.all(np.sum(result.normals * points, axis=-1) <= 1e-6)


def test_tilted_plane_has_low_angular_error() -> None:
    cam = _camera()
    depth = exact_plane_depth(cam, normal=(0.15, -0.25, 0.95), distance=-2.5)
    points, valid = backproject_depth(depth, cam)
    predicted = estimate_normals(points, valid, depth, NormalMode.EDGE_AWARE)
    expected = np.array([0.15, -0.25, 0.95], dtype=np.float64)
    expected /= np.linalg.norm(expected)
    expected = np.broadcast_to(expected, points.shape).copy()
    if np.sum(expected[0, 0] * points[0, 0]) > 0:
        expected = -expected
    metrics = angular_metrics(predicted.normals, expected, predicted.normal_valid_mask)
    assert metrics.mean_degrees < 0.1
    assert metrics.p95_degrees < 0.25


def test_sphere_normals_are_reasonably_accurate() -> None:
    cam = _camera()
    depth = sphere_depth(cam)
    points, valid = backproject_depth(depth, cam)
    predicted = estimate_normals(points, valid, depth, NormalMode.MULTI_SCALE)
    metrics = angular_metrics(predicted.normals, sphere_normals(points), predicted.normal_valid_mask)
    assert metrics.valid_percentage > 35.0
    assert metrics.mean_degrees < 2.0


def test_edge_aware_step_does_not_join_foreground_and_background() -> None:
    cam = CameraModel(20, 10, 20, 20, 9.5, 4.5)
    depth = step_depth(cam)
    points, valid = backproject_depth(depth, cam)
    baseline = estimate_normals(points, valid, depth, NormalMode.BASELINE)
    edge_aware = estimate_normals(points, valid, depth, NormalMode.EDGE_AWARE)
    boundary = [(4, 9), (4, 10)]
    for y, x in boundary:
        assert edge_aware.normal_valid_mask[y, x]
        assert edge_aware.normals[y, x, 2] < -0.99
        assert edge_aware.confidence[y, x] < edge_aware.confidence[4, 4]
    assert baseline.normals[4, 9, 2] > -0.2


def test_invalid_hole_and_border_are_explicit() -> None:
    cam = _camera(12, 10)
    depth = fronto_parallel_plane(cam)
    depth[4, 4] = np.nan
    points, valid = backproject_depth(depth, cam)
    result = estimate_normals(points, valid, depth, NormalMode.EDGE_AWARE)
    assert not result.normal_valid_mask[4, 4]
    assert not np.isfinite(result.normals[4, 4]).any()
    assert np.all(result.confidence[~result.normal_valid_mask] == 0)
    assert result.normal_valid_mask[0, 0]


def test_multiscale_selects_smallest_reliable_radius() -> None:
    cam = _camera(32, 24)
    depth = fronto_parallel_plane(cam)
    depth[12, 15] = np.nan
    points, valid = backproject_depth(depth, cam)
    result = estimate_normals(points, valid, depth, NormalMode.MULTI_SCALE)
    assert result.selected_radius[2, 2] == 1
    assert result.selected_radius[12, 15] == 0
    assert result.selected_radius[12, 14] in (1, 2, 4)


def test_confidence_and_normal_visualization() -> None:
    cam = _camera(16, 12)
    depth = fronto_parallel_plane(cam)
    points, valid = backproject_depth(depth, cam)
    result = estimate_normals(points, valid, depth)
    assert np.all((result.confidence >= 0) & (result.confidence <= 1))
    rgb = normals_to_rgb(result.normals, result.normal_valid_mask)
    assert rgb.dtype == np.uint8 and rgb.shape == (12, 16, 3)
    assert tuple(rgb[0, 0]) == (127, 127, 0)


def test_geometry_state_populates_phase2_fields() -> None:
    cam = _camera(16, 12)
    depth = fronto_parallel_plane(cam)
    state = geometry_from_depth_state(DepthState(depth, 1.0, 7, "relative"), cam)
    assert state.normal_valid_mask is not None
    assert state.normal_confidence is not None
    assert state.selected_radius is not None
    assert state.normals is not None and state.normals.shape == (12, 16, 3)


def test_select_multiscale_synthetic_cases() -> None:
    """Verify multiscale candidate selection logic: Cases A, B, C, D and tie-breaking."""
    shape = (1, 5)
    # Candidate 1 (radius 1)
    c1 = NormalResult(
        normals=np.array([[[1, 0, 0], [1, 0, 0], [1, 0, 0], [np.nan, np.nan, np.nan], [1, 0, 0]]], dtype=np.float32),
        normal_valid_mask=np.array([[True, True, True, False, True]]),
        confidence=np.array([[0.8, 0.4, 0.3, 0.0, 0.4]], dtype=np.float32),
        discontinuity_strength=np.zeros(shape, dtype=np.float32),
        selected_radius=np.full(shape, 1, dtype=np.int16),
        neighbor_support=np.ones(shape, dtype=np.float32),
    )

    # Candidate 2 (radius 2)
    c2 = NormalResult(
        normals=np.array([[[0, 1, 0], [0, 1, 0], [0, 1, 0], [np.nan, np.nan, np.nan], [0, 1, 0]]], dtype=np.float32),
        normal_valid_mask=np.array([[True, True, True, False, True]]),
        confidence=np.array([[0.9, 0.7, 0.5, 0.0, 0.4]], dtype=np.float32),
        discontinuity_strength=np.zeros(shape, dtype=np.float32),
        selected_radius=np.full(shape, 2, dtype=np.int16),
        neighbor_support=np.ones(shape, dtype=np.float32),
    )

    result = _select_multiscale([c1, c2], acceptance_threshold=0.6)

    # Pixel 0: Case A - radius 1 accepted (0.8 >= 0.6) -> selects r=1 even though r=2 has higher conf
    assert result.selected_radius[0, 0] == 1
    assert result.normal_valid_mask[0, 0]
    assert np.isclose(result.confidence[0, 0], 0.8)
    np.testing.assert_allclose(result.normals[0, 0], [1, 0, 0])

    # Pixel 1: Case B - radius 1 unaccepted (0.4 < 0.6), radius 2 accepted (0.7 >= 0.6) -> selects r=2
    assert result.selected_radius[0, 1] == 2
    assert result.normal_valid_mask[0, 1]
    assert np.isclose(result.confidence[0, 1], 0.7)
    np.testing.assert_allclose(result.normals[0, 1], [0, 1, 0])

    # Pixel 2: Case C - neither accepted (0.3 and 0.5 < 0.6), both valid -> selects best valid (r=2, conf 0.5)
    assert result.selected_radius[0, 2] == 2
    assert result.normal_valid_mask[0, 2]
    assert np.isclose(result.confidence[0, 2], 0.5)
    np.testing.assert_allclose(result.normals[0, 2], [0, 1, 0])

    # Pixel 3: Case D - no radius valid -> selects r=0, valid=False, conf=0.0, normals=NaN
    assert result.selected_radius[0, 3] == 0
    assert not result.normal_valid_mask[0, 3]
    assert result.confidence[0, 3] == 0.0
    assert np.isnan(result.normals[0, 3]).all()

    # Pixel 4: Tie-breaking - neither accepted (0.4 and 0.4 < 0.6), both valid -> tie broken to smaller radius (r=1)
    assert result.selected_radius[0, 4] == 1
    assert result.normal_valid_mask[0, 4]
    assert np.isclose(result.confidence[0, 4], 0.4)
    np.testing.assert_allclose(result.normals[0, 4], [1, 0, 0])


def test_select_multiscale_empty_raises_value_error() -> None:
    with pytest.raises(ValueError, match="candidates list cannot be empty"):
        _select_multiscale([], acceptance_threshold=0.6)


@pytest.mark.parametrize("scale", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
def test_scale_invariance_tilted_plane(scale: float) -> None:
    """Uniform scaling D -> k*D preserves normals, valid mask, radius and confidence."""
    cam = _camera(64, 48)
    base_depth = exact_plane_depth(cam, normal=(0.2, -0.3, 0.9), distance=-2.0)

    # Reference at scale = 1.0
    ref_points, ref_valid = backproject_depth(base_depth, cam)
    ref_res = estimate_normals(ref_points, ref_valid, base_depth, NormalMode.MULTI_SCALE)

    # Scaled
    scaled_depth = base_depth * scale
    scaled_points, scaled_valid = backproject_depth(scaled_depth, cam)
    scaled_res = estimate_normals(scaled_points, scaled_valid, scaled_depth, NormalMode.MULTI_SCALE)

    np.testing.assert_array_equal(scaled_res.normal_valid_mask, ref_res.normal_valid_mask)
    np.testing.assert_array_equal(scaled_res.selected_radius, ref_res.selected_radius)
    np.testing.assert_allclose(scaled_res.confidence, ref_res.confidence, atol=1e-5)

    valid = ref_res.normal_valid_mask
    np.testing.assert_allclose(scaled_res.normals[valid], ref_res.normals[valid], atol=2e-5)


@pytest.mark.parametrize("scale", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
def test_scale_invariance_sphere(scale: float) -> None:
    """Sphere normals and confidence are scale invariant under D -> k*D."""
    cam = _camera(64, 48)
    base_depth = sphere_depth(cam)

    ref_points, ref_valid = backproject_depth(base_depth, cam)
    ref_res = estimate_normals(ref_points, ref_valid, base_depth, NormalMode.MULTI_SCALE)

    scaled_depth = base_depth * scale
    scaled_points, scaled_valid = backproject_depth(scaled_depth, cam)
    scaled_res = estimate_normals(scaled_points, scaled_valid, scaled_depth, NormalMode.MULTI_SCALE)

    np.testing.assert_array_equal(scaled_res.normal_valid_mask, ref_res.normal_valid_mask)
    np.testing.assert_array_equal(scaled_res.selected_radius, ref_res.selected_radius)
    np.testing.assert_allclose(scaled_res.confidence, ref_res.confidence, atol=1e-5)

    valid = ref_res.normal_valid_mask
    np.testing.assert_allclose(scaled_res.normals[valid], ref_res.normals[valid], atol=1e-5)


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
def test_scale_invariance_step_discontinuity(scale: float) -> None:
    """Step edge discontinuity detection and normal estimation are scale invariant."""
    cam = CameraModel(20, 10, 20, 20, 9.5, 4.5)
    base_depth = step_depth(cam)

    ref_points, ref_valid = backproject_depth(base_depth, cam)
    ref_res = estimate_normals(ref_points, ref_valid, base_depth, NormalMode.EDGE_AWARE)

    scaled_depth = base_depth * scale
    scaled_points, scaled_valid = backproject_depth(scaled_depth, cam)
    scaled_res = estimate_normals(scaled_points, scaled_valid, scaled_depth, NormalMode.EDGE_AWARE)

    np.testing.assert_array_equal(scaled_res.normal_valid_mask, ref_res.normal_valid_mask)
    np.testing.assert_allclose(scaled_res.discontinuity_strength, ref_res.discontinuity_strength, atol=1e-5)
    np.testing.assert_allclose(scaled_res.confidence, ref_res.confidence, atol=1e-5)
    np.testing.assert_allclose(scaled_res.normals, ref_res.normals, atol=1e-5)


def test_input_confidence_modulation_and_safety() -> None:
    """Verify input_confidence modulation, NaN safety, out-of-bounds clamping, and shape checking."""
    cam = _camera(16, 12)
    depth = fronto_parallel_plane(cam)
    depth[0, 0] = np.nan
    points, valid = backproject_depth(depth, cam)

    # Incompatible shape raises ValueError
    with pytest.raises(ValueError, match="input_confidence must have shape"):
        estimate_normals(points, valid, depth, input_confidence=np.ones((5, 5)))

    # Construct input confidence with normal, NaN, out-of-bounds values
    input_conf = np.ones((12, 16), dtype=np.float32)
    input_conf[2, 2] = np.nan
    input_conf[3, 3] = 2.0   # > 1.0
    input_conf[4, 4] = -1.0  # < 0.0
    input_conf[5, 5] = 0.5

    result = estimate_normals(points, valid, depth, NormalMode.MULTI_SCALE, input_confidence=input_conf)

    # Invalid normal pixel at (0, 0) must have 0 confidence even with input 1.0
    assert not result.normal_valid_mask[0, 0]
    assert result.confidence[0, 0] == 0.0

    # NaN in input confidence becomes 0.0
    assert result.confidence[2, 2] == 0.0

    # Out of bounds clamped to [0, 1]
    assert 0.0 <= result.confidence[3, 3] <= 1.0
    assert result.confidence[4, 4] == 0.0

    # Valid modulation: 0.5 * raw confidence
    ref_res = estimate_normals(points, valid, depth, NormalMode.MULTI_SCALE)
    assert np.isclose(result.confidence[5, 5], ref_res.confidence[5, 5] * 0.5, atol=1e-6)

    # All confidence values strictly in [0, 1] and no NaNs
    assert np.all((result.confidence >= 0.0) & (result.confidence <= 1.0))
    assert not np.isnan(result.confidence).any()
