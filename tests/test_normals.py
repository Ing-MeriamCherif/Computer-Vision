import numpy as np

from geometry import CameraModel, DepthState, backproject_depth, geometry_from_depth_state
from geometry.debug import exact_plane_depth, fronto_parallel_plane, sphere_depth, sphere_normals, step_depth
from geometry.normals import NormalMode, angular_metrics, estimate_normals, normals_to_rgb


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
