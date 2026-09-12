import numpy as np
import pytest

from geometry import CameraModel, DepthScaleMode, backproject_depth
from geometry.debug import fit_plane, tilted_plane


def test_invalid_depth_is_masked_and_nan() -> None:
    cam = CameraModel(3, 2, 2, 2, 1, 0.5)
    depth = np.array([[1.0, np.nan, 2.0], [0.0, -1.0, np.inf]])
    points, valid = backproject_depth(depth, cam, DepthScaleMode.RELATIVE)
    np.testing.assert_array_equal(valid, [[True, False, True], [False, False, False]])
    assert np.isnan(points[0, 1]).all()


def test_inverse_depth_requires_explicit_conversion() -> None:
    cam = CameraModel(3, 2, 2, 2, 1, 0.5)
    with pytest.raises(ValueError, match="explicit calibrated"):
        backproject_depth(np.ones((2, 3)), cam, DepthScaleMode.INVERSE)


def test_tilted_plane_reconstructs_with_low_residual() -> None:
    cam = CameraModel(64, 48, 60, 62, 31.5, 23.5)
    points, valid = backproject_depth(tilted_plane(cam), cam, DepthScaleMode.RELATIVE)
    fit = fit_plane(points, valid)
    assert fit.point_count == 64 * 48
    # Exact ray-plane intersection produces points lying on the plane to near floating-point precision
    assert fit.rms_residual < 1e-6
    assert fit.max_abs_residual < 1e-6


def test_depth_camera_dimension_mismatch_rejected() -> None:
    cam = CameraModel(64, 48, 60, 62, 31.5, 23.5)
    # Height mismatch
    with pytest.raises(ValueError, match="does not match camera resolution.*camera intrinsics must first be explicitly transformed"):
        backproject_depth(np.ones((50, 64), dtype=np.float32), cam)
    # Width mismatch
    with pytest.raises(ValueError, match="does not match camera resolution.*camera intrinsics must first be explicitly transformed"):
        backproject_depth(np.ones((48, 60), dtype=np.float32), cam)
    # Swapped dimensions (48, 64) vs camera height 48, width 64 -> (camera.height, camera.width) is (48, 64)
    with pytest.raises(ValueError, match="does not match camera resolution"):
        backproject_depth(np.ones((64, 48), dtype=np.float32), cam)


def test_exact_ray_plane_reconstruction() -> None:
    from geometry.debug import exact_plane_depth
    cam = CameraModel(64, 48, 60, 62, 31.5, 23.5)
    normal = np.array([0.15, -0.25, 0.95], dtype=np.float64)
    distance = -2.5
    depth = exact_plane_depth(cam, normal=normal, distance=distance)
    assert np.isfinite(depth).all()
    points, valid = backproject_depth(depth, cam, DepthScaleMode.RELATIVE)
    assert valid.all()

    # Verify points satisfy n . X + d = 0 to near floating-point precision
    norm = np.linalg.norm(normal)
    n_unit = normal / norm
    d_unit = distance / norm
    residuals = np.abs(points[valid] @ n_unit + d_unit)
    assert np.max(residuals) < 1e-6
    assert np.mean(residuals) < 1e-6

    # Verify fit_plane recovers plane with near-zero residual
    fit = fit_plane(points, valid)
    assert fit.rms_residual < 1e-6


def test_exact_plane_invalid_intersections_and_inputs() -> None:
    from geometry.debug import exact_plane_depth
    cam = CameraModel(10, 10, 10, 10, 4.5, 4.5)

    # Invalid normal magnitude / shape
    with pytest.raises(ValueError, match="must have 3 elements"):
        exact_plane_depth(cam, normal=(1.0, 0.0))
    with pytest.raises(ValueError, match="finite, non-zero magnitude"):
        exact_plane_depth(cam, normal=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite, non-zero magnitude"):
        exact_plane_depth(cam, normal=(np.nan, 0.0, 1.0))

    # Plane entirely behind camera (normal [0, 0, 1], distance +2.0 => Z = -2.0)
    depth_behind = exact_plane_depth(cam, normal=(0.0, 0.0, 1.0), distance=2.0)
    assert np.isnan(depth_behind).all()

    # Ray nearly parallel to plane
    # If normal is [1, 0, 0] and distance is 0, normal . [0, 0, 1] == 0
    depth_parallel = exact_plane_depth(cam, normal=(1.0, 0.0, 0.0), distance=-1.0)
    # The optical axis ray at (4.5, 4.5) has r = [0, 0, 1]^T, so n . r = 0 -> invalid/nan
    assert np.isnan(depth_parallel[4, 4]) or np.isnan(depth_parallel[5, 5])
