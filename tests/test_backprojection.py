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
    # tilted_plane is intentionally emitted as float32, so this tolerance
    # includes the expected quantization error at roughly two depth units.
    assert fit.rms_residual < 1e-3
