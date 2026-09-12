import json

import numpy as np

from geometry import CameraModel


def camera() -> CameraModel:
    return CameraModel(640, 480, 500.0, 510.0, 319.5, 239.5, (0.1, -0.01), calibrated=True)


def test_projection_unprojection_round_trip() -> None:
    cam = camera()
    points = np.array([[0.0, 0.0, 2.0], [0.4, -0.2, 3.0], [-0.5, 0.3, 1.2]])
    pixels = cam.project(points)
    reconstructed = cam.unproject(pixels[:, 0], pixels[:, 1], points[:, 2])
    np.testing.assert_allclose(reconstructed, points, atol=1e-12)


def test_known_pixel_and_depth() -> None:
    cam = camera()
    np.testing.assert_allclose(cam.unproject(319.5, 239.5, 2.0), [0.0, 0.0, 2.0])
    np.testing.assert_allclose(cam.pixel_to_ray(819.5, 239.5), [1.0, 0.0, 1.0])


def test_scaled_intrinsics() -> None:
    scaled = camera().scaled_intrinsics(320, 240)
    assert (scaled.width, scaled.height) == (320, 240)
    np.testing.assert_allclose([scaled.fx, scaled.fy, scaled.cx, scaled.cy], [250, 255, 159.75, 119.75])


def test_camera_serialization(tmp_path) -> None:
    original = camera()
    path = tmp_path / "camera.json"
    original.save_json(path)
    loaded = CameraModel.load_json(path)
    assert loaded == original
    assert json.loads(path.read_text())["coordinate_convention"]["z"] == "forward"
