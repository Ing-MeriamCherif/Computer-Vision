import numpy as np

from geometry import CameraModel, ImageTransform


def test_resize_maps_intrinsics_and_points() -> None:
    cam = CameraModel(1280, 720, 800, 810, 640, 360)
    t = ImageTransform.resize(1280, 720, 640, 360)
    mapped = t.map_camera(cam, 640, 360)
    np.testing.assert_allclose([mapped.fx, mapped.fy, mapped.cx, mapped.cy], [400, 405, 320, 180])
    np.testing.assert_allclose(t.apply(np.array([640, 360])), [320, 180])


def test_crop_padding_composition_and_inverse() -> None:
    crop = ImageTransform.crop(100, 20)
    pad = ImageTransform.padding(8, 4)
    composed = crop.compose(pad)
    np.testing.assert_allclose(composed.apply([100, 20]), [8, 4])
    np.testing.assert_allclose(composed.inverse().apply(composed.apply([321.5, 55.25])), [321.5, 55.25])


def test_letterbox_preserves_aspect_and_offsets() -> None:
    transform = ImageTransform.letterbox(1280, 720, 640, 640)
    np.testing.assert_allclose([transform.sx, transform.sy, transform.tx, transform.ty], [0.5, 0.5, 0, 140])
