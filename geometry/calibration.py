"""Optional OpenCV checkerboard calibration and rectification utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .camera import CameraModel


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("OpenCV is required for calibration; install opencv-python-headless") from exc
    return cv2


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    camera: CameraModel
    reprojection_error_px: float
    views_used: int


def calibrate_checkerboard(
    image_paths: Iterable[str | Path],
    board_size: tuple[int, int] = (9, 6),
    square_size: float = 1.0,
) -> CalibrationResult:
    """Calibrate from checkerboard images using OpenCV's pinhole model."""

    cv2 = _cv2()
    if square_size <= 0 or min(board_size) <= 1:
        raise ValueError("board_size must contain at least 2 corners and square_size must be positive")
    object_template = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    object_template[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    object_template *= square_size
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"could not read calibration image: {path}")
        current_size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError(
                f"calibration image '{path}' resolution {current_size} (width={current_size[0]}, height={current_size[1]}) "
                f"does not match expected reference resolution {image_size} (width={image_size[0]}, height={image_size[1]})"
            )
        found, corners = cv2.findChessboardCorners(image, board_size, None)
        if not found:
            continue
        refined = cv2.cornerSubPix(
            image, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
        )
        object_points.append(object_template.copy())
        image_points.append(refined)
    if not object_points or image_size is None:
        raise ValueError("no checkerboard views were detected")
    _rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    errors = []
    for obj, observed, rvec, tvec in zip(object_points, image_points, rotations, translations):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, distortion)
        errors.append(np.mean(np.linalg.norm(projected.reshape(-1, 2) - observed.reshape(-1, 2), axis=1)))
    camera = CameraModel(
        width=image_size[0], height=image_size[1], fx=matrix[0, 0], fy=matrix[1, 1],
        cx=matrix[0, 2], cy=matrix[1, 2],
        distortion=tuple(float(x) for x in distortion.ravel()),
        calibration_width=image_size[0], calibration_height=image_size[1], calibrated=True,
    )
    return CalibrationResult(camera, float(np.mean(errors)), len(object_points))


def undistort_image(image: np.ndarray, camera: CameraModel) -> np.ndarray:
    """Return an ideal-pinhole/rectified image using the stored distortion."""

    cv2 = _cv2()
    matrix = camera.camera_matrix
    distortion = np.asarray(camera.distortion, dtype=np.float64)
    return cv2.undistort(image, matrix, distortion)
