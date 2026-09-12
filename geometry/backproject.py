"""Depth validation and dense camera-space back-projection."""

from __future__ import annotations

from enum import Enum

import numpy as np

from .camera import CameraModel


class DepthScaleMode(str, Enum):
    METRIC = "metric"
    RELATIVE = "relative"
    INVERSE = "inverse"


def depth_valid_mask(depth: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Return finite, positive depth validity, optionally intersected with a mask."""

    depth_arr = np.asarray(depth)
    valid = np.isfinite(depth_arr) & (depth_arr > 0)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != depth_arr.shape:
            raise ValueError("valid_mask must have the same shape as depth")
        valid &= supplied
    return valid


def backproject_depth(
    depth: np.ndarray,
    camera: CameraModel,
    scale_mode: DepthScaleMode | str = DepthScaleMode.RELATIVE,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a dense Z-depth image to ``(H,W,3)`` XYZ and a validity mask.

    Relative depth is used as a consistent arbitrary Z coordinate. Inverse
    depth/disparity is deliberately rejected because no metric or affine
    mapping to Z may be invented by this subsystem.
    """

    mode = DepthScaleMode(scale_mode)
    if mode is DepthScaleMode.INVERSE:
        raise ValueError("inverse depth requires an explicit calibrated disparity-to-Z conversion")
    depth_arr = np.asarray(depth, dtype=np.float64)
    if depth_arr.ndim != 2:
        raise ValueError("depth must be a 2D array")
    if depth_arr.shape != (camera.height, camera.width):
        raise ValueError(
            f"depth resolution {depth_arr.shape} (height={depth_arr.shape[0]}, width={depth_arr.shape[1]}) "
            f"does not match camera resolution {(camera.height, camera.width)} "
            f"(height={camera.height}, width={camera.width}); "
            "camera intrinsics must first be explicitly transformed to the depth resolution"
        )
    valid = depth_valid_mask(depth_arr, valid_mask)
    height, width = depth_arr.shape
    v, u = np.indices((height, width), dtype=np.float64)
    points = camera.unproject(u, v, depth_arr)
    points[~valid] = np.nan
    return points.astype(np.float32), valid
