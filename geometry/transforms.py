"""Explicit affine image-coordinate transforms for resolution changes.

Coordinate and Pixel Semantics Convention
-----------------------------------------
1. Continuous Pixel Coordinates:
   The coordinate system uses a continuous 2D plane with origin (0.0, 0.0) at the
   outer top-left corner of the image boundary. Discrete pixel indices (u, v) with
   u in [0, W - 1] and v in [0, H - 1] correspond to discrete sample points, with
   their geometric centers at (u + 0.5, v + 0.5) under standard continuous image
   conventions. In pinhole ray projection, rays are cast through coordinate (u, v).

2. Affine Coordinate Mapping:
   Image transforms apply an affine 2D coordinate map:
       target_x = sx * source_x + tx
       target_y = sy * source_y + ty
   Where:
   - Resize scales coordinates: sx = target_w / source_w, sy = target_h / source_h
   - Crop translates coordinates: tx = -crop_left, ty = -crop_top
   - Padding translates coordinates: tx = pad_left, ty = pad_top
   - Letterbox applies isotropic scaling followed by centering translations.

3. Camera Intrinsics Transformation:
   Under map_camera(), intrinsics transform according to:
       fx' = sx * fx
       fy' = sy * fy
       cx' = sx * cx + tx
       cy' = sy * cy + ty
   This guarantees that camera rays unprojected at target pixel coordinates
   correspond identically to rays unprojected from source coordinates.

4. External Framework Integration & Half-Pixel Caveats:
   - OpenCV (cv2.resize): Bilinear interpolation samples continuous coordinates as
     (u + 0.5) * s - 0.5.
   - PyTorch (torch.nn.functional.interpolate / grid_sample): "align_corners=False"
     maps pixel centers with half-pixel adjustments, whereas "align_corners=True"
     maps corner pixel centers to corner pixel centers.
   - Model preprocessors (e.g. Depth Anything V2, torchvision transforms): May
     follow different rounding or half-pixel conventions.
   - Framework Integration Rule: When moving between this geometry subsystem and
     external framework tensors/sampling grids, differences in half-pixel alignment
     must be explicitly handled at integration boundaries; do not rely on implicit
     coincidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import CameraModel


@dataclass(frozen=True, slots=True)
class ImageTransform:
    """Map source pixel coordinates to target coordinates.

    Coordinates are continuous top-left-origin pixel coordinates and the map is:
    ``target = (sx * source_x + tx, sy * source_y + ty)``. Keeping crop and
    padding offsets explicit prevents incorrect principal-point updates.

    See module docstring for complete coordinate, pixel-center, and framework
    boundary documentation.
    """

    sx: float = 1.0
    sy: float = 1.0
    tx: float = 0.0
    ty: float = 0.0

    def __post_init__(self) -> None:
        if not all(np.isfinite(v) for v in (self.sx, self.sy, self.tx, self.ty)):
            raise ValueError("transform parameters must be finite")
        if self.sx == 0 or self.sy == 0:
            raise ValueError("transform scale cannot be zero")

    @classmethod
    def resize(cls, source_width: int, source_height: int, target_width: int, target_height: int) -> "ImageTransform":
        if min(source_width, source_height, target_width, target_height) <= 0:
            raise ValueError("image dimensions must be positive")
        return cls(target_width / source_width, target_height / source_height)

    @classmethod
    def crop(cls, left: float, top: float) -> "ImageTransform":
        return cls(tx=-float(left), ty=-float(top))

    @classmethod
    def padding(cls, left: float, top: float) -> "ImageTransform":
        return cls(tx=float(left), ty=float(top))

    @classmethod
    def letterbox(cls, source_width: int, source_height: int, target_width: int, target_height: int) -> "ImageTransform":
        if min(source_width, source_height, target_width, target_height) <= 0:
            raise ValueError("image dimensions must be positive")
        scale = min(target_width / source_width, target_height / source_height)
        return cls(scale, scale, (target_width - source_width * scale) / 2.0,
                   (target_height - source_height * scale) / 2.0)

    def compose(self, after: "ImageTransform") -> "ImageTransform":
        """Return ``after(self(source))``."""

        return ImageTransform(after.sx * self.sx, after.sy * self.sy,
                              after.sx * self.tx + after.tx,
                              after.sy * self.ty + after.ty)

    def inverse(self) -> "ImageTransform":
        return ImageTransform(1.0 / self.sx, 1.0 / self.sy, -self.tx / self.sx, -self.ty / self.sy)

    def apply(self, points: np.ndarray) -> np.ndarray:
        points_arr = np.asarray(points, dtype=np.float64)
        if points_arr.shape[-1] != 2:
            raise ValueError("points must have a final dimension of length 2")
        return points_arr * np.array((self.sx, self.sy)) + np.array((self.tx, self.ty))

    def map_camera(self, camera: CameraModel, width: int, height: int) -> CameraModel:
        """Map camera intrinsics through this transform to a target image."""

        if width <= 0 or height <= 0:
            raise ValueError("target dimensions must be positive")
        return CameraModel(
            width=width, height=height,
            fx=camera.fx * self.sx, fy=camera.fy * self.sy,
            cx=camera.cx * self.sx + self.tx, cy=camera.cy * self.sy + self.ty,
            distortion=camera.distortion,
            calibration_width=camera.calibration_width,
            calibration_height=camera.calibration_height,
            calibrated=camera.calibrated,
        )
