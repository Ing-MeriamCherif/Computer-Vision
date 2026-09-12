"""Pinhole camera model and vectorized projection helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CameraModel:
    """A calibrated pinhole camera in the project camera-space convention.

    Coordinates are ``X=image-right``, ``Y=image-down``, and
    ``Z=forward``. ``depth`` is always a Z-depth, not Euclidean ray length.
    Projection methods operate on ideal (rectified) pixels by default. Lens
    distortion is retained for calibration and can be applied by the optional
    OpenCV helpers in :mod:`geometry.calibration`.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = ()
    calibration_width: int | None = None
    calibration_height: int | None = None
    calibrated: bool = False

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        for name in ("fx", "fy", "cx", "cy"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("fx and fy must be positive")
        if self.calibration_width is not None and self.calibration_width <= 0:
            raise ValueError("calibration_width must be positive")
        if self.calibration_height is not None and self.calibration_height <= 0:
            raise ValueError("calibration_height must be positive")
        object.__setattr__(self, "distortion", tuple(float(x) for x in self.distortion))

    @property
    def calibration_resolution(self) -> tuple[int, int]:
        return (
            self.calibration_width or self.width,
            self.calibration_height or self.height,
        )

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def pixel_to_ray(self, u: ArrayLike, v: ArrayLike) -> np.ndarray:
        """Return z-normalized camera rays with shape ``broadcast(u,v)+(3,)``.

        The returned ray is ``(x, y, 1)`` rather than a unit vector. This makes
        multiplying by Z-depth exactly equivalent to :meth:`unproject`.
        """

        u_arr, v_arr = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
        x = (u_arr - self.cx) / self.fx
        y = (v_arr - self.cy) / self.fy
        return np.stack((x, y, np.ones_like(x)), axis=-1)

    def unproject(self, u: ArrayLike, v: ArrayLike, depth: ArrayLike) -> np.ndarray:
        """Unproject pixels and Z-depth into camera-space XYZ, vectorized."""

        u_arr, v_arr, z_arr = np.broadcast_arrays(
            np.asarray(u, dtype=np.float64),
            np.asarray(v, dtype=np.float64),
            np.asarray(depth, dtype=np.float64),
        )
        rays = self.pixel_to_ray(u_arr, v_arr)
        points = rays * z_arr[..., None]
        invalid = ~np.isfinite(z_arr) | (z_arr <= 0)
        return np.where(invalid[..., None], np.nan, points)

    def project(self, points: ArrayLike) -> np.ndarray:
        """Project camera-space XYZ points to ideal pixels, vectorized."""

        points_arr = np.asarray(points, dtype=np.float64)
        if points_arr.shape[-1] != 3:
            raise ValueError("points must have a final dimension of length 3")
        z = points_arr[..., 2]
        valid = np.isfinite(points_arr).all(axis=-1) & (z > 0)
        safe_z = np.where(valid, z, 1.0)
        pixels = np.stack(
            (self.fx * points_arr[..., 0] / safe_z + self.cx,
             self.fy * points_arr[..., 1] / safe_z + self.cy),
            axis=-1,
        )
        return np.where(valid[..., None], pixels, np.nan)

    def scaled_intrinsics(self, width: int, height: int) -> "CameraModel":
        """Return this model mapped to a resized image of ``width x height``."""

        if width <= 0 or height <= 0:
            raise ValueError("target width and height must be positive")
        sx, sy = width / self.width, height / self.height
        return CameraModel(
            width=width,
            height=height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            distortion=self.distortion,
            calibration_width=self.calibration_width,
            calibration_height=self.calibration_height,
            calibrated=self.calibrated,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion": list(self.distortion),
            "calibration_resolution": [*self.calibration_resolution],
            "calibration_resolution_explicit": self.calibration_width is not None and self.calibration_height is not None,
            "calibrated": self.calibrated,
            "coordinate_convention": {"x": "right", "y": "down", "z": "forward"},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraModel":
        resolution = data.get("calibration_resolution", [data["width"], data["height"]])
        explicit_resolution = bool(data.get("calibration_resolution_explicit", "calibration_resolution" in data))
        return cls(
            width=int(data["width"]), height=int(data["height"]),
            fx=float(data["fx"]), fy=float(data["fy"]),
            cx=float(data["cx"]), cy=float(data["cy"]),
            distortion=tuple(float(x) for x in data.get("distortion", ())),
            calibration_width=int(resolution[0]) if explicit_resolution else None,
            calibration_height=int(resolution[1]) if explicit_resolution else None,
            calibrated=bool(data.get("calibrated", False)),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "CameraModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
