"""Validated shared frame and light contracts.

Coordinate convention at the camera boundary: +X is right, +Y is down,
+Z is forward from the camera. All positions and depths are in meters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

import numpy as np

FloatArray: TypeAlias = np.ndarray
BoolArray: TypeAlias = np.ndarray


def _finite_scalar(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _array(name: str, value: np.ndarray, *, shape: tuple[int, ...] | None = None,
           dtypes: tuple[np.dtype, ...] | None = None) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if dtypes is not None and value.dtype not in dtypes:
        expected = ", ".join(str(dtype) for dtype in dtypes)
        raise TypeError(f"{name} must have dtype {expected}, got {value.dtype}")
    return value


_FLOAT32 = (np.dtype(np.float32),)
_NORMAL_DTYPES = (np.dtype(np.float16), np.dtype(np.float32))
_BOOL = (np.dtype(np.bool_),)


@dataclass
class Light:
    """One virtual point light, expressed in camera coordinates and meters."""

    position_camera_m: FloatArray
    color_rgb: FloatArray
    intensity: float
    radius_m: float | None = None
    active: bool = True
    confidence: float = 1.0

    def __post_init__(self) -> None:
        _array("position_camera_m", self.position_camera_m, shape=(3,), dtypes=_FLOAT32)
        _array("color_rgb", self.color_rgb, shape=(3,), dtypes=_FLOAT32)
        if not np.isfinite(self.position_camera_m).all():
            raise ValueError("position_camera_m must contain only finite values")
        if not np.isfinite(self.color_rgb).all():
            raise ValueError("color_rgb must contain only finite values")
        if np.any(self.color_rgb < 0.0) or np.any(self.color_rgb > 1.0):
            raise ValueError("color_rgb values must be in the range 0..1")
        if _finite_scalar("intensity", self.intensity) < 0.0:
            raise ValueError("intensity must be >= 0")
        if self.radius_m is not None and _finite_scalar("radius_m", self.radius_m) <= 0.0:
            raise ValueError("radius_m must be > 0 when provided")
        if not isinstance(self.active, (bool, np.bool_)):
            raise TypeError("active must be a bool")
        confidence = _finite_scalar("confidence", self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in the range 0..1")


@dataclass
class LightState:
    """Latest smoothed light state; at most two lights are supported."""

    lights: list[Light]
    timestamp_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.lights, list):
            raise TypeError("lights must be a list[Light]")
        if len(self.lights) > 2:
            raise ValueError("LightState supports at most 2 lights")
        if any(not isinstance(light, Light) for light in self.lights):
            raise TypeError("every item in lights must be a Light")
        _finite_scalar("timestamp_s", self.timestamp_s)


@dataclass
class DepthFrame:
    """Metric depth and camera intrinsics for one RGB-aligned frame."""

    depth_m: FloatArray
    valid_mask: BoolArray
    fx: float
    fy: float
    cx: float
    cy: float
    timestamp_s: float

    def __post_init__(self) -> None:
        depth = _array("depth_m", self.depth_m, dtypes=_FLOAT32)
        if depth.ndim != 2 or min(depth.shape, default=0) <= 0:
            raise ValueError("depth_m must be a non-empty HxW array")
        mask = _array("valid_mask", self.valid_mask, shape=depth.shape, dtypes=_BOOL)
        if not np.isfinite(depth).all():
            raise ValueError("depth_m must be finite; use valid_mask for invalid pixels")
        if np.any(depth[mask] <= 0.0):
            raise ValueError("valid depth_m samples must be > 0 meters")
        if _finite_scalar("fx", self.fx) <= 0.0 or _finite_scalar("fy", self.fy) <= 0.0:
            raise ValueError("fx and fy must be > 0")
        _finite_scalar("cx", self.cx)
        _finite_scalar("cy", self.cy)
        _finite_scalar("timestamp_s", self.timestamp_s)


@dataclass
class NormalFrame:
    """Camera-space surface normals aligned to an RGB/depth frame."""

    normals_camera: FloatArray
    valid_mask: BoolArray
    timestamp_s: float
    edge_mask: BoolArray | None = None

    def __post_init__(self) -> None:
        normals = _array("normals_camera", self.normals_camera, dtypes=_NORMAL_DTYPES)
        if normals.ndim != 3 or normals.shape[2] != 3 or min(normals.shape[:2], default=0) <= 0:
            raise ValueError("normals_camera must be a non-empty HxWx3 array")
        mask = _array("valid_mask", self.valid_mask, shape=normals.shape[:2], dtypes=_BOOL)
        if not np.isfinite(normals).all():
            raise ValueError("normals_camera must contain only finite values")
        if np.any(np.linalg.norm(normals[mask].astype(np.float32), axis=1) <= 1e-6):
            raise ValueError("valid normals must be non-zero; normalize them before shading")
        if self.edge_mask is not None:
            _array("edge_mask", self.edge_mask, shape=normals.shape[:2], dtypes=_BOOL)
        _finite_scalar("timestamp_s", self.timestamp_s)


@dataclass
class RenderPacket:
    """All synchronized inputs consumed by the renderer for one frame."""

    rgb: np.ndarray
    depth: DepthFrame
    normals: NormalFrame
    lights: LightState
    frame_id: int
    timestamp_s: float

    def __post_init__(self) -> None:
        rgb = _array("rgb", self.rgb, dtypes=(np.dtype(np.uint8),))
        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2], default=0) <= 0:
            raise ValueError("rgb must be a non-empty HxWx3 uint8 RGB array")
        if not isinstance(self.depth, DepthFrame):
            raise TypeError("depth must be a DepthFrame")
        if not isinstance(self.normals, NormalFrame):
            raise TypeError("normals must be a NormalFrame")
        if not isinstance(self.lights, LightState):
            raise TypeError("lights must be a LightState")
        if self.depth.depth_m.shape != rgb.shape[:2]:
            raise ValueError("rgb and depth resolutions must match")
        if self.normals.normals_camera.shape[:2] != rgb.shape[:2]:
            raise ValueError("rgb and normals resolutions must match")
        if isinstance(self.frame_id, (bool, np.bool_)) or not isinstance(self.frame_id, (int, np.integer)):
            raise TypeError("frame_id must be an integer")
        if self.frame_id < 0:
            raise ValueError("frame_id must be >= 0")
        _finite_scalar("timestamp_s", self.timestamp_s)
