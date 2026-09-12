"""Stable cross-module state contracts for depth and geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backproject import DepthScaleMode
from .camera import CameraModel


@dataclass(slots=True)
class DepthState:
    depth: np.ndarray
    timestamp: float
    source_frame_id: int | str
    scale_mode: DepthScaleMode | str
    valid_mask: np.ndarray | None = None
    confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.depth = np.asarray(self.depth)
        if self.depth.ndim != 2:
            raise ValueError("DepthState.depth must be a 2D array")
        self.scale_mode = DepthScaleMode(self.scale_mode)
        if self.valid_mask is not None and np.asarray(self.valid_mask).shape != self.depth.shape:
            raise ValueError("DepthState.valid_mask shape must match depth")
        if self.confidence is not None and np.asarray(self.confidence).shape != self.depth.shape:
            raise ValueError("DepthState.confidence shape must match depth")


@dataclass(slots=True)
class GeometryState:
    timestamp: float
    source_frame_id: int | str
    depth: np.ndarray
    positions_3d: np.ndarray
    valid_mask: np.ndarray
    camera: CameraModel
    scale_mode: DepthScaleMode | str
    normals: np.ndarray | None = None
    confidence: np.ndarray | None = None
    temporal_age: np.ndarray | None = None
    history_valid: np.ndarray | None = None
    occlusion_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.depth = np.asarray(self.depth)
        self.positions_3d = np.asarray(self.positions_3d)
        self.valid_mask = np.asarray(self.valid_mask, dtype=bool)
        self.scale_mode = DepthScaleMode(self.scale_mode)
        if self.depth.ndim != 2 or self.valid_mask.shape != self.depth.shape:
            raise ValueError("GeometryState depth and valid_mask must be matching 2D arrays")
        if self.positions_3d.shape != (*self.depth.shape, 3):
            raise ValueError("positions_3d must have shape (H, W, 3)")
