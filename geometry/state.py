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
        if self.valid_mask is not None:
            self.valid_mask = np.asarray(self.valid_mask, dtype=bool)
            if self.valid_mask.shape != self.depth.shape:
                raise ValueError(
                    f"DepthState.valid_mask shape {self.valid_mask.shape} must match depth shape {self.depth.shape}"
                )
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence)
            if self.confidence.shape != self.depth.shape:
                raise ValueError(
                    f"DepthState.confidence shape {self.confidence.shape} must match depth shape {self.depth.shape}"
                )


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
    normal_valid_mask: np.ndarray | None = None
    normal_confidence: np.ndarray | None = None
    selected_radius: np.ndarray | None = None
    spatial_confidence: np.ndarray | None = None
    history_confidence: np.ndarray | None = None
    temporal_confidence: np.ndarray | None = None
    depth_alignment_residual: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.depth = np.asarray(self.depth)
        self.positions_3d = np.asarray(self.positions_3d)
        self.valid_mask = np.asarray(self.valid_mask, dtype=bool)
        self.scale_mode = DepthScaleMode(self.scale_mode)
        if self.depth.ndim != 2:
            raise ValueError("GeometryState depth must be a 2D array")
        if self.valid_mask.shape != self.depth.shape:
            raise ValueError(
                f"GeometryState valid_mask shape {self.valid_mask.shape} must match depth shape {self.depth.shape}"
            )
        h, w = self.depth.shape
        if self.positions_3d.shape != (h, w, 3):
            raise ValueError(
                f"positions_3d must have shape ({h}, {w}, 3), got {self.positions_3d.shape}"
            )
        if self.camera.height != h or self.camera.width != w:
            raise ValueError(
                f"camera resolution (height={self.camera.height}, width={self.camera.width}) does not match "
                f"depth shape (height={h}, width={w})"
            )
        if self.normals is not None:
            self.normals = np.asarray(self.normals)
            if self.normals.shape != (h, w, 3):
                raise ValueError(
                    f"normals must have shape ({h}, {w}, 3), got {self.normals.shape}"
                )
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence)
            if self.confidence.shape != (h, w):
                raise ValueError(
                    f"confidence must have shape ({h}, {w}), got {self.confidence.shape}"
                )
        if self.temporal_age is not None:
            self.temporal_age = np.asarray(self.temporal_age)
            if self.temporal_age.shape != (h, w):
                raise ValueError(
                    f"temporal_age must have shape ({h}, {w}), got {self.temporal_age.shape}"
                )
        if self.history_valid is not None:
            self.history_valid = np.asarray(self.history_valid, dtype=bool)
            if self.history_valid.shape != (h, w):
                raise ValueError(
                    f"history_valid must have shape ({h}, {w}), got {self.history_valid.shape}"
                )
        if self.occlusion_mask is not None:
            self.occlusion_mask = np.asarray(self.occlusion_mask, dtype=bool)
            if self.occlusion_mask.shape != (h, w):
                raise ValueError(
                    f"occlusion_mask must have shape ({h}, {w}), got {self.occlusion_mask.shape}"
                )
        if self.normal_valid_mask is not None:
            self.normal_valid_mask = np.asarray(self.normal_valid_mask, dtype=bool)
            if self.normal_valid_mask.shape != (h, w):
                raise ValueError(
                    f"normal_valid_mask must have shape ({h}, {w}), got {self.normal_valid_mask.shape}"
                )
        if self.normal_confidence is not None:
            self.normal_confidence = np.asarray(self.normal_confidence)
            if self.normal_confidence.shape != (h, w):
                raise ValueError(
                    f"normal_confidence must have shape ({h}, {w}), got {self.normal_confidence.shape}"
                )
        if self.selected_radius is not None:
            self.selected_radius = np.asarray(self.selected_radius)
            if self.selected_radius.shape != (h, w):
                raise ValueError(
                    f"selected_radius must have shape ({h}, {w}), got {self.selected_radius.shape}"
                )
        for name in ("spatial_confidence", "history_confidence", "temporal_confidence", "depth_alignment_residual"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value)
                if value.shape != (h, w):
                    raise ValueError(f"{name} must have shape ({h}, {w}), got {value.shape}")
                setattr(self, name, value)
