"""Immutable P1/P2/P3 handoff data; intentionally contains no rendering code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .camera import CameraModel
from .state import GeometryState


@dataclass(frozen=True, slots=True)
class HandXYZ:
    hand_id: int
    palm_uv: tuple[float, float]
    xyz_camera: tuple[float, float, float] | None
    confidence: float
    timestamp: float
    source_frame_id: int | str
    age_ms: float
    handedness: str | None = None
    source_age_ms: float | None = None
    completion_age_ms: float | None = None


@dataclass(frozen=True, slots=True)
class P4InputState:
    """The only output boundary required by P1/P2/P3 remediation."""

    rgb_frame: np.ndarray
    rgb_capture_id: int | str
    rgb_timestamp: float
    camera_model: CameraModel
    geometry_state: GeometryState
    hand_states: tuple[HandXYZ, ...]
    data_age_metrics: dict[str, Any]
    geometry_target_capture_id: int | str | None = None
    source_depth_capture_id: int | str | None = None
    confidence_metadata: dict[str, Any] | None = None
