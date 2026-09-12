"""Development-time validation for the Person 3 -> Person 4 handoff."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .normals import NormalConfig
from .state import GeometryState


@dataclass(frozen=True, slots=True)
class GeometryValidationReport:
    valid: bool
    errors: tuple[str, ...]


def validate_renderer_geometry(
    state: GeometryState,
    *,
    normal_config: NormalConfig | None = None,
    max_history_age: int | None = None,
    tolerance: float = 1e-4,
    raise_on_error: bool = False,
) -> GeometryValidationReport:
    """Validate invariants required by a renderer without repairing state."""
    errors: list[str] = []
    normal_config = normal_config or NormalConfig()
    h, w = state.depth.shape
    valid = state.valid_mask
    if state.camera.height != h or state.camera.width != w:
        errors.append("camera resolution does not match depth")
    if state.positions_3d.shape != (h, w, 3):
        errors.append("positions_3d must have shape (H, W, 3)")
    if state.normals is None or state.normal_valid_mask is None:
        errors.append("normals and normal_valid_mask are required for renderer handoff")
    else:
        if state.normals.shape != (h, w, 3):
            errors.append("normals must have shape (H, W, 3)")
        normal_valid = state.normal_valid_mask
        if np.any(normal_valid & ~valid):
            errors.append("normal_valid_mask contains invalid geometry")
        finite_normals = np.isfinite(state.normals).all(axis=-1)
        lengths = np.linalg.norm(np.nan_to_num(state.normals), axis=-1)
        if np.any(normal_valid & ~finite_normals):
            errors.append("valid normals must be finite")
        if np.any(normal_valid & (np.abs(lengths - 1.0) > tolerance)):
            errors.append("valid normals must be unit length")
        if np.any(normal_valid & valid & (np.sum(state.normals * state.positions_3d, axis=-1) > tolerance)):
            errors.append("valid normals must face the camera")
        if state.selected_radius is not None:
            allowed = set(normal_config.radii) | {0}
            if np.any(~np.isin(state.selected_radius, tuple(allowed))):
                errors.append("selected_radius contains a value outside the configured radii")
            if np.any(normal_valid & (state.selected_radius == 0)) or np.any(~normal_valid & (state.selected_radius != 0)):
                errors.append("selected_radius is inconsistent with normal_valid_mask")
    finite_geometry = np.isfinite(state.depth) & np.isfinite(state.positions_3d).all(axis=-1)
    if np.any(valid & ~finite_geometry):
        errors.append("valid geometry must be finite")
    if np.any(valid & (state.depth <= 0)):
        errors.append("valid depth must be positive")
    if np.any(valid & (np.abs(state.positions_3d[..., 2] - state.depth) > tolerance * np.maximum(state.depth, 1.0))):
        errors.append("P_z must agree with depth")
    if state.confidence is not None:
        confidence = np.asarray(state.confidence)
        if np.any(~np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1)):
            errors.append("confidence must be finite and in [0, 1]")
        if np.any(~valid & (confidence > tolerance)):
            errors.append("invalid geometry must not have high confidence")
    if state.temporal_age is not None and max_history_age is not None:
        if np.any(state.temporal_age > max_history_age):
            errors.append("temporal_age exceeds max_history_age")
    report = GeometryValidationReport(not errors, tuple(errors))
    if raise_on_error and not report.valid:
        raise ValueError("renderer geometry validation failed: " + "; ".join(report.errors))
    return report
