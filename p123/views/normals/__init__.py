"""Mode 3: CUDA multiscale surface normals."""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.visualization import normals_to_rgb_diagnostic


def render(snapshot: Any) -> tuple[np.ndarray | None, str, str | None]:
    """Render multiscale surface normals (R=Nx, G=Ny, B=Nz)."""
    if snapshot.rgb_frame is None:
        return None, "MODE 3 — NORMALS", "waiting for camera"
    state = snapshot.fast_geometry_state or snapshot.geometry_state
    if state is None or state.normals is None:
        dimmed = (snapshot.rgb_frame // 4).astype(np.uint8)
        return dimmed, "MODE 3 — NORMALS", "waiting for geometry worker"
    return normals_to_rgb_diagnostic(state.normals, state.normal_valid_mask), "MODE 3 — SURFACE NORMALS (CUDA Multiscale)", None
