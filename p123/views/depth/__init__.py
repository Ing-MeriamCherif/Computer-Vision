"""Mode 2: Latest Mariem CUDA depth visualization."""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.visualization import depth_to_rgb


def render(snapshot: Any) -> tuple[np.ndarray | None, str, str | None]:
    """Render Mariem CUDA depth map (warm=near, cool=far)."""
    if snapshot.rgb_frame is None:
        return None, "MODE 2 — DEPTH", "waiting for camera"
    state = snapshot.fast_geometry_state or snapshot.depth_state
    if state is None or state.depth is None:
        dimmed = (snapshot.rgb_frame // 4).astype(np.uint8)
        return dimmed, "MODE 2 — DEPTH", "waiting for depth worker"
    return depth_to_rgb(state.depth, state.valid_mask), "MODE 2 — DEPTH (Mariem CUDA)", None
