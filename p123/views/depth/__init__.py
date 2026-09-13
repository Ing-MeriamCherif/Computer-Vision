"""Mode 2: native-resolution depth visualization with display-only stabilization."""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.visualization import _robust_bounds, depth_to_rgb


class _DisplayRange:
    def __init__(self) -> None:
        self.bounds: tuple[float, float] | None = None

    def update(self, depth: np.ndarray, valid: np.ndarray | None) -> tuple[float, float]:
        current = _robust_bounds(depth, np.isfinite(depth) & (depth > 1e-6) if valid is None else np.asarray(valid, dtype=bool))
        if self.bounds is None:
            self.bounds = current
        else:
            lo, hi = self.bounds
            # EMA prevents per-frame percentile pumping while still following
            # genuine scene changes; this state affects display only.
            self.bounds = (0.15 * current[0] + 0.85 * lo, 0.15 * current[1] + 0.85 * hi)
        return self.bounds


_DISPLAY_RANGE = _DisplayRange()


def render(snapshot: Any) -> tuple[np.ndarray | None, str, str | None]:
    """Render the selected CUDA depth map (warm=near, cool=far)."""
    if snapshot.rgb_frame is None:
        return None, "MODE 2 — DEPTH", "waiting for camera"
    state = snapshot.fast_geometry_state or snapshot.depth_state
    if state is None or state.depth is None:
        dimmed = (snapshot.rgb_frame // 4).astype(np.uint8)
        return dimmed, "MODE 2 — DEPTH", "waiting for depth worker"
    bounds = _DISPLAY_RANGE.update(state.depth, state.valid_mask)
    return depth_to_rgb(state.depth, state.valid_mask, bounds=bounds), "MODE 2 — DEPTH", None
