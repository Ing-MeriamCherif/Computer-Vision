"""Mode 2: native-resolution depth visualization with display-only stabilization."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from geometry.visualization import _robust_bounds, depth_to_rgb


class _DisplayRange:
    def __init__(self) -> None:
        self.bounds: tuple[float, float] | None = None
        self.updates = 0

    def update(self, depth: np.ndarray, valid: np.ndarray | None) -> tuple[float, float]:
        current = _robust_bounds(depth, np.isfinite(depth) & (depth > 1e-6) if valid is None else np.asarray(valid, dtype=bool))
        if self.bounds is None:
            self.bounds = current
        else:
            lo, hi = self.bounds
            self.updates += 1
            # Expansion follows genuine new depth slowly; contraction is much
            # slower so a hand cannot globally pump the palette. Clamp each
            # update to 3% after the one-second warm-up.
            alpha = 0.05 if self.updates < 15 else 0.01
            candidate = (alpha * current[0] + (1 - alpha) * lo, alpha * current[1] + (1 - alpha) * hi)
            max_step = 0.03
            self.bounds = tuple(old + float(np.clip(new - old, -abs(old) * max_step, abs(old) * max_step)) for old, new in zip((lo, hi), candidate))
        return self.bounds


_DISPLAY_RANGE = _DisplayRange()
_LAST_FAST_STATE: Any | None = None


def render(snapshot: Any) -> tuple[np.ndarray | None, str, str | None]:
    """Render the selected CUDA depth map (warm=near, cool=far)."""
    if snapshot.rgb_frame is None:
        return None, "MODE 2 — DEPTH", "waiting for camera"
    # Once the fast path publishes, retain its latest generation during
    # worker gaps; never alternate raw and smoothed maps frame-to-frame.
    global _LAST_FAST_STATE
    if snapshot.fast_geometry_state is not None:
        _LAST_FAST_STATE = snapshot.fast_geometry_state
    state = _LAST_FAST_STATE or snapshot.depth_state
    if state is None or state.depth is None:
        dimmed = (snapshot.rgb_frame // 4).astype(np.uint8)
        return dimmed, "MODE 2 — DEPTH", "waiting for depth worker"
    depth = np.asarray(state.depth, dtype=np.float32)
    valid = None if state.valid_mask is None else np.asarray(state.valid_mask, dtype=bool)
    target_shape = tuple(np.asarray(snapshot.rgb_frame).shape[:2])
    if depth.shape != target_shape:
        # Some teammate providers publish a lower-resolution map. Upsample
        # only the display copy to the native camera canvas; GeometryState is
        # left untouched for downstream geometry correctness.
        target_w, target_h = target_shape[1], target_shape[0]
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        if valid is not None:
            valid = cv2.resize(valid.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    bounds = _DISPLAY_RANGE.update(depth, valid)
    return depth_to_rgb(depth, valid, bounds=bounds), "MODE 2 — DEPTH", None
