"""Mode 4: Lightweight live temporal confidence."""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.visualization import confidence_to_rgb


def render(snapshot: Any) -> tuple[np.ndarray | None, str, str | None]:
    """Render temporal confidence consistency map."""
    if snapshot.rgb_frame is None:
        return None, "MODE 4 — TEMPORAL CONFIDENCE", "waiting for camera"
    state = snapshot.fast_geometry_state or snapshot.geometry_state
    if state is None or state.temporal_confidence is None:
        dimmed = (snapshot.rgb_frame // 4).astype(np.uint8)
        return dimmed, "MODE 4 — TEMPORAL CONFIDENCE", "waiting for temporal geometry"
    return confidence_to_rgb(state.temporal_confidence, state.valid_mask), "MODE 4 — TEMPORAL CONFIDENCE (Stability)", None
