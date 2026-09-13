"""Mode 4: lightweight live temporal confidence."""

import numpy as np
from geometry.visualization import confidence_to_rgb

def render(snapshot):
    if snapshot.rgb_frame is None:
        return None, "MODE 4 — TEMPORAL / CONFIDENCE", "waiting for camera"
    state = snapshot.fast_geometry_state or snapshot.geometry_state
    if state is None or state.temporal_confidence is None:
        return np.zeros_like(snapshot.rgb_frame), "MODE 4 — TEMPORAL / CONFIDENCE", "waiting for temporal geometry"
    return confidence_to_rgb(state.temporal_confidence, state.valid_mask), "MODE 4 — TEMPORAL CONFIDENCE (fast live)", None
