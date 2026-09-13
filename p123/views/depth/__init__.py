"""Mode 2: latest Mariem depth visualization."""

import numpy as np
from geometry.visualization import depth_to_rgb

def render(snapshot):
    if snapshot.rgb_frame is None:
        return None, "MODE 2 — DEPTH", "waiting for camera"
    state = snapshot.fast_geometry_state or snapshot.depth_state
    if state is None:
        return np.zeros_like(snapshot.rgb_frame), "MODE 2 — DEPTH", "waiting for depth worker"
    return depth_to_rgb(state.depth, state.valid_mask), "MODE 2 — DEPTH (Mariem live, warm=near, cool=far)", None
