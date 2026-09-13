"""Mode 3: CUDA multiscale surface normals."""

import numpy as np
from geometry.visualization import normals_to_rgb_diagnostic

def render(snapshot):
    if snapshot.rgb_frame is None:
        return None, "MODE 3 — NORMALS", "waiting for camera"
    state = snapshot.fast_geometry_state or snapshot.geometry_state
    if state is None or state.normals is None:
        return np.zeros_like(snapshot.rgb_frame), "MODE 3 — NORMALS", "waiting for geometry worker"
    return normals_to_rgb_diagnostic(state.normals, state.normal_valid_mask), "MODE 3 — NORMALS (CUDA live, R=Nx G=Ny B=Nz)", None
