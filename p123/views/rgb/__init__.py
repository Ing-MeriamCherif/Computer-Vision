"""Mode 1: live RGB camera view."""

def render(snapshot):
    if snapshot.rgb_frame is None:
        return None, "MODE 1 — RGB CAMERA", "waiting for camera"
    return snapshot.rgb_frame, "MODE 1 — RGB CAMERA", None
