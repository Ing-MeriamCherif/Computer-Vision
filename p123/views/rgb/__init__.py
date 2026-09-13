"""Mode 1: Live RGB camera sensor feed."""

from __future__ import annotations

from typing import Any

import numpy as np


def render(snapshot: Any) -> tuple[np.ndarray | None, str, str | None]:
    """Render physical camera raw sensor feed."""
    if snapshot.rgb_frame is None:
        return None, "MODE 1 — RGB CAMERA", "waiting for camera"
    return snapshot.rgb_frame, "MODE 1 — RGB CAMERA (Sensor Feed)", None
