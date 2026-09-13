"""Independent live display views for P123 modes 1–6."""

from __future__ import annotations

import cv2
import numpy as np

from . import common, depth, hands, normals, rgb, temporal, xyz


_VIEWS = {
    1: rgb,
    2: depth,
    3: normals,
    4: temporal,
    5: hands,
    6: xyz,
}


def render(snapshot, mode: int, display_size: tuple[int, int] | None = None) -> np.ndarray | None:
    """Render exactly one latest snapshot through the selected view module."""
    view = _VIEWS.get(int(mode), rgb)
    image, title, waiting = view.render(snapshot)
    if image is None:
        return None
    return common.finish(image, title, waiting, snapshot, int(mode), display_size)


__all__ = ["render"]
