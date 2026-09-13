"""Independent live display views for P123 modes 1–6."""

from __future__ import annotations

from typing import Any

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


def render(
    snapshot: Any,
    mode: int,
    display_size: tuple[int, int] | None = None,
    display_fps: float | None = None,
    show_debug: bool = False,
) -> np.ndarray:
    """Render exactly one latest snapshot through the selected view module."""
    view = _VIEWS.get(int(mode), rgb)
    image, title, waiting = view.render(snapshot)
    return common.finish(
        image,
        title,
        waiting,
        snapshot,
        int(mode),
        display_size=display_size,
        display_fps=display_fps,
        show_debug=show_debug,
    )


__all__ = ["common", "render"]
