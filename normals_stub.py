"""Sobel normal stub on dummy depth (shape-ready for real depth later).

Fast (<5ms @480p). Output HxWx3 float32 unit normals, camera coords.
Dummy depth = smooth gradient + synthetic foreground rect so the 2x2
grid shows something before DepthAnything lands.
"""
from __future__ import annotations

import cv2
import numpy as np


def dummy_depth(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = 2.0 + 0.5 * (xx / w)  # background slope 2.0-2.5m
    # foreground rect at 1.2m (center) to visualize edges
    x0, x1 = int(w * 0.35), int(w * 0.65)
    y0, y1 = int(h * 0.3), int(h * 0.7)
    depth[y0:y1, x0:x1] = 1.2
    return depth.astype(np.float32)


def depth_to_normals_sobel(depth: np.ndarray, fx: float, fy: float) -> np.ndarray:
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    nx = -fx * gx
    ny = -fy * gy
    nz = np.ones_like(depth)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    return np.stack([nx / norm, ny / norm, nz / norm], axis=-1).astype(np.float32)


def normals_to_rgb(normals: np.ndarray) -> np.ndarray:
    vis = ((normals + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR for imshow
