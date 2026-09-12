"""Validation-only render: warm tint + light arrow + 2x2 grid.

NOT final shading (no Lambertian/specular/shadow here — Person 4 owns that).
Proves: palm -> L_pos -> intensity path works live.
"""
from __future__ import annotations

import cv2
import numpy as np


def apply_warm_tint(bgr: np.ndarray, intensity: float) -> np.ndarray:
    # Legacy global tint (kept for compat). Prefer apply_torch_glow: the
    # global version flashes the whole screen whenever the hand appears.
    out = bgr.astype(np.float32)
    out[:, :, 2] *= intensity * 1.1  # R
    out[:, :, 1] *= intensity        # G
    out[:, :, 0] *= intensity * 0.9  # B
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_torch_glow(bgr: np.ndarray, palm_uv, intensity: float,
                     radius_px: float = 260.0) -> np.ndarray:
    """Torch-style light: LOCAL radial glow around the hand, base image untouched.

    This is why the old view "turned lighter": the tint multiplied every pixel
    by intensity, so the whole screen flashed on appear/disappear. The glow
    adds warm light with gaussian falloff only near the torch, like a real
    flame — further pixels get nothing, matching the intensity falloff model.
    Computed at half-res for speed, then upscaled.
    """
    if palm_uv is None or intensity <= 0.01:
        return bgr
    h, w = bgr.shape[:2]
    sh, sw = h // 2, w // 2
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    u, v = palm_uv[0] * sw / w, palm_uv[1] * sh / h
    r = max(radius_px * sw / w, 1.0)
    d2 = ((xx - u) ** 2 + (yy - v) ** 2) / (r * r)
    fall = np.exp(-d2 * 2.5).astype(np.float32) * min(float(intensity), 1.5)
    glow = np.zeros((sh, sw, 3), dtype=np.float32)
    glow[:, :, 2] = 255.0 * fall * 0.55   # warm R
    glow[:, :, 1] = 170.0 * fall * 0.35   # G
    glow[:, :, 0] = 100.0 * fall * 0.25   # B
    glow = cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(bgr.astype(np.float32) + glow, 0, 255).astype(np.uint8)


def draw_light_arrow(bgr: np.ndarray, palm_uv, intensity: float,
                     z_m: float | None = None, held: bool = False) -> np.ndarray:
    h, w = bgr.shape[:2]
    out = bgr.copy()
    if palm_uv is None:
        return out
    u, v = int(palm_uv[0]), int(palm_uv[1])
    color = (255, 255, 0) if held else (0, 255, 255)
    cv2.circle(out, (u, v), 10, color, 2)
    cv2.arrowedLine(out, (w // 2, h // 2), (u, v), (0, 255, 0), 2, tipLength=0.08)
    label = f"I={intensity:.2f}"
    if z_m is not None:
        label += f" Z={z_m:.2f}m"
    if held:
        label += " (held)"
    cv2.putText(out, label, (u + 12, v - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def make_grid(rgb_landmarks_bgr: np.ndarray, depth: np.ndarray,
              normals_bgr: np.ndarray, tinted_bgr: np.ndarray,
              fps: float, det_fps: float | None = None) -> np.ndarray:
    h, w = rgb_landmarks_bgr.shape[:2]
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    d = cv2.applyColorMap(d, cv2.COLORMAP_MAGMA)
    d = cv2.resize(d, (w, h))
    n = cv2.resize(normals_bgr, (w, h))
    t = cv2.resize(tinted_bgr, (w, h))
    top = np.hstack([rgb_landmarks_bgr, d])
    bot = np.hstack([n, t])
    grid = np.vstack([top, bot])
    cv2.putText(grid, f"FPS {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    if det_fps is not None:
        cv2.putText(grid, f"DET {det_fps:.1f}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    return grid
