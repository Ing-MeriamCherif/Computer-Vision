"""Phase 2 — depth -> normals, unified interface.

Methods:
  sobel       Phase-1 baseline (fast, blurs edges)
  d2nt_basic  D2NT paper eq.9, no 3D coords, FD gradients
  d2nt_v2     + Discontinuity-Aware Gradient (DAG) filter
  d2nt_v3     + DAG + MRF-based refinement (recommended by authors)

Convention out: camera coords +X right, +Y down, +Z forward, unit length.
D2NT may return normals facing away (-Z); we flip toward camera.
"""
from __future__ import annotations

import cv2
import numpy as np


def _orient_toward_camera(n: np.ndarray) -> np.ndarray:
    flip = n[:, :, 2] < 0
    n[flip] *= -1.0
    return n


def sobel_normals(depth_m: np.ndarray, fx: float, fy: float) -> np.ndarray:
    # Identical to Phase-1 normals_stub (contexte.md form): n = (-fx*dZ/du, -fy*dZ/dv, 1)
    # NOTE: OpenCV Sobel ksize=3 sums to 8x the true gradient -> scale=1/8.
    # (The Phase-1 stub omits this; display-only, never measured.)
    d = depth_m.astype(np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    nx, ny = -fx * gx, -fy * gy
    nz = np.ones_like(d)
    n = np.stack([nx, ny, nz], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return _orient_toward_camera(n.astype(np.float64))


def d2nt_normals(depth_m: np.ndarray, K: np.ndarray,
                 version: str = "d2nt_v3") -> np.ndarray:
    from d2nt import depth2normal
    d = np.ascontiguousarray(depth_m.astype(np.float64))
    n = depth2normal(d, np.asarray(K, dtype=np.float64), version=version)
    n = np.asarray(n, dtype=np.float64)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return _orient_toward_camera(n)


def translate(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float,
              method: str = "d2nt_v3") -> np.ndarray:
    if method == "sobel":
        return sobel_normals(depth_m, fx, fy)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return d2nt_normals(depth_m, K, version=method)


def recipe(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float,
           method: str = "d2nt_v2", scale: float = 0.5,
           bilateral_d: int = 5) -> np.ndarray:
    """Phase-2 winning recipe: downscale -> bilateral -> translate -> upscale.

    Bilateral + D2NT at half-res beats full-res Sobel on noisy depth at
    lower cost (bench: edge 13.6 vs 44.7 deg, ~13 vs ~19ms).
    """
    h, w = depth_m.shape[:2]
    if scale < 1.0:
        dw, dh = int(w * scale), int(h * scale)
        d = cv2.resize(depth_m.astype(np.float32), (dw, dh),
                       interpolation=cv2.INTER_LINEAR)
        fx_s, fy_s, cx_s, cy_s = fx * scale, fy * scale, cx * scale, cy * scale
    else:
        d = depth_m.astype(np.float32)
        dh, dw, fx_s, fy_s, cx_s, cy_s = h, w, fx, fy, cx, cy
    if bilateral_d > 0:
        d = cv2.bilateralFilter(d, bilateral_d, 0.05, bilateral_d)
    n = translate(d, fx_s, fy_s, cx_s, cy_s, method)
    if scale < 1.0:
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return _orient_toward_camera(n)
