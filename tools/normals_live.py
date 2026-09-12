"""Live D2NT view: animated synthetic depth -> Sobel vs recipe side-by-side.

Proves the Phase-2 recipe runs in real time before real depth lands.
Box slides + depth noise -> watch edges: Sobel smears, D2NT holds them.

  python tools/normals_live.py [--no-show --frames 120] [--noise 0.005]
Keys (window): 1=Sobel 2=D2NT-v2 recipe 3=split view, q=quit.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from normal_translator import recipe, sobel_normals
from utils import FPSMeter


def frame_depth(t: float, w: int, h: int, fx: float, fy: float,
                cx: float, cy: float, noise: float):
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    x = (uu - cx) / fx
    depth = 3.0 + 0.4 * x + 0.2 * (vv - cy) / fy
    bw, bh = int(w * 0.22), int(h * 0.28)
    x0 = int(w * 0.5 - bw / 2 + 0.18 * w * np.sin(t * 0.9))
    y0 = int(h * 0.5 - bh / 2 + 0.10 * h * np.cos(t * 0.7))
    depth[y0:y0 + bh, x0:x0 + bw] = 1.5
    if noise > 0:
        depth = depth + np.random.randn(h, w) * noise
    return depth


def to_rgb(n: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((((n + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)),
                        cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--wh", nargs=2, type=int, default=[640, 480])
    args = ap.parse_args()
    w, h = args.wh
    fx = fy = (w / 2) / np.tan(np.radians(30))
    cx, cy = w / 2, h / 2
    mode = "split"
    fpsm = FPSMeter()
    i = 0
    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            depth = frame_depth(t, w, h, fx, fy, cx, cy, args.noise)
            s0 = time.perf_counter()
            ns = sobel_normals(depth, fx, fy)
            t_sobel = (time.perf_counter() - s0) * 1000
            s0 = time.perf_counter()
            nd = recipe(depth, fx, fy, cx, cy, config.NORMALS_METHOD,
                        config.NORMALS_SCALE, config.NORMALS_BILATERAL_D)
            t_d = (time.perf_counter() - s0) * 1000
            fps = fpsm.tick()
            left, right = to_rgb(ns), to_rgb(nd)
            dvis = cv2.applyColorMap(cv2.normalize(
                depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_MAGMA)
            if mode == "sobel":
                view = left
            elif mode == "d2nt":
                view = right
            else:
                view = np.vstack([np.hstack([dvis, left]),
                                  np.hstack([right, right])])
                view = cv2.resize(view, (w * 2, h))
            cv2.putText(view, f"Sobel {t_sobel:.1f}ms | D2NT {t_d:.1f}ms | FPS {fps:.1f}",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if not args.no_show:
                cv2.imshow("normals live 1=sobel 2=d2nt 3=split q=quit", view)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                elif k == ord("1"):
                    mode = "sobel"
                elif k == ord("2"):
                    mode = "d2nt"
                elif k == ord("3"):
                    mode = "split"
            i += 1
            if args.frames and i >= args.frames:
                break
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    print(f"frames={i} method={config.NORMALS_METHOD}@{config.NORMALS_SCALE}x")


if __name__ == "__main__":
    main()
