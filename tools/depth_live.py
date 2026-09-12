"""Camera -> DepthAnythingV2 -> Sobel vs D2NT-v2 recipe, live.

Your camera as the depth source: move your hand/objects, watch the normal
maps react. Header shows depth ms + normals ms + FPS per side.

  python tools/depth_live.py [--no-show --frames 60] [--method d2nt_v2|sobel]
Keys: 1=Sobel 2=D2NT recipe, q=quit. First run downloads the model (~100MB).
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
from depth_estimator import DepthEstimator
from normal_translator import recipe, sobel_normals
from utils import FPSMeter


def to_rgb(n: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((((n + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)),
                        cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--method", default=None)
    ap.add_argument("--depth-every", type=int, default=3,
                    help="recompute depth every Nth frame (CPU stand-in for GPU)")
    args = ap.parse_args()
    method = (args.method or config.NORMALS_METHOD).lower()
    W, H = config.CAMERA_WIDTH, config.CAMERA_HEIGHT
    fx, fy, cx, cy = config.estimate_intrinsics(W, H)
    print("[depth_live] loading depth model...")
    est = DepthEstimator()
    print(f"[depth_live] depth on {est.device} (metric={est.metric})")
    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    assert cap.isOpened(), "camera failed"
    fpsm = FPSMeter()
    i = 0
    depth = None
    t_depth = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (W, H))
            if config.CAM_MIRROR:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if depth is None or i % max(1, args.depth_every) == 0:
                t0 = time.perf_counter()
                depth = est.estimate(rgb)
                t_depth = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            if method == "sobel":
                n = sobel_normals(depth, fx, fy)
            else:
                n = recipe(depth, fx, fy, cx, cy, method,
                           config.NORMALS_SCALE, config.NORMALS_BILATERAL_D)
            t_norm = (time.perf_counter() - t0) * 1000
            fps = fpsm.tick()
            dvis = cv2.applyColorMap(cv2.normalize(
                depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_MAGMA)
            view = np.hstack([frame, dvis, to_rgb(n)])
            view = cv2.resize(view, (W * 3 // 2, H // 2))
            cv2.putText(view, f"depth {t_depth:.0f}ms | {method} {t_norm:.0f}ms | FPS {fps:.1f}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if not args.no_show:
                cv2.imshow("depth live RGB|depth|normals 1=sobel 2=d2nt q=quit", view)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                elif k == ord("1"):
                    method = "sobel"
                elif k == ord("2"):
                    method = "d2nt_v2"
            i += 1
            if args.frames and i >= args.frames:
                break
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    print(f"frames={i} method={method}")


if __name__ == "__main__":
    main()
