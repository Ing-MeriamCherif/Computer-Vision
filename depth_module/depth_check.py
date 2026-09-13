"""Depth-only check screen: camera -> async DepthModel -> RGB|depth viz.

Uses the SAME async path as full_pipeline.py (DepthWorker, latest-only),
so this validates the real pipeline depth, not the old sync DepthEstimator.

  python depth_check.py [--no-show --frames 60]   (run from inside depth_module/)

Keys: q=quit. Header shows depth age (staleness) + capture FPS.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from depth.async_depth import DepthWorker, LatestDepthBuffer, LatestFrameBuffer
from depth.config import DEPTH_CONFIG
from depth.model import DepthModel
from utils import FPSMeter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    args = ap.parse_args()

    W, H = config.CAMERA_WIDTH, config.CAMERA_HEIGHT
    device = "cuda" if DEPTH_CONFIG.device == "cuda" else "cpu"
    model = DepthModel(backend=DEPTH_CONFIG.backend, device=device,
                       input_size=DEPTH_CONFIG.input_size, fp16=False,
                       metric=False)
    print(f"[depth_check] {DEPTH_CONFIG.backend} device={device} (sync warmup...)")
    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    assert cap.isOpened(), "camera failed"
    ok, f0 = cap.read()
    f0 = cv2.resize(f0, (W, H))
    state0 = model.infer(f0)
    frame_buf, depth_buf = LatestFrameBuffer(), LatestDepthBuffer()
    depth_buf.put(state0)
    worker = DepthWorker(model, frame_buf, depth_buf)
    worker.start()

    fpsm = FPSMeter()
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (W, H))
            if config.CAM_MIRROR:
                frame = cv2.flip(frame, 1)
            frame_buf.put(frame)
            st = depth_buf.get()
            age_ms = (time.time() - st.timestamp) * 1000 if st else 1e9
            d = st.depth_map if st else np.zeros((H, W), np.float32)
            fps = fpsm.tick()
            dvis = cv2.applyColorMap(cv2.normalize(
                d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_MAGMA)
            view = np.hstack([frame, dvis])
            cv2.putText(view, f"backend={st.backend_name} age={age_ms:.0f}ms "
                              f"infer={st.inference_ms:.0f}ms FPS={fps:.1f}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if not args.no_show:
                cv2.imshow("depth check RGB|depth q=quit", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            i += 1
            if args.frames and i >= args.frames:
                break
    finally:
        worker.stop()
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    print(f"[depth_check] frames={i}")


if __name__ == "__main__":
    main()
