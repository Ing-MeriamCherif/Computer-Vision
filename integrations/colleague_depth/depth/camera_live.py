"""Live camera depth viewer (stays open until you quit).

Usage:
    python -m depth.camera_live --camera-index 0 --input-size 336

Press q in the window to quit. Shows camera | depth side by side with
inference ms / FPS overlay. Start with 336 for speed, raise to 518 for quality.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Live camera depth viewer")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--backend", default="depth_anything_v2_small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--input-size", type=int, default=336,
                        help="Model input size (336 fast, 518 detailed)")
    args = parser.parse_args()

    import cv2

    from depth.model import DepthModel

    model = DepthModel(backend=args.backend, device=args.device,
                       input_size=args.input_size, fp16=args.fp16)
    print("Loading model...", end=" ", flush=True)
    model.warmup(iterations=3)
    print("ready.")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Camera {args.camera_index} did not open.")

    # Force MJPG codec — YUYV is slow and can produce black frames on USB cams.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Discard first frames — USB cameras often deliver blank/black initially.
    print("Warming up camera...", end=" ", flush=True)
    for _ in range(30):
        cap.read()
    print("done.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    print(f"Camera {args.camera_index}: {w}x{h} @ {fps_cam:.0f}fps fourcc={fourcc_str}")
    print(f"Live depth on camera {args.camera_index} — press q to quit.")

    ema_ms = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera stopped delivering frames.")
                break

            t0 = time.perf_counter()
            state = model.infer(frame)
            ms = (time.perf_counter() - t0) * 1000.0
            ema_ms = ms if ema_ms is None else 0.9 * ema_ms + 0.1 * ms

            depth_u8 = (np.clip(state.depth_map, 0.0, 1.0) * 255.0).astype(np.uint8)
            colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
            if colored.shape[:2] != frame.shape[:2]:
                colored = cv2.resize(colored, (frame.shape[1], frame.shape[0]))
            view = np.hstack([frame, colored])
            cv2.putText(view, f"{ema_ms:.1f}ms {1000.0 / ema_ms:.1f}FPS {args.input_size}x{args.input_size}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("camera_live [camera | depth] (q to quit)", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
