"""Live async depth viewer — optimized for FPS.

Architecture:
    Camera thread -> LatestFrameBuffer -> DepthWorker (background)
                                              |
                                         LatestDepthBuffer
                                              |
                                         Renderer (main thread)

Press q to quit. Move mouse over depth view for exact pixel value.
"""

from __future__ import annotations

import argparse
import threading
import time

import cv2
import numpy as np

from depth.async_depth import DepthWorker, LatestDepthBuffer, LatestFrameBuffer
from depth.config import DEPTH_CONFIG
from depth.model import DepthModel


def camera_capture_loop(cap, frame_buffer: LatestFrameBuffer, running: threading.Event):
    while running.is_set():
        ok, frame = cap.read()
        if ok and frame is not None:
            frame_buffer.put(frame)
        else:
            time.sleep(0.005)


def open_camera(index: int, width: int = 640, height: int = 480):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Camera {index} did not open.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(30):
        cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {index}: {w}x{h}")
    return cap


def main():
    parser = argparse.ArgumentParser(description="Live async depth")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--input-size", type=int, default=DEPTH_CONFIG.input_size)
    parser.add_argument("--device", default=DEPTH_CONFIG.device)
    parser.add_argument("--metric", action="store_true", default=DEPTH_CONFIG.metric,
                        help="Use metric model (depth in meters)")
    args = parser.parse_args()

    cap = open_camera(args.camera_index)
    model = DepthModel(backend=DEPTH_CONFIG.backend, device=args.device,
                       input_size=args.input_size, fp16=DEPTH_CONFIG.fp16,
                       metric=args.metric)
    print("Loading model...", end=" ", flush=True)
    model.warmup(iterations=3)
    print("ready.")

    frame_buffer = LatestFrameBuffer()
    depth_buffer = LatestDepthBuffer()
    worker = DepthWorker(model, frame_buffer, depth_buffer)
    worker.start()

    running = threading.Event()
    running.set()
    cam_thread = threading.Thread(target=camera_capture_loop,
                                  args=(cap, frame_buffer, running), daemon=True)
    cam_thread.start()

    ema_ms = None
    ema_fps = None
    frame_count = 0
    mouse_x, mouse_y = -1, -1
    MAX_DEPTH_AGE_MS = 200

    def on_mouse(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_x, mouse_y = x, y

    win = "async_depth (q to quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("Live — press q to quit. Mouse over depth for pixel value.")

    try:
        while True:
            state = depth_buffer.get()
            frame = frame_buffer.get()
            if frame is None or state is None:
                cv2.waitKey(1)
                continue

            # Reject stale depth
            age_ms = (time.time() - state.timestamp) * 1000
            if age_ms > MAX_DEPTH_AGE_MS:
                cv2.waitKey(1)
                continue

            ms = state.inference_ms
            ema_ms = ms if ema_ms is None else 0.9 * ema_ms + 0.1 * ms
            ema_fps = (1000.0 / ema_ms) if ema_ms > 0 else 0
            frame_count += 1

            depth = state.depth_map
            # For display: metric needs normalization, relative is already 0-1
            if args.metric:
                d_min, d_max = depth.min(), depth.max()
                if d_max - d_min > 1e-6:
                    depth_norm = (depth - d_min) / (d_max - d_min)
                else:
                    depth_norm = np.zeros_like(depth)
                depth_u8 = (np.clip(depth_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                depth_u8 = (np.clip(depth, 0.0, 1.0) * 255.0).astype(np.uint8)
            colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
            if colored.shape[:2] != frame.shape[:2]:
                colored = cv2.resize(colored, (frame.shape[1], frame.shape[0]))
            view = np.hstack([frame, colored])

            # Crosshair on depth half
            cam_w = frame.shape[1]
            if mouse_x > cam_w and mouse_y >= 0:
                dx, dy = mouse_x - cam_w, mouse_y
                dh, dw = depth.shape
                sx = max(0, min(int(dx / colored.shape[1] * dw), dw - 1))
                sy = max(0, min(int(dy / colored.shape[0] * dh), dh - 1))
                val = depth[sy, sx]
                cv2.line(view, (cam_w, mouse_y), (view.shape[1], mouse_y), (0, 255, 255), 1)
                cv2.line(view, (mouse_x, 0), (mouse_x, view.shape[0]), (0, 255, 255), 1)
                unit = "m" if args.metric else ""
                label = f"({sx},{sy}) {val:.4f}{unit}"
                lx = mouse_x + 10 if mouse_x + 120 < view.shape[1] else mouse_x - 120
                ly = mouse_y - 10 if mouse_y > 30 else mouse_y + 20
                cv2.putText(view, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # Stats overlay
            mode = "METRIC" if args.metric else "RELATIVE"
            w_stats = worker.stats_dict()
            line1 = f"{mode} {args.input_size}x{args.input_size}  {ema_ms:.1f}ms  {ema_fps:.1f}FPS"
            line2 = f"depth_age={age_ms:.0f}ms  processed={w_stats['processed']}  worker={'alive' if w_stats['alive'] else 'DEAD'}"
            cv2.putText(view, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(view, line2, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(win, view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        running.clear()
        worker.stop()
        cam_thread.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()
        w = worker.stats_dict()
        print(f"\nDone. rendered={frame_count}  processed={w['processed']}  skipped={w['skipped']}")


if __name__ == "__main__":
    main()
