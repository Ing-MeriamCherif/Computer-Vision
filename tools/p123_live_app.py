"""Thin launcher for the P123 physical-camera diagnostic views."""

from __future__ import annotations

import argparse
import time

import cv2

from geometry.p123_live_runtime import P123LiveRuntime
from p123.views import render as _panel


def _parse_depth_size(value: str, native: tuple[int, int]) -> tuple[int, int]:
    text = str(value).lower().strip()
    if text == "native":
        return native
    if "x" in text:
        h, w = (int(part) for part in text.split("x", 1))
        if min(h, w) < 14:
            raise ValueError("--depth-size dimensions must be >= 14")
        return h, w
    size = int(text)
    if size < 14:
        raise ValueError("--depth-size must be >= 14")
    return size, size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P123 physical-camera asynchronous diagnostic app")
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--depth-backend", choices=["mariem"], default="mariem", help="Use Mariem's CUDA depth module (the sole P123 depth backend)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 depth inference (benchmark first; FP32 is faster on GTX 1650 Ti)")
    parser.add_argument("--depth-size", default="336", help="Mariem model input side in pixels (default: 336; use 420 for higher quality)")
    parser.add_argument("--full-temporal", action="store_true", help="Enable the slower CPU temporal reference worker")
    parser.add_argument("--fourcc", choices=["auto", "MJPG", "YUYV"], default="auto")
    parser.add_argument("--hand-backend", choices=["auto", "tasks", "legacy", "colleague"], default="auto")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        camera = int(args.camera) if str(args.camera).isdigit() else args.camera
        depth_size = _parse_depth_size(args.depth_size, (args.height, args.width))
        runtime = P123LiveRuntime(camera_device=camera, width=args.width, height=args.height, fps=args.fps,
                                  depth_size=depth_size, depth_backend=args.depth_backend, use_fp16=args.fp16,
                                  hand_backend=args.hand_backend, full_temporal=args.full_temporal)
        runtime.camera_worker.requested_fourcc = args.fourcc.upper()
        runtime.start()
    except Exception as exc:
        print(f"P123 STARTUP FAILED: {type(exc).__name__}: {exc}")
        return 1
    print("P123 ASYNCHRONOUS LIVE RUNTIME")
    print(f"physical camera={args.camera} resolution={args.width}x{args.height} requested_fps={args.fps}")
    print(f"depth backend={args.depth_backend} input={args.depth_size} precision={'fp16' if args.fp16 else 'fp32'}")
    depth_device = getattr(runtime.depth_provider, "device", None)
    normal_backend = getattr(runtime, "_normal_backend", None)
    normal_device = getattr(normal_backend, "device", "cpu") if normal_backend is not None else "cpu/unavailable"
    print(f"cuda depth={depth_device or 'unknown'} normals={normal_device}")
    print("keys: 1 RGB  2 depth  3 normals  4 temporal/confidence  5 hands  6 XYZ  q quit")
    mode = 1
    ui_state = {"mode": mode}
    started = time.monotonic()
    window = "NRW P123 Live Diagnostics"
    try:
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

            def on_mouse(event, x, y, _flags, state):
                if event == cv2.EVENT_LBUTTONUP and y >= max(0, args.height - 42) and 0 <= x < args.width:
                    state["mode"] = min(6, max(1, int(x / max(args.width / 6.0, 1.0)) + 1))

            cv2.setMouseCallback(window, on_mouse, ui_state)
        while args.duration is None or time.monotonic() - started < args.duration:
            snapshot = runtime.snapshot()
            if not args.headless:
                mode = int(ui_state["mode"])
                view = _panel(snapshot, mode, (args.width, args.height))
                if view is not None:
                    cv2.imshow(window, cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if ord("1") <= key <= ord("6"):
                    mode = key - ord("0")
                    ui_state["mode"] = mode
            else:
                time.sleep(0.02)
    finally:
        final = runtime.snapshot()
        runtime.stop()
        cv2.destroyAllWindows()
        print(f"frames={final.metrics.captured} capture_hz={final.metrics.capture_hz} depth_hz={final.metrics.depth_hz} normals_hz={final.metrics.normal_hz} geometry_hz={final.metrics.geometry_hz} hand_hz={final.metrics.hand_hz} xyz_hz={final.metrics.xyz_hz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
