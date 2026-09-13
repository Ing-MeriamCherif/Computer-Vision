"""Launcher for the P123 physical-camera diagnostic app with Google Material 3 UI."""

from __future__ import annotations

import argparse
import time
from collections import deque

import cv2

from geometry.p123_live_runtime import P123LiveRuntime
from p123.views import render as _panel
from p123.views.common import hit_test_navigation


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


def _parse_display_size(value: str, camera_size: tuple[int, int]) -> tuple[int, int]:
    text = str(value).lower().strip()
    if text in ("fhd", "1080p", "fullscreen"):
        return (1920, 1080)
    if text == "native":
        return camera_size
    if text == "auto":
        # Default to FHD for seamless fullscreen display
        return (1920, 1080)
    if "x" in text:
        w_str, h_str = text.split("x", 1)
        return int(w_str), int(h_str)
    size = int(text)
    return size, size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P123 physical-camera asynchronous diagnostic app (Material 3 UI)")
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--depth-backend", choices=["local", "colleague", "mariem"], default="local", help="Depth provider (local production default; colleague and mariem alternatives)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 depth inference (benchmark first; FP32 is faster on GTX 1650 Ti)")
    parser.add_argument("--depth-size", default="336x448", help="Local production input HxW (default: 336x448); use 'native' explicitly, or a square side for teammate providers")
    parser.add_argument("--display-size", default="1920x1080", help="UI display resolution (default: 1920x1080 FHD; or 'native', 'auto', WxH)")
    parser.add_argument("--fullscreen", action=argparse.BooleanOptionalAction, default=True, help="Run in fullscreen mode (default: True; use --no-fullscreen for windowed)")
    parser.add_argument("--full-temporal", action="store_true", help="Enable the slower CPU temporal reference worker")
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True, help="Mirror the live camera left-to-right (default: True)")
    parser.add_argument("--fourcc", choices=["auto", "MJPG", "YUYV"], default="auto")
    parser.add_argument("--hand-backend", choices=["auto", "tasks", "legacy", "colleague"], default="auto")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        cam_str = str(args.camera).strip()
        if cam_str.isdigit():
            camera = int(cam_str)
        elif cam_str.startswith("/dev/video") and cam_str[10:].isdigit():
            camera = int(cam_str[10:])
        else:
            camera = args.camera
        depth_size = _parse_depth_size(args.depth_size, (args.height, args.width))
        display_size = _parse_display_size(args.display_size, (args.width, args.height))

        runtime = P123LiveRuntime(
            camera_device=camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            depth_size=depth_size,
            depth_backend=args.depth_backend,
            use_fp16=args.fp16,
            hand_backend=args.hand_backend,
            full_temporal=args.full_temporal,
            mirror=args.mirror,
        )
        runtime.camera_worker.requested_fourcc = args.fourcc.upper()
        runtime.start()
    except Exception as exc:  # noqa: BLE001
        print(f"P123 STARTUP FAILED: {type(exc).__name__}: {exc}")
        return 1

    print("============================================================")
    print("  NRW P123 LIVE DIAGNOSTICS — MATERIAL 3 INTERFACE")
    print("============================================================")
    print(f"  Physical Camera:   {args.camera} ({args.width}x{args.height} @ {args.fps} FPS)")
    print(f"  Display Canvas:    {display_size[0]}x{display_size[1]}")
    print(f"  Depth Backend:     {args.depth_backend} (input {args.depth_size}, {'fp16' if args.fp16 else 'fp32'})")
    print(f"  Depth Runtime:     {getattr(runtime.depth_provider, 'backend_name', 'unknown')}")
    depth_device = getattr(runtime.depth_provider, "device", None)
    normal_backend = getattr(runtime, "_normal_backend", None)
    normal_device = getattr(normal_backend, "device", "cpu") if normal_backend is not None else "cpu/unavailable"
    print(f"  CUDA Devices:      depth={depth_device or 'unknown'} | normals={normal_device}")
    print("  Navigation Keys:   [1] RGB  [2] Depth  [3] Normals  [4] Temporal  [5] Hands  [6] XYZ")
    print("  Controls:          [D] Telemetry HUD  [F] Fullscreen  [Q/ESC] Quit")
    print("============================================================")

    mode = 1
    ui_state = {"mode": mode, "show_debug": False}
    started = time.monotonic()
    window = "NRW P123 Live Diagnostics (Material 3)"
    frame_times: deque[float] = deque(maxlen=30)
    is_fullscreen = bool(args.fullscreen)

    try:
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, display_size[0], display_size[1])
            if is_fullscreen:
                cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            def on_mouse(event: int, x: int, y: int, _flags: int, state: dict) -> None:
                if event == cv2.EVENT_LBUTTONUP:
                    hit = hit_test_navigation(x, y, display_size)
                    if hit is not None:
                        state["mode"] = hit

            cv2.setMouseCallback(window, on_mouse, ui_state)

        while args.duration is None or time.monotonic() - started < args.duration:
            now = time.monotonic()
            frame_times.append(now)
            display_fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else None

            snapshot = runtime.snapshot()
            if not args.headless:
                mode = int(ui_state["mode"])
                view = _panel(
                    snapshot,
                    mode,
                    display_size,
                    display_fps=display_fps,
                    show_debug=ui_state.get("show_debug", False),
                )
                if view is not None:
                    cv2.imshow(window, cv2.cvtColor(view, cv2.COLOR_RGB2BGR))

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if ord("1") <= key <= ord("6"):
                    ui_state["mode"] = key - ord("0")
                elif key in (ord("d"), ord("D")):
                    ui_state["show_debug"] = not ui_state.get("show_debug", False)
                elif key in (ord("f"), ord("F")):
                    is_fullscreen = not is_fullscreen
                    prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, prop)
                    if not is_fullscreen:
                        cv2.resizeWindow(window, display_size[0], display_size[1])
            else:
                time.sleep(0.02)
    finally:
        final = runtime.snapshot()
        runtime.stop()
        cv2.destroyAllWindows()
        print(f"Session summary: frames={final.metrics.captured} capture_hz={final.metrics.capture_hz} depth_hz={final.metrics.depth_hz} normals_hz={final.metrics.normal_hz} geometry_hz={final.metrics.geometry_hz} hand_hz={final.metrics.hand_hz} xyz_hz={final.metrics.xyz_hz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
