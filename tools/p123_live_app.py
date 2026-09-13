"""Launcher for the P123 physical-camera diagnostic app with Google Material 3 UI."""

from __future__ import annotations

import argparse
import time
from collections import deque

import cv2

from geometry.p123_live_runtime import P123LiveRuntime
from p123.views import relight as relight_view
from p123.views import render as _panel
from p123.views.common import hit_test_navigation, hit_test_source_toggle
from p123.views.relight import configure as configure_relight


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


def _camera_key(value: str | int) -> tuple[str, int | str]:
    text = str(value).strip()
    if text.isdigit():
        return "device", int(text)
    if text.startswith("/dev/video") and text[10:].isdigit():
        return "device", int(text[10:])
    return "name", text.casefold()


def _source_profile(args: argparse.Namespace, source: str, *, legacy_camera_is_phone: bool) -> tuple[str, int, int]:
    if source == "phone":
        return str(args.phone_camera), 1920, 1080
    camera = args.camera if args.camera is not None and not legacy_camera_is_phone else args.webcam_camera
    return str(camera), args.width, args.height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P123 physical-camera asynchronous diagnostic app (Material 3 UI)")
    parser.add_argument("--camera", default=None, help="Legacy initial camera selector; use --webcam-camera/--phone-camera to configure both sources")
    parser.add_argument("--webcam-camera", default="/dev/video0", help="Webcam device used by the in-app source toggle")
    parser.add_argument("--phone-camera", default="/dev/video2", help="Phone device used by the in-app source toggle")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--depth-backend", choices=["local", "colleague", "mariem"], default="local", help="Depth provider (local production default; colleague and mariem alternatives)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 depth inference (benchmark first; FP32 is faster on GTX 1650 Ti)")
    parser.add_argument("--depth-size", default="336x448", help="Local production input HxW (default: 336x448); use 'native' explicitly, or a square side for teammate providers")
    parser.add_argument("--display-size", default="1920x1080", help="UI display resolution (default: 1920x1080 FHD; or 'native', 'auto', WxH)")
    parser.add_argument("--fullscreen", action=argparse.BooleanOptionalAction, default=True, help="Run in fullscreen mode (default: True; use --no-fullscreen for windowed)")
    parser.add_argument("--mode", type=int, choices=range(1, 8), default=1, help="Starting view: 1 RGB through 7 hand relight")
    parser.add_argument("--lighting-quality", choices=["low", "balanced", "high"], default="balanced")
    parser.add_argument("--relight-backend", choices=["auto", "rtx", "raster"], default="auto", help="Mode 7 renderer selection")
    parser.add_argument("--full-temporal", action="store_true", help="Enable the slower CPU temporal reference worker")
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True, help="Mirror the live camera left-to-right (default: True)")
    parser.add_argument("--fourcc", choices=["auto", "MJPG", "YUYV"], default="auto")
    parser.add_argument("--hand-backend", choices=["auto", "tasks", "legacy", "colleague"], default="auto")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_relight(args.lighting_quality, args.relight_backend)
    legacy_camera_is_phone = args.camera is not None and _camera_key(args.camera) == _camera_key(args.phone_camera)
    selected_source = "phone" if legacy_camera_is_phone else "webcam"

    def make_runtime(source: str, depth_provider=None) -> P123LiveRuntime:
        camera_name, source_width, source_height = _source_profile(
            args, source, legacy_camera_is_phone=legacy_camera_is_phone
        )
        cam_str = camera_name.strip()
        if cam_str.isdigit():
            camera_device = int(cam_str)
        elif cam_str.startswith("/dev/video") and cam_str[10:].isdigit():
            camera_device = int(cam_str[10:])
        else:
            camera_device = camera_name
        source_depth_size = _parse_depth_size(args.depth_size, (source_height, source_width))
        result = P123LiveRuntime(
            camera_device=camera_device,
            width=source_width,
            height=source_height,
            fps=args.fps,
            depth_provider=depth_provider,
            depth_size=source_depth_size,
            depth_backend=args.depth_backend,
            use_fp16=args.fp16,
            hand_backend=args.hand_backend,
            full_temporal=args.full_temporal,
            mirror=args.mirror,
        )
        result.camera_worker.requested_fourcc = args.fourcc.upper()
        result.start()
        return result

    try:
        _, initial_width, initial_height = _source_profile(
            args, selected_source, legacy_camera_is_phone=legacy_camera_is_phone
        )
        display_size = _parse_display_size(args.display_size, (initial_width, initial_height))
        runtime = make_runtime(selected_source)
    except Exception as exc:  # noqa: BLE001
        print(f"P123 STARTUP FAILED: {type(exc).__name__}: {exc}")
        return 1

    print("============================================================")
    print("  NRW P123 LIVE DIAGNOSTICS — MATERIAL 3 INTERFACE")
    print("============================================================")
    active_camera, active_width, active_height = _source_profile(
        args, selected_source, legacy_camera_is_phone=legacy_camera_is_phone
    )
    print(f"  Physical Camera:   {active_camera} ({active_width}x{active_height} @ {args.fps} FPS)")
    print(f"  Display Canvas:    {display_size[0]}x{display_size[1]}")
    print(f"  Depth Backend:     {args.depth_backend} (input {args.depth_size}, {'fp16' if args.fp16 else 'fp32'})")
    print(f"  Depth Runtime:     {getattr(runtime.depth_provider, 'backend_name', 'unknown')}")
    depth_device = getattr(runtime.depth_provider, "device", None)
    normal_backend = getattr(runtime, "_normal_backend", None)
    normal_device = getattr(normal_backend, "device", "cpu") if normal_backend is not None else "cpu/unavailable"
    print(f"  CUDA Devices:      depth={depth_device or 'unknown'} | normals={normal_device}")
    print("  Navigation Keys:   [1] RGB  [2] Depth  [3] Normals  [4] Temporal  [5] Hands  [6] XYZ  [7] Relight")
    print(f"  Relight Quality:   {args.lighting_quality}")
    print(f"  Relight Backend:   {args.relight_backend} (auto selects RTX OptiX or OpenGL raster)")
    print("  Controls:          [D] Telemetry HUD  [F] Fullscreen  [Q/ESC] Quit")
    print("============================================================")

    mode = int(args.mode)
    ui_state = {"mode": mode, "show_debug": False, "source": selected_source, "requested_source": None}
    started = time.monotonic()
    window = "NRW P123 Live Diagnostics (Material 3)"
    frame_times: deque[float] = deque(maxlen=30)
    display_fps: float | None = None
    last_relight_state: tuple[str, int] | None = None
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
                    source = hit_test_source_toggle(x, y, display_size)
                    if source is not None and source != state["source"]:
                        state["requested_source"] = source

            cv2.setMouseCallback(window, on_mouse, ui_state)

        while args.duration is None or time.monotonic() - started < args.duration:
            requested_source = ui_state.pop("requested_source", None)
            if requested_source is not None and requested_source != selected_source:
                shared_depth = runtime.depth_provider
                runtime.stop()
                try:
                    runtime = make_runtime(requested_source, depth_provider=shared_depth)
                    selected_source = requested_source
                    ui_state["source"] = selected_source
                    relight_view._renderer.reset_for_source_change()
                except Exception as exc:  # noqa: BLE001
                    print(f"CAMERA SWITCH FAILED ({requested_source}): {type(exc).__name__}: {exc}")
                    runtime = make_runtime(selected_source, depth_provider=shared_depth)
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
                    camera_source=selected_source,
                )
                if mode == 7:
                    relight = relight_view._renderer
                    stats = relight.last_lighting_stats
                    state = (str(stats.get("renderer", "WAITING")), int(relight.last_light_count))
                    if state != last_relight_state:
                        print(
                            f"Mode 7 state: renderer={state[0]} lights={state[1]} "
                            f"xyz_age_ms={relight.last_xyz_source_age_ms} "
                            f"gpu_render_ms={stats.get('gpu_render_ms', stats.get('lighting_ms'))}",
                            flush=True,
                        )
                        last_relight_state = state
                if view is not None:
                    cv2.imshow(window, cv2.cvtColor(view, cv2.COLOR_RGB2BGR))

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if ord("1") <= key <= ord("7"):
                    ui_state["mode"] = key - ord("0")
                elif key in (ord("d"), ord("D")):
                    ui_state["show_debug"] = not ui_state.get("show_debug", False)
                elif key in (ord("c"), ord("C")):
                    ui_state["requested_source"] = "phone" if selected_source == "webcam" else "webcam"
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
        relight_view._renderer.close()
        print(
            "Session summary: "
            f"frames={final.metrics.captured} capture_hz={final.metrics.capture_hz} "
            f"depth_hz={final.metrics.depth_hz} normals_hz={final.metrics.normal_hz} "
            f"hand_hz={final.metrics.hand_hz} xyz_hz={final.metrics.xyz_hz} "
            f"display_fps={display_fps} geometry_age_p95_ms={final.metrics.geometry_age_p95_ms} "
            f"hand_age_p95_ms={final.metrics.hand_age_p95_ms} xyz_age_p95_ms={final.metrics.xyz_age_p95_ms}"
        )
        relight_stats = relight_view._renderer.last_lighting_stats
        if relight_stats:
            print(
                "Relight summary: "
                f"renderer={relight_stats.get('renderer')} "
                f"quality={relight_stats.get('quality')} lights={relight_stats.get('lights')} "
                f"gpu_render_ms={relight_stats.get('gpu_render_ms', relight_stats.get('lighting_ms'))} "
                f"shadows={relight_stats.get('shadow_quality')} volumes={relight_stats.get('volumetric_quality')} "
                f"geometry_age_ms={relight_view._renderer.last_geometry_age_ms} "
                f"xyz_source_age_ms={relight_view._renderer.last_xyz_source_age_ms}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
