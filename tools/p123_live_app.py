"""Live P1/P2/P3 diagnostic application (physical camera, no P4 rendering)."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from geometry.p123_live_runtime import P123LiveRuntime, P123Snapshot
from geometry.visualization import confidence_to_rgb, depth_to_rgb, normals_to_rgb_diagnostic


def _fit(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


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


def _overlay(image: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28 + 17 * len(lines)), (12, 14, 18), -1)
    cv2.putText(out, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, 43 + i * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 220, 225), 1, cv2.LINE_AA)
    return out


def _draw_mode_buttons(image: np.ndarray, active_mode: int) -> np.ndarray:
    """Draw the six clickable P123 mode controls along the bottom edge."""
    out = image.copy()
    labels = ("1 RGB", "2 DEPTH", "3 NORMALS", "4 TEMP", "5 HANDS", "6 XYZ")
    h, w = out.shape[:2]
    button_w = max(1, w // len(labels))
    for idx, label in enumerate(labels, start=1):
        x0 = (idx - 1) * button_w
        x1 = w if idx == len(labels) else idx * button_w
        color = (38, 105, 150) if idx == active_mode else (28, 32, 40)
        cv2.rectangle(out, (x0, h - 32), (x1 - 2, h - 2), color, -1)
        cv2.putText(out, label, (x0 + 8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 245, 245), 1, cv2.LINE_AA)
    return out


def _panel(
    snapshot: P123Snapshot,
    mode: int,
    display_size: tuple[int, int] | None = None,
) -> np.ndarray | None:
    rgb = snapshot.rgb_frame
    if rgb is None:
        return None
    if mode == 1:
        image = rgb
        title = "MODE 1 — RGB CAMERA"
    elif mode == 2:
        depth_state = snapshot.fast_geometry_state or snapshot.depth_state
        if depth_state is None:
            image = np.zeros_like(rgb)
            title = "MODE 2 — DEPTH"
            waiting = "waiting for depth worker"
        else:
            image = depth_to_rgb(depth_state.depth, depth_state.valid_mask)
            title = "MODE 2 — DEPTH (smoothed live, warm=near, cool=far)"
            waiting = None
    elif mode == 3:
        normal_state = snapshot.fast_geometry_state or snapshot.geometry_state
        if normal_state is None or normal_state.normals is None:
            image = np.zeros_like(rgb)
            title = "MODE 3 — NORMALS"
            waiting = "waiting for geometry worker"
        else:
            image = normals_to_rgb_diagnostic(normal_state.normals, normal_state.normal_valid_mask)
            title = "MODE 3 — NORMALS (CUDA live, R=Nx G=Ny B=Nz)"
            waiting = None
    elif mode == 4:
        temporal_state = snapshot.fast_geometry_state or snapshot.geometry_state
        if temporal_state is None or temporal_state.temporal_confidence is None:
            image = np.zeros_like(rgb)
            title = "MODE 4 — TEMPORAL / CONFIDENCE"
            waiting = "waiting for temporal geometry"
        else:
            image = confidence_to_rgb(temporal_state.temporal_confidence, temporal_state.valid_mask)
            title = "MODE 4 — TEMPORAL CONFIDENCE (fast live)" if snapshot.fast_geometry_state is not None else "MODE 4 — TEMPORAL CONFIDENCE"
            waiting = None
    elif mode == 5:
        image = rgb.copy()
        title = "MODE 5 — HANDS"
        if snapshot.hand_state is not None:
            for hand in snapshot.hand_state.hands:
                u, v = map(int, hand.palm_uv)
                cv2.circle(image, (u, v), 12, (0, 255, 210), 2)
                cv2.putText(image, f"H{hand.hand_id} {hand.confidence:.2f}", (u + 14, v), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        image = rgb.copy()
        title = "MODE 6 — XYZ CONTRACT"
        # Keep the camera coordinate convention in a fixed, readable corner.
        legend_y = max(120, image.shape[0] - 132)
        origin = (38, legend_y + 34)
        cv2.rectangle(image, (8, legend_y), (205, min(image.shape[0] - 38, legend_y + 104)), (12, 18, 22), -1)
        cv2.circle(image, origin, 3, (240, 240, 240), -1)
        cv2.arrowedLine(image, origin, (108, origin[1]), (255, 70, 70), 2, cv2.LINE_AA, tipLength=0.18)
        cv2.arrowedLine(image, origin, (origin[0], origin[1] + 54), (70, 255, 70), 2, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(image, "+X right", (112, origin[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(image, "+Y down", (origin[0] + 5, origin[1] + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 255, 90), 1, cv2.LINE_AA)
        cv2.putText(image, "+Z forward", (12, legend_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 150, 255), 1, cv2.LINE_AA)
        xyz_by_id = {item.hand_id: item for item in snapshot.xyz}
        visible_hands = () if snapshot.hand_state is None else snapshot.hand_state.hands
        if not visible_hands:
            cv2.putText(image, "Waiting for a detected hand...", (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 120), 1, cv2.LINE_AA)
        for tracked in visible_hands:
            u, v = map(int, tracked.palm_uv)
            # Keep the XYZ annotation attached to the physical hand location.
            if tracked.landmarks_uv is not None and len(tracked.landmarks_uv):
                points = np.asarray(tracked.landmarks_uv, dtype=np.int32)
                x0, y0 = np.min(points, axis=0).tolist()
                x1, y1 = np.max(points, axis=0).tolist()
                cv2.rectangle(image, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), (0, 220, 190), 2)
                for px, py in points:
                    cv2.circle(image, (int(px), int(py)), 2, (0, 190, 255), -1)
            cv2.circle(image, (u, v), 10, (0, 255, 210), 2)
            xyz = xyz_by_id.get(tracked.hand_id)
            if xyz is None or xyz.xyz_camera is None:
                label = f"H{tracked.hand_id} depth pending"
            else:
                x, y, z = xyz.xyz_camera
                label = f"H{xyz.hand_id} XYZ=({x:.2f},{y:.2f},{z:.2f}) c={xyz.confidence:.2f}"
            text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
            tx = min(max(4, u - text_width // 2), max(4, image.shape[1] - text_width - 4))
            ty = max(22, v - 18)
            cv2.rectangle(image, (tx - 3, ty - 16), (tx + text_width + 3, ty + 4), (12, 18, 22), -1)
            cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 210), 1, cv2.LINE_AA)
    if display_size is not None and (image.shape[1], image.shape[0]) != display_size:
        image = _fit(image, display_size)
    metrics = snapshot.metrics
    xyz_age = max((item.age_ms for item in snapshot.xyz), default=None)
    xyz_state = "fresh" if snapshot.xyz and (xyz_age or 9999) <= 220 else "degraded" if snapshot.xyz else "none"
    lines = [
        f"capture {snapshot.rgb_capture_id} | CAM {metrics.capture_hz or 0:.1f} Hz | overwritten {metrics.overwritten_before_consumption}",
        f"depth {metrics.depth_hz or 0:.1f} Hz age p95 {metrics.depth_age_p95_ms or 0:.0f} ms | normals {metrics.normal_hz or 0:.1f} Hz/{metrics.normal_age_p95_ms or 0:.0f}ms | temp {metrics.temporal_hz or 0:.1f} Hz",
        f"hands {metrics.hand_hz or 0:.1f} Hz/{0 if snapshot.hand_state is None else len(snapshot.hand_state.hands)} | XYZ {xyz_state}{'' if xyz_age is None else f' {xyz_age:.0f}ms'}",
    ]
    if snapshot.geometry_state is not None:
        lines.append(f"depth source {snapshot.geometry_state.source_frame_id} | processing {snapshot.geometry_state.processing_frame_id}")
    if "waiting" in locals() and waiting is not None:
        lines.append(waiting)
    return _draw_mode_buttons(_overlay(image, title, lines), mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P123 physical-camera asynchronous diagnostic app")
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--depth-backend", choices=["mariem"], default="mariem", help="Use Mariem's CUDA depth module (the sole P123 depth backend)")
    parser.add_argument(
        "--fp16", action="store_true",
        help="Use FP16 depth inference (benchmark first; FP32 is faster on some GPUs such as GTX 1650 Ti)",
    )
    parser.add_argument(
        "--depth-size", default="336",
        help="Mariem model input side in pixels (default: 336; use 420 for higher quality)",
    )
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
        runtime = P123LiveRuntime(
            camera_device=camera, width=args.width, height=args.height, fps=args.fps,
            depth_size=depth_size, depth_backend=args.depth_backend,
            use_fp16=args.fp16, hand_backend=args.hand_backend,
            full_temporal=args.full_temporal,
        )
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
