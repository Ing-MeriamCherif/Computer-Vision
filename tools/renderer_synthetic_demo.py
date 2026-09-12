"""Interactive P0/P1 renderer demo using synthetic or webcam RGB input."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contracts.render_types import DepthFrame, Light, LightState, NormalFrame, RenderPacket
from renderer.renderer import DebugMode, Renderer


def letterbox_bgr_to_rgb(frame_bgr: np.ndarray, destination_rgb: np.ndarray, cv2) -> tuple[int, int, int, int]:
    """Aspect-fit one BGR camera frame into a reusable RGB canvas."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"camera frame must be HxWx3, got {frame_bgr.shape}")
    if destination_rgb.ndim != 3 or destination_rgb.shape[2] != 3 or destination_rgb.dtype != np.uint8:
        raise ValueError("destination must be an HxWx3 uint8 RGB array")

    source_h, source_w = frame_bgr.shape[:2]
    target_h, target_w = destination_rgb.shape[:2]
    scale = min(target_w / source_w, target_h / source_h)
    scaled_w = max(1, min(target_w, int(round(source_w * scale))))
    scaled_h = max(1, min(target_h, int(round(source_h * scale))))
    x0 = (target_w - scaled_w) // 2
    y0 = (target_h - scaled_h) // 2

    resized_bgr = cv2.resize(frame_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    destination_rgb.fill(0)
    destination_rgb[y0:y0 + scaled_h, x0:x0 + scaled_w] = resized_bgr[:, :, ::-1]
    return x0, y0, scaled_w, scaled_h


@dataclass
class WebcamSource:
    capture: object
    cv2: object
    camera_index: int
    reported_width: int
    reported_height: int
    reported_fps: float
    received_width: int
    received_height: int
    first_frame: np.ndarray | None
    frames_received: int = 0

    @classmethod
    def open(cls, camera_index: int) -> "WebcamSource":
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for webcam input; install requirements.txt") from exc

        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"could not open webcam at camera index {camera_index}")
        reported_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        reported_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise RuntimeError(f"webcam index {camera_index} opened but did not return a frame")
        received_height, received_width = frame.shape[:2]
        return cls(
            capture=capture,
            cv2=cv2,
            camera_index=camera_index,
            reported_width=reported_width,
            reported_height=reported_height,
            reported_fps=reported_fps,
            received_width=received_width,
            received_height=received_height,
            first_frame=frame,
        )

    def update_packet(self, packet: RenderPacket, frame_index: int) -> float:
        start_ns = time.perf_counter_ns()
        if self.first_frame is not None:
            frame = self.first_frame
            self.first_frame = None
        else:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"webcam index {self.camera_index} stopped returning frames")
        actual_h, actual_w = frame.shape[:2]
        self.received_width = actual_w
        self.received_height = actual_h
        letterbox_bgr_to_rgb(frame, packet.rgb, self.cv2)
        now = time.perf_counter()
        packet.frame_id = frame_index
        packet.timestamp_s = now
        packet.depth.timestamp_s = now
        packet.normals.timestamp_s = now
        packet.lights.timestamp_s = now
        self.frames_received += 1
        return (time.perf_counter_ns() - start_ns) / 1_000_000.0

    def release(self) -> None:
        self.capture.release()


def make_synthetic_packet(width: int = 960, height: int = 540) -> RenderPacket:
    """Build one RGB-aligned test frame; no CPU pixel-by-pixel rendering is used."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    checker = (((np.arange(width)[None, :] // 32) + (np.arange(height)[:, None] // 32)) % 2).astype(np.uint8)

    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip(35.0 + 150.0 * x + 25.0 * checker, 0.0, 255.0).astype(np.uint8)
    rgb[:, :, 1] = np.clip(45.0 + 145.0 * y + 20.0 * (1.0 - x), 0.0, 255.0).astype(np.uint8)
    rgb[:, :, 2] = np.clip(105.0 + 95.0 * (1.0 - y) + 30.0 * checker, 0.0, 255.0).astype(np.uint8)

    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    foreground = (xx >= int(width * 0.30)) & (xx < int(width * 0.70)) & (yy >= int(height * 0.25)) & (yy < int(height * 0.75))
    depth_m = np.full((height, width), 3.0, dtype=np.float32)
    depth_m[foreground] = 1.5
    rgb[foreground, 0] = 225
    rgb[foreground, 1] = 105
    rgb[foreground, 2] = 45

    # Camera-facing surfaces point toward the camera (-Z); a small tilted patch
    # makes the normal debug mode show a second, distinct direction.
    normals = np.zeros((height, width, 3), dtype=np.float32)
    normals[:, :, 2] = -1.0
    tilted = foreground & (xx >= int(width * 0.40)) & (xx < int(width * 0.52))
    normals[tilted, 0] = 0.5
    normals[tilted, 2] = -np.sqrt(np.float32(0.75))
    valid_mask = np.ones((height, width), dtype=np.bool_)
    now = time.perf_counter()

    light = Light(
        position_camera_m=np.array([0.0, 0.0, 2.2], dtype=np.float32),
        color_rgb=np.array([1.0, 0.92, 0.78], dtype=np.float32),
        intensity=1.0,
        radius_m=0.08,
        active=True,
        confidence=1.0,
    )
    return RenderPacket(
        rgb=rgb,
        depth=DepthFrame(
            depth_m=depth_m,
            valid_mask=valid_mask,
            fx=float(width) * 0.9,
            fy=float(height) * 0.9,
            cx=(width - 1) * 0.5,
            cy=(height - 1) * 0.5,
            timestamp_s=now,
        ),
        normals=NormalFrame(normals_camera=normals, valid_mask=valid_mask.copy(), timestamp_s=now),
        lights=LightState(lights=[light], timestamp_s=now),
        frame_id=0,
        timestamp_s=now,
    )


def _move_light(glfw, window, light: Light, delta_s: float, speed_m_s: float) -> None:
    direction = np.zeros(3, dtype=np.float32)
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        direction[0] -= 1.0
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        direction[0] += 1.0
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        direction[1] -= 1.0  # W moves up (-Y); +Y points down.
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        direction[1] += 1.0
    if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
        direction[2] -= 1.0  # Q moves toward the camera (-Z).
    if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
        direction[2] += 1.0  # E moves forward into the scene (+Z).
    magnitude = float(np.linalg.norm(direction))
    if magnitude > 0.0:
        light.position_camera_m += direction * (speed_m_s * delta_s / magnitude)


def _print_gl_info(context) -> None:
    info = context.info
    print("OpenGL information:")
    for label, key in (
        ("GPU vendor", "GL_VENDOR"),
        ("GPU renderer", "GL_RENDERER"),
        ("OpenGL version", "GL_VERSION"),
    ):
        print(f"  {label}: {info.get(key, 'unavailable')}")

    glsl_version = info.get("GL_SHADING_LANGUAGE_VERSION")
    if not glsl_version:
        try:
            import ctypes
            if sys.platform == "win32":
                opengl = ctypes.WinDLL("opengl32.dll")
            elif sys.platform == "darwin":
                opengl = ctypes.CDLL("/System/Library/Frameworks/OpenGL.framework/OpenGL")
            else:
                opengl = ctypes.CDLL("libGL.so.1")
            gl_get_string = opengl.glGetString
            gl_get_string.argtypes = [ctypes.c_uint]
            gl_get_string.restype = ctypes.c_char_p
            raw_version = gl_get_string(0x8B8C)  # GL_SHADING_LANGUAGE_VERSION
            if raw_version:
                glsl_version = raw_version.decode("ascii", "replace")
        except Exception as exc:
            glsl_version = f"unavailable ({type(exc).__name__}: {exc})"
    print(f"  GLSL version: {glsl_version or 'unavailable'}")
    renderer_name = str(info.get("GL_RENDERER", "unavailable"))
    print(f"GPU: {renderer_name}")
    if "NVIDIA GeForce RTX 4060" not in renderer_name:
        print("Benchmark classification: INVALID FOR TARGET PERFORMANCE")
    try:
        import glfw
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor) if monitor else None
        if mode is not None:
            print(f"  Primary display refresh: {mode.refresh_rate} Hz")
    except Exception:
        pass


def _run_benchmark(
    glfw,
    context,
    renderer: Renderer,
    window,
    packet: RenderPacket,
    *,
    warmup_frames: int,
    measured_frames: int,
    light_speed: float,
    data_prep_ms: float,
    context_setup_ms: float,
    renderer_setup_ms: float,
    vsync_on: bool,
    input_name: str,
    prepare_frame: Callable[[RenderPacket, int], float] | None,
) -> int:
    """Run unlogged warmup frames, then collect per-stage CPU timings."""
    total_frames = warmup_frames + measured_frames
    warmup_ms = np.empty(warmup_frames, dtype=np.float64)
    full_ms = np.empty(measured_frames, dtype=np.float64)
    upload_ms = np.empty(measured_frames, dtype=np.float64)
    render_submit_ms = np.empty(measured_frames, dtype=np.float64)
    swap_ms = np.empty(measured_frames, dtype=np.float64)
    frame_prep_ms = np.empty(measured_frames, dtype=np.float64)
    previous_time = time.perf_counter()

    gpu_query_count = min(12, measured_frames)
    gpu_query_indices = set(np.linspace(0, measured_frames - 1, gpu_query_count, dtype=np.int64).tolist())
    gpu_queries = {}
    gpu_query_error = None
    try:
        # Allocate queries before measurement, and retrieve them only after the run.
        gpu_queries = {index: context.query(time=True) for index in gpu_query_indices}
    except Exception as exc:  # Some drivers may not expose timer queries.
        gpu_queries = {}
        gpu_query_error = f"{type(exc).__name__}: {exc}"

    for frame_index in range(total_frames):
        if glfw.window_should_close(window):
            raise RuntimeError(f"Benchmark window closed after {frame_index} of {total_frames} frames")

        frame_start_ns = time.perf_counter_ns()
        glfw.poll_events()
        now = time.perf_counter()
        delta_s = min(now - previous_time, 0.1)
        previous_time = now
        _move_light(glfw, window, packet.lights.lights[0], delta_s, light_speed)

        prep_ms = prepare_frame(packet, frame_index) if prepare_frame is not None else 0.0

        upload_start_ns = time.perf_counter_ns()
        renderer.upload_packet(packet)
        upload_end_ns = time.perf_counter_ns()

        render_start_ns = time.perf_counter_ns()
        measured_index = frame_index - warmup_frames
        gpu_query = gpu_queries.get(measured_index)
        if gpu_query is not None:
            # Results are read only after the run, so query retrieval cannot stall measured frames.
            with gpu_query:
                renderer.render(packet, DebugMode.RGB, upload_inputs=False)
        else:
            renderer.render(packet, DebugMode.RGB, upload_inputs=False)
        render_end_ns = time.perf_counter_ns()

        swap_start_ns = time.perf_counter_ns()
        glfw.swap_buffers(window)
        frame_end_ns = time.perf_counter_ns()

        frame_time_ms = (frame_end_ns - frame_start_ns) / 1_000_000.0
        if frame_index < warmup_frames:
            warmup_ms[frame_index] = frame_time_ms
        else:
            sample_index = frame_index - warmup_frames
            full_ms[sample_index] = frame_time_ms
            upload_ms[sample_index] = (upload_end_ns - upload_start_ns) / 1_000_000.0
            render_submit_ms[sample_index] = (render_end_ns - render_start_ns) / 1_000_000.0
            swap_ms[sample_index] = (frame_end_ns - swap_start_ns) / 1_000_000.0
            frame_prep_ms[sample_index] = prep_ms

    average_frame_ms = float(np.mean(full_ms))
    warmup_average_ms = float(np.mean(warmup_ms))
    print(f"{input_name.title()} benchmark: VSync {'ON' if vsync_on else 'OFF'} | {warmup_frames} warmup frames ignored | {measured_frames} measured frames | {packet.rgb.shape[1]}x{packet.rgb.shape[0]}")
    print(f"Warmup throughput (diagnostic only, excluded from results): {1000.0 / warmup_average_ms:.2f} FPS ({warmup_average_ms:.3f} ms/frame)")
    short_sample_count = min(45, warmup_frames)
    short_sample_ms = float(np.mean(warmup_ms[:short_sample_count]))
    print(f"First {short_sample_count} frames (short-run diagnostic, excluded): {1000.0 / short_sample_ms:.2f} FPS ({short_sample_ms:.3f} ms/frame)")
    print(f"Initial RGB/depth/normal packet preparation: {data_prep_ms:.3f} ms (one-time)")
    print(f"GLFW window/context initialization: {context_setup_ms:.3f} ms (one-time)")
    print(f"Renderer initialization (shader compile, VAO, texture allocation/upload): {renderer_setup_ms:.3f} ms (one-time)")
    print(f"Average FPS: {1000.0 / average_frame_ms:.2f}")
    print(f"Average full frame time: {average_frame_ms:.3f} ms")
    print(f"Median full frame time: {float(np.median(full_ms)):.3f} ms")
    print(f"P95 full frame time: {float(np.percentile(full_ms, 95)):.3f} ms")
    print(f"Max full frame time: {float(np.max(full_ms)):.3f} ms (min FPS {1000.0 / float(np.max(full_ms)):.2f})")
    print(f"Average texture upload CPU-call time: {float(np.mean(upload_ms)):.3f} ms/frame")
    if prepare_frame is not None:
        print(f"Average webcam capture/letterbox time: {float(np.mean(frame_prep_ms)):.3f} ms/frame")
    print(f"Average render CPU-submit time: {float(np.mean(render_submit_ms)):.3f} ms/frame")
    print(f"Average buffer-swap time: {float(np.mean(swap_ms)):.3f} ms/frame")
    if gpu_queries:
        try:
            gpu_render_ms = np.array([query.elapsed / 1_000_000.0 for query in gpu_queries.values()], dtype=np.float64)
            print(
                "GPU render timer query (sampled draws): "
                f"avg {float(np.mean(gpu_render_ms)):.3f} ms, "
                f"median {float(np.median(gpu_render_ms)):.3f} ms, "
                f"p95 {float(np.percentile(gpu_render_ms, 95)):.3f} ms, "
                f"n={gpu_render_ms.size}; results read after run"
            )
        except Exception as exc:
            print(f"GPU render timer query unavailable: {type(exc).__name__}: {exc}")
    elif gpu_query_error:
        print(f"GPU render timer query unavailable: {gpu_query_error}")
    print("Render CPU-submit time does not force GPU completion; buffer swap is timed separately.")
    return 0


def run_demo(
    width: int,
    height: int,
    max_frames: int | None,
    light_speed: float,
    *,
    benchmark: bool,
    benchmark_frames: int,
    warmup_frames: int,
    vsync_on: bool,
    input_name: str,
    camera_index: int,
) -> int:
    try:
        import glfw
        import moderngl
    except ImportError as exc:
        print(
            f"Missing graphics dependency: {exc}. Install with `python -m pip install -r requirements.txt`.",
            file=sys.stderr,
        )
        return 2

    if not glfw.init():
        print("GLFW initialization failed: glfw.init() returned False.", file=sys.stderr)
        return 1

    window = None
    context = None
    renderer = None
    status_line_open = False
    webcam = None
    try:
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        context_setup_start_ns = time.perf_counter_ns()
        window = glfw.create_window(width, height, f"Person 4 Renderer ({input_name})", None, None)
        if not window:
            print(f"GLFW window creation failed for {width}x{height} OpenGL 3.3.", file=sys.stderr)
            return 1

        glfw.make_context_current(window)
        glfw.swap_interval(1 if vsync_on else 0)
        context = moderngl.create_context(require=330)
        context_setup_ms = (time.perf_counter_ns() - context_setup_start_ns) / 1_000_000.0

        data_prep_start_ns = time.perf_counter_ns()
        packet = make_synthetic_packet(width=width, height=height)
        data_prep_ms = (time.perf_counter_ns() - data_prep_start_ns) / 1_000_000.0
        prepare_frame = None
        if input_name == "webcam":
            webcam = WebcamSource.open(camera_index)
            webcam.update_packet(packet, 0)
            prepare_frame = webcam.update_packet
            reported_fps = f"{webcam.reported_fps:.2f}" if webcam.reported_fps > 0.0 else "unavailable"
            print(f"Webcam index: {camera_index}")
            print(f"Webcam reported resolution: {webcam.reported_width}x{webcam.reported_height}")
            print(f"Webcam reported FPS: {reported_fps}")
            print(f"Webcam received frame size: {webcam.received_width}x{webcam.received_height}")
            print(f"Webcam display: aspect-preserving letterbox, BGR converted to RGB, no image/video saved")

        renderer_setup_start_ns = time.perf_counter_ns()
        renderer = Renderer(context, packet)
        renderer_setup_ms = (time.perf_counter_ns() - renderer_setup_start_ns) / 1_000_000.0
        _print_gl_info(context)
        state = {"mode": DebugMode.RGB}

        def on_key(callback_window, key, _scancode, action, _mods):
            if action not in (glfw.PRESS, glfw.REPEAT):
                return
            if key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3):
                state["mode"] = DebugMode(key - glfw.KEY_0)
            elif key == glfw.KEY_ESCAPE and action == glfw.PRESS:
                glfw.set_window_should_close(callback_window, True)

        glfw.set_key_callback(window, on_key)
        if benchmark:
            return _run_benchmark(
                glfw,
                context,
                renderer,
                window,
                packet,
                warmup_frames=warmup_frames,
                measured_frames=benchmark_frames,
                light_speed=light_speed,
                data_prep_ms=data_prep_ms,
                context_setup_ms=context_setup_ms,
                renderer_setup_ms=renderer_setup_ms,
                vsync_on=vsync_on,
                input_name=input_name,
                prepare_frame=prepare_frame,
            )

        print("Controls: 1 RGB, 2 depth, 3 normals | A/D X, W/S Y, Q/E Z | Esc quit")
        previous_time = time.perf_counter()
        fps_start = previous_time
        fps_frames = 0
        fps = 0.0
        frames = 0

        while not glfw.window_should_close(window):
            glfw.poll_events()
            now = time.perf_counter()
            delta_s = min(now - previous_time, 0.1)
            previous_time = now
            light = packet.lights.lights[0]
            _move_light(glfw, window, light, delta_s, light_speed)

            if prepare_frame is not None:
                prepare_frame(packet, frames + 1)

            renderer.render(packet, state["mode"])
            glfw.swap_buffers(window)
            frames += 1
            fps_frames += 1

            if now - fps_start >= 0.5:
                fps = fps_frames / (now - fps_start)
                fps_frames = 0
                fps_start = now
                position = light.position_camera_m
                label = f"FPS {fps:5.1f} | Mode {state['mode'].name} | Light X {position[0]:+.2f} Y {position[1]:+.2f} Z {position[2]:+.2f} m"
                glfw.set_window_title(window, label)
                sys.stdout.write("\r" + label + "   ")
                sys.stdout.flush()
                status_line_open = True

            if max_frames is not None and frames >= max_frames:
                glfw.set_window_should_close(window, True)

        if status_line_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return 0
    except Exception as exc:
        if status_line_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
        print(f"Renderer/OpenGL error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if renderer is not None:
            renderer.release()
        if webcam is not None:
            webcam.release()
            print(f"Webcam index {camera_index} released after {webcam.frames_received} received frames.")
        if context is not None:
            context.release()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--frames", type=int, default=None, help="exit after this many frames (for smoke testing)")
    parser.add_argument("--benchmark", action="store_true", help="run a warmup plus a measured benchmark")
    parser.add_argument("--benchmark-frames", type=int, default=600, help="measured frames; benchmark requires at least 600")
    parser.add_argument("--warmup-frames", type=int, default=60, help="ignored warmup frames; benchmark requires at least 60")
    parser.add_argument("--vsync", choices=("on", "off"), default="on", help="buffer-swap VSync mode")
    parser.add_argument("--light-speed", type=float, default=0.75, help="light movement speed in meters/second")
    parser.add_argument("--input", choices=("synthetic", "webcam"), default="synthetic", help="RGB input source")
    parser.add_argument("--camera-index", type=int, default=0, help="webcam device index; default is 0")
    args = parser.parse_args(argv)
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    if args.benchmark and args.frames is not None:
        parser.error("use --benchmark-frames instead of --frames with --benchmark")
    if args.benchmark and args.benchmark_frames < 600:
        parser.error("--benchmark-frames must be at least 600")
    if args.benchmark and args.warmup_frames < 60:
        parser.error("--warmup-frames must be at least 60")
    if args.width <= 0 or args.height <= 0 or args.light_speed < 0.0:
        parser.error("width/height must be positive and light-speed must be non-negative")
    if args.camera_index < 0:
        parser.error("camera-index must be non-negative")
    return run_demo(
        args.width,
        args.height,
        args.frames,
        args.light_speed,
        benchmark=args.benchmark,
        benchmark_frames=args.benchmark_frames,
        warmup_frames=args.warmup_frames,
        vsync_on=args.vsync == "on",
        input_name=args.input,
        camera_index=args.camera_index,
    )


if __name__ == "__main__":
    raise SystemExit(main())
