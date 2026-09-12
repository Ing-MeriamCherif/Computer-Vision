"""Interactive renderer demo with synthetic or webcam RGB and mock geometry."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contracts.render_types import DepthFrame, Light, LightState, NormalFrame, RenderPacket
from renderer.capture import CapturedRGBFrame, WebcamCaptureWorker
from renderer.config import LightingConfig, ShadowConfig
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


def letterbox_rgb_to_rgb(frame_rgb: np.ndarray, destination_rgb: np.ndarray, cv2) -> tuple[int, int, int, int]:
    """Aspect-fit an RGB camera frame into a reusable RGB canvas."""
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3 or frame_rgb.dtype != np.uint8:
        raise ValueError(f"camera RGB frame must be HxWx3 uint8, got {frame_rgb.shape} {frame_rgb.dtype}")
    if destination_rgb.ndim != 3 or destination_rgb.shape[2] != 3 or destination_rgb.dtype != np.uint8:
        raise ValueError("destination must be an HxWx3 uint8 RGB array")

    source_h, source_w = frame_rgb.shape[:2]
    target_h, target_w = destination_rgb.shape[:2]
    scale = min(target_w / source_w, target_h / source_h)
    scaled_w = max(1, min(target_w, int(round(source_w * scale))))
    scaled_h = max(1, min(target_h, int(round(source_h * scale))))
    x0 = (target_w - scaled_w) // 2
    y0 = (target_h - scaled_h) // 2

    resized_rgb = cv2.resize(frame_rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    destination_rgb.fill(0)
    destination_rgb[y0:y0 + scaled_h, x0:x0 + scaled_w] = resized_rgb
    return x0, y0, scaled_w, scaled_h


class WebcamPacketUpdater:
    """Copy only newly captured RGB frames into the stable RenderPacket canvas."""

    def __init__(self, worker: WebcamCaptureWorker) -> None:
        self.worker = worker
        self.current_frame: CapturedRGBFrame | None = None

    def update(self, packet: RenderPacket, _render_index: int) -> tuple[float, bool]:
        start_ns = time.perf_counter_ns()
        self.worker.raise_if_failed()
        frame = self.worker.frames.get_latest()
        changed = frame is not None and (self.current_frame is None or frame.sequence != self.current_frame.sequence)
        if changed:
            letterbox_rgb_to_rgb(frame.rgb, packet.rgb, self.worker.cv2)
            packet.frame_id = frame.sequence
            packet.timestamp_s = frame.captured_at_s
            packet.depth.timestamp_s = frame.captured_at_s
            packet.normals.timestamp_s = frame.captured_at_s
            packet.lights.timestamp_s = frame.captured_at_s
            self.current_frame = frame
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        return elapsed_ms, bool(changed)


def make_synthetic_packet(
    width: int = 960,
    height: int = 540,
    *,
    foreground_bounds: tuple[float, float, float, float] | None = (0.30, 0.25, 0.70, 0.75),
) -> RenderPacket:
    """Build deterministic RGB/depth/normals with documented pinhole intrinsics.

    Synthetic calibration assumes a roughly 58-degree horizontal field of view
    and square pixels: fx = fy = 0.9 * width. At 960x540 this gives 864 px,
    with the principal point at (479.5, 269.5).
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    checker = (((np.arange(width)[None, :] // 32) + (np.arange(height)[:, None] // 32)) % 2).astype(np.uint8)

    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip(35.0 + 150.0 * x + 25.0 * checker, 0.0, 255.0).astype(np.uint8)
    rgb[:, :, 1] = np.clip(45.0 + 145.0 * y + 20.0 * (1.0 - x), 0.0, 255.0).astype(np.uint8)
    rgb[:, :, 2] = np.clip(105.0 + 95.0 * (1.0 - y) + 30.0 * checker, 0.0, 255.0).astype(np.uint8)

    if foreground_bounds is None:
        foreground = np.zeros((height, width), dtype=np.bool_)
        bounds = None
    else:
        if len(foreground_bounds) != 4 or not all(np.isfinite(value) for value in foreground_bounds):
            raise ValueError("foreground_bounds must be finite normalized (left, top, right, bottom) bounds")
        left, top, right, bottom = foreground_bounds
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            raise ValueError("foreground_bounds must be ordered within [0, 1]")
        x0, x1 = int(width * left), int(width * right)
        y0, y1 = int(height * top), int(height * bottom)
        foreground = np.zeros((height, width), dtype=np.bool_)
        foreground[y0:y1, x0:x1] = True
        bounds = (x0, y0, x1, y1)
    depth_m = np.full((height, width), 3.0, dtype=np.float32)
    depth_m[foreground] = 1.5
    rgb[foreground, 0] = 225
    rgb[foreground, 1] = 105
    rgb[foreground, 2] = 45

    # Camera-facing surfaces point toward the camera (-Z); a small tilted patch
    # makes the normal debug mode show a second, distinct direction.
    normals = np.zeros((height, width, 3), dtype=np.float32)
    normals[:, :, 2] = -1.0
    if bounds is None:
        tilted = np.zeros((height, width), dtype=np.bool_)
    else:
        x0, y0, x1, y1 = bounds
        tilted = np.zeros((height, width), dtype=np.bool_)
        tilted[y0:y1, max(x0, int(width * 0.40)):min(x1, int(width * 0.52))] = True
    normals[tilted, 0] = 0.5
    normals[tilted, 2] = -np.sqrt(np.float32(0.75))
    valid_mask = np.ones((height, width), dtype=np.bool_)
    now = time.perf_counter()

    light = Light(
        # In front of the 1.5 m and 3.0 m surfaces so -Z-facing normals light.
        position_camera_m=np.array([0.0, 0.0, 0.75], dtype=np.float32),
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
            fy=float(width) * 0.9,
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


def _print_gl_info(context) -> str:
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
        print("WARNING: OpenGL is not using the target RTX 4060; benchmark is INVALID FOR TARGET PERFORMANCE.")
    try:
        import glfw
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor) if monitor else None
        if mode is not None:
            print(f"  Primary display refresh: {mode.refresh_rate} Hz")
    except Exception:
        pass
    return renderer_name


_MODE_NAMES = {
    "rgb": DebugMode.RGB,
    "depth": DebugMode.DEPTH,
    "normals": DebugMode.NORMALS,
    "lambertian": DebugMode.LAMBERTIAN,
    "specular": DebugMode.SPECULAR,
    "final": DebugMode.FINAL,
    "shadow-mask": DebugMode.SHADOW_MASK,
    "shadow-final": DebugMode.SHADOW_FINAL,
}


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
    prepare_frame: Callable[[RenderPacket, int], tuple[float, bool]] | None,
    benchmark_mode: DebugMode,
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
    if benchmark_mode == DebugMode.SHADOW_FINAL:
        gpu_stages = ("lighting", "shadow", "composition")
    elif benchmark_mode in (DebugMode.LAMBERTIAN, DebugMode.SPECULAR, DebugMode.FINAL):
        gpu_stages = ("lighting",)
    else:
        gpu_stages = ()
    gpu_queries: dict[int, dict[str, object]] = {}
    gpu_query_error = None
    try:
        # Allocate per-stage queries before measurement and read them only afterward.
        gpu_queries = {
            index: {stage: context.query(time=True) for stage in gpu_stages}
            for index in gpu_query_indices
        }
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

        prep_ms = 0.0
        rgb_updated = True
        if prepare_frame is not None:
            prep_ms, rgb_updated = prepare_frame(packet, frame_index)

        upload_start_ns = time.perf_counter_ns()
        if input_name == "webcam":
            if rgb_updated:
                renderer.upload_rgb(packet.rgb)
        else:
            renderer.upload_packet(packet)
        upload_end_ns = time.perf_counter_ns()

        render_start_ns = time.perf_counter_ns()
        measured_index = frame_index - warmup_frames
        # Results are read only after the run, so query retrieval cannot stall measured frames.
        renderer.render(
            packet,
            benchmark_mode,
            upload_inputs=False,
            gpu_queries=gpu_queries.get(measured_index),
        )
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
    print(f"{input_name.title()} {benchmark_mode.name} benchmark: VSync {'ON' if vsync_on else 'OFF'} | {warmup_frames} warmup frames ignored | {measured_frames} measured frames | {packet.rgb.shape[1]}x{packet.rgb.shape[0]}")
    if benchmark_mode in (DebugMode.SHADOW_MASK, DebugMode.SHADOW_FINAL):
        print(
            f"Shadow pass: {renderer.resources.shadow_size[0]}x{renderer.resources.shadow_size[1]} | "
            f"steps {renderer.shadow_config.shadow_steps} | enabled {renderer.shadow_config.shadow_enabled} | "
            f"bias {renderer.shadow_config.shadow_bias_m:.3f} m | "
            f"thickness {renderer.shadow_config.shadow_thickness_m:.3f} m | "
            f"start offset {renderer.shadow_config.ray_start_offset:.3f} m"
        )
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
        print(f"Average webcam latest-frame fetch/letterbox time: {float(np.mean(frame_prep_ms)):.3f} ms/frame")
    print(f"Average render CPU-submit time: {float(np.mean(render_submit_ms)):.3f} ms/frame")
    print(f"Average buffer-swap time: {float(np.mean(swap_ms)):.3f} ms/frame")
    if gpu_queries:
        try:
            context.finish()
            for stage in gpu_stages:
                stage_ms = np.array(
                    [query_set[stage].elapsed / 1_000_000.0 for query_set in gpu_queries.values()],
                    dtype=np.float64,
                )
                if stage_ms.size:
                    print(
                        f"GPU {stage} time (sampled draws): avg {float(np.mean(stage_ms)):.4f} ms, "
                        f"median {float(np.median(stage_ms)):.4f} ms, "
                        f"p95 {float(np.percentile(stage_ms, 95)):.4f} ms, n={stage_ms.size}"
                    )
        except Exception as exc:
            print(f"GPU render timer query unavailable: {type(exc).__name__}: {exc}")
    elif gpu_query_error:
        print(f"GPU render timer query unavailable: {gpu_query_error}")
    print("Render CPU-submit time does not force GPU completion; buffer swap is timed separately.")
    return 0


def _run_webcam_live(
    glfw,
    context,
    renderer: Renderer,
    window,
    packet: RenderPacket,
    worker: WebcamCaptureWorker,
    updater: WebcamPacketUpdater,
    *,
    light_speed: float,
    max_frames: int | None,
    duration_seconds: float | None,
    gpu_name: str,
    state: dict,
) -> int:
    """Render latest webcam RGB while a worker continuously drains the camera."""
    print("Controls: 1 RGB, 2 depth, 3 normals, 4 Lambertian, 5 specular, 6 final, 7 shadow mask, 8 final + shadows | A/D X, W/S Y, Q/E Z | Esc quit")

    gpu_queries = []
    gpu_query_error = None
    try:
        gpu_queries = [context.query(time=True) for _ in range(40)]
    except Exception as exc:  # Some OpenGL drivers do not expose timer queries.
        gpu_queries = []
        gpu_query_error = f"{type(exc).__name__}: {exc}"

    measurement_start = time.perf_counter()
    previous_time = measurement_start
    next_report_time = measurement_start + 1.0
    next_query_time = measurement_start
    query_interval_s = max(0.25, (duration_seconds or 20.0) / max(1, len(gpu_queries)))
    start_camera_stats = worker.frames.stats()
    previous_camera_stats = start_camera_stats
    previous_report_time = measurement_start
    last_presented_sequence = 0
    rendered_frames = 0
    presented_camera_frames = 0
    presented_camera_frames_total = 0
    webcam_packet_prep_total_ms = 0.0
    webcam_packet_prep_count = 0
    upload_total_ms = 0.0
    upload_count = 0
    render_submit_total_ms = 0.0
    swap_total_ms = 0.0
    full_frame_total_ms = 0.0
    presentation_latency_total_ms = 0.0
    presentation_latency_max_ms = 0.0
    presentation_latency_count = 0
    frame_count = 0
    query_index = 0

    while not glfw.window_should_close(window):
        worker.raise_if_failed()
        frame_start_ns = time.perf_counter_ns()
        glfw.poll_events()
        now = time.perf_counter()
        delta_s = min(now - previous_time, 0.1)
        previous_time = now
        _move_light(glfw, window, packet.lights.lights[0], delta_s, light_speed)

        prep_ms, rgb_updated = updater.update(packet, rendered_frames)
        webcam_packet_prep_total_ms += prep_ms
        webcam_packet_prep_count += 1
        if rgb_updated:
            upload_start_ns = time.perf_counter_ns()
            renderer.upload_rgb(packet.rgb)
            upload_total_ms += (time.perf_counter_ns() - upload_start_ns) / 1_000_000.0
            upload_count += 1

        render_start_ns = time.perf_counter_ns()
        query = None
        if query_index < len(gpu_queries) and now >= next_query_time:
            query = gpu_queries[query_index]
            query_index += 1
            next_query_time += query_interval_s
        if query is None:
            renderer.render(packet, state["mode"], upload_inputs=False)
        else:
            with query:
                renderer.render(packet, state["mode"], upload_inputs=False)
        render_submit_total_ms += (time.perf_counter_ns() - render_start_ns) / 1_000_000.0

        swap_start_ns = time.perf_counter_ns()
        glfw.swap_buffers(window)
        frame_end_ns = time.perf_counter_ns()
        swap_total_ms += (frame_end_ns - swap_start_ns) / 1_000_000.0
        full_frame_ms = (frame_end_ns - frame_start_ns) / 1_000_000.0
        full_frame_total_ms += full_frame_ms
        rendered_frames += 1
        frame_count += 1

        frame = updater.current_frame
        if frame is not None and frame.sequence != last_presented_sequence:
            presented_camera_frames += 1
            presented_camera_frames_total += 1
            last_presented_sequence = frame.sequence
            age_ms = max(0.0, (time.perf_counter() - frame.captured_at_s) * 1000.0)
            presentation_latency_total_ms += age_ms
            presentation_latency_max_ms = max(presentation_latency_max_ms, age_ms)
            presentation_latency_count += 1

        end_time = time.perf_counter()
        if end_time >= next_report_time:
            camera_stats = worker.frames.stats()
            interval_s = max(end_time - previous_report_time, 1e-9)
            captured_interval = camera_stats.frames_captured - previous_camera_stats.frames_captured
            replaced_interval = camera_stats.frames_replaced - previous_camera_stats.frames_replaced
            unique_interval = presented_camera_frames
            report_frames = frame_count
            report_fps = report_frames / interval_s
            unique_fps = unique_interval / interval_s
            latest_age_ms = (
                max(0.0, (end_time - camera_stats.latest_capture_time_s) * 1000.0)
                if camera_stats.latest_capture_time_s is not None
                else float("nan")
            )
            label = (
                f"Renderer {report_fps:5.1f} FPS | capture {captured_interval / interval_s:4.1f} FPS "
                f"({captured_interval} new, {replaced_interval} replaced) | unique {unique_fps:4.1f}/s "
                f"| latest age {latest_age_ms:5.1f} ms | {state['mode'].name} | {gpu_name}"
            )
            glfw.set_window_title(window, label)
            sys.stdout.write("\r" + label + "   ")
            sys.stdout.flush()
            previous_camera_stats = camera_stats
            previous_report_time = end_time
            frame_count = 0
            presented_camera_frames = 0
            next_report_time = end_time + 1.0

        if duration_seconds is not None and end_time - measurement_start >= duration_seconds:
            glfw.set_window_should_close(window, True)
        if max_frames is not None and rendered_frames >= max_frames:
            glfw.set_window_should_close(window, True)

    if rendered_frames == 0:
        raise RuntimeError("webcam renderer exited before presenting any frames")
    sys.stdout.write("\n")
    sys.stdout.flush()

    measurement_end = time.perf_counter()
    elapsed_s = max(measurement_end - measurement_start, 1e-9)
    end_camera_stats = worker.frames.stats()
    captured = end_camera_stats.frames_captured - start_camera_stats.frames_captured
    replaced = end_camera_stats.frames_replaced - start_camera_stats.frames_replaced
    camera_fps = captured / elapsed_s
    loop_fps = rendered_frames / elapsed_s
    unique_fps = presented_camera_frames_total / elapsed_s

    gpu_draw_ms = None
    if gpu_queries:
        try:
            # Synchronize only after the live measurement so timer results never stall frames.
            context.finish()
            gpu_draw_ms = np.asarray(
                [query.elapsed / 1_000_000.0 for query in gpu_queries[:query_index]], dtype=np.float64
            )
        except Exception as exc:
            gpu_query_error = f"{type(exc).__name__}: {exc}"
        finally:
            # ModernGL Query objects have no explicit release(); dropping the
            # references lets their wrappers release them with the context.
            gpu_queries.clear()

    print("Webcam live metrics:")
    print(f"  Measurement duration: {elapsed_s:.2f} s")
    print(f"  Camera-reported FPS: {worker.reported_fps:.2f}" if worker.reported_fps > 0 else "  Camera-reported FPS: unavailable")
    print(f"  Actual capture FPS: {camera_fps:.2f} ({captured} frames captured)")
    print(f"  Frames replaced before renderer consumption: {replaced}")
    print(f"  Renderer loop FPS: {loop_fps:.2f} ({rendered_frames} frames)")
    print(f"  Unique camera updates presented per second: {unique_fps:.2f}")
    print(f"  Latest-frame fetch/letterbox CPU time: {webcam_packet_prep_total_ms / webcam_packet_prep_count:.3f} ms/render")
    print(f"  RGB texture upload CPU time: {upload_total_ms / max(upload_count, 1):.3f} ms/update ({upload_count} updates)")
    print(f"  Render submit CPU time: {render_submit_total_ms / rendered_frames:.3f} ms/frame")
    if gpu_draw_ms is not None and gpu_draw_ms.size:
        print(
            f"  GPU draw time: avg {float(np.mean(gpu_draw_ms)):.3f} ms, "
            f"p95 {float(np.percentile(gpu_draw_ms, 95)):.3f} ms (n={gpu_draw_ms.size})"
        )
    elif gpu_query_error:
        print(f"  GPU draw timer query unavailable: {gpu_query_error}")
    print(f"  Buffer swap time: {swap_total_ms / rendered_frames:.3f} ms/frame")
    print(f"  Full renderer loop time: {full_frame_total_ms / rendered_frames:.3f} ms/frame average")
    if presentation_latency_count:
        print(
            f"  Presented camera-frame age: avg {presentation_latency_total_ms / presentation_latency_count:.2f} ms, "
            f"max {presentation_latency_max_ms:.2f} ms"
        )
    print("  Backlog: one latest-frame slot only; stale queued frames are discarded by replacement.")
    print(f"  OpenGL GPU: {gpu_name}")
    return 0


def run_demo(
    width: int,
    height: int,
    max_frames: int | None,
    duration_seconds: float | None,
    light_speed: float,
    *,
    benchmark: bool,
    benchmark_frames: int,
    warmup_frames: int,
    vsync_on: bool,
    input_name: str,
    camera_index: int,
    initial_mode: DebugMode,
    lighting_config: LightingConfig,
    shadow_config: ShadowConfig,
    benchmark_mode: DebugMode,
    foreground_bounds: tuple[float, float, float, float] | None,
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
    webcam_updater = None
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
        packet = make_synthetic_packet(
            width=width,
            height=height,
            foreground_bounds=foreground_bounds,
        )
        data_prep_ms = (time.perf_counter_ns() - data_prep_start_ns) / 1_000_000.0
        prepare_frame = None
        if input_name == "webcam":
            webcam = WebcamCaptureWorker(camera_index).start()
            webcam_updater = WebcamPacketUpdater(webcam)
            webcam_updater.update(packet, 0)
            prepare_frame = webcam_updater.update
            reported_fps = f"{webcam.reported_fps:.2f}" if webcam.reported_fps > 0.0 else "unavailable"
            print(f"Webcam index: {camera_index}")
            print(f"Webcam reported resolution: {webcam.reported_width}x{webcam.reported_height}")
            print(f"Webcam reported FPS: {reported_fps}")
            print(f"Webcam received frame size: {webcam.received_width}x{webcam.received_height}")
            print("WEBCAM RGB + MOCK GEOMETRY: real webcam RGB with synthetic depth/normals; geometry does not describe the camera scene.")
            print("Webcam display: aspect-preserving letterbox, RGB latest-frame mailbox, local only; no saving or network transmission")

        renderer_setup_start_ns = time.perf_counter_ns()
        renderer = Renderer(context, packet, config=lighting_config, shadow_config=shadow_config)
        renderer_setup_ms = (time.perf_counter_ns() - renderer_setup_start_ns) / 1_000_000.0
        gpu_name = _print_gl_info(context)
        state = {"mode": initial_mode}

        def on_key(callback_window, key, _scancode, action, _mods):
            if action not in (glfw.PRESS, glfw.REPEAT):
                return
            if key in (
                glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4,
                glfw.KEY_5, glfw.KEY_6, glfw.KEY_7, glfw.KEY_8,
            ):
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
                benchmark_mode=benchmark_mode,
            )

        if input_name == "webcam":
            return _run_webcam_live(
                glfw,
                context,
                renderer,
                window,
                packet,
                webcam,
                webcam_updater,
                light_speed=light_speed,
                max_frames=max_frames,
                duration_seconds=duration_seconds,
                gpu_name=gpu_name,
                state=state,
            )

        print("Controls: 1 RGB, 2 depth, 3 normals, 4 Lambertian, 5 specular, 6 final, 7 shadow mask, 8 final + shadows | A/D X, W/S Y, Q/E Z | Esc quit")
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
                label = f"FPS {fps:5.1f} | {state['mode'].name} | GPU {gpu_name} | Light {position[0]:+.2f},{position[1]:+.2f},{position[2]:+.2f} m"
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
        if webcam is not None:
            try:
                webcam.stop()
            except Exception as exc:
                print(f"Webcam shutdown error: {type(exc).__name__}: {exc}", file=sys.stderr)
            stats = webcam.frames.stats()
            print(
                f"Webcam index {camera_index} capture thread stopped; VideoCapture released={webcam.released}; "
                f"{stats.frames_captured} frames captured."
            )
        if renderer is not None:
            renderer.release()
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
    parser.add_argument("--duration-seconds", type=float, default=None, help="stop a live webcam test after this many seconds")
    parser.add_argument("--benchmark", action="store_true", help="run a warmup plus a measured benchmark")
    parser.add_argument("--benchmark-frames", type=int, default=600, help="measured frames; benchmark requires at least 600")
    parser.add_argument("--warmup-frames", type=int, default=60, help="ignored warmup frames; benchmark requires at least 60")
    parser.add_argument("--vsync", choices=("on", "off"), default="on", help="buffer-swap VSync mode")
    parser.add_argument("--light-speed", type=float, default=0.75, help="light movement speed in meters/second")
    parser.add_argument("--input", choices=("synthetic", "webcam"), default="synthetic", help="RGB input source")
    parser.add_argument("--camera-index", type=int, default=0, help="webcam device index; default is 0")
    parser.add_argument("--mode", choices=tuple(_MODE_NAMES), default="rgb", help="initial interactive visualization")
    parser.add_argument(
        "--benchmark-mode",
        choices=("rgb", "lambertian", "specular", "final", "shadow-mask", "shadow-final"),
        default="rgb",
        help="visualization measured by --benchmark",
    )
    parser.add_argument("--ambient-strength", type=float, default=0.15)
    parser.add_argument("--specular-strength", type=float, default=0.20)
    parser.add_argument("--shininess", type=float, default=48.0)
    parser.add_argument("--attenuation-k", type=float, default=0.6)
    parser.add_argument("--shadow-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shadow-resolution-scale", type=float, default=0.5)
    parser.add_argument("--shadow-steps", type=int, default=12)
    parser.add_argument("--shadow-bias-m", type=float, default=0.015)
    parser.add_argument("--shadow-thickness-m", type=float, default=0.15)
    parser.add_argument("--ray-start-offset", type=float, default=0.01)
    parser.add_argument(
        "--foreground-bounds",
        type=float,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        default=(0.30, 0.25, 0.70, 0.75),
        help="normalized synthetic occluder bounds; use --no-occluder to remove it",
    )
    parser.add_argument("--no-occluder", action="store_true", help="omit the synthetic foreground rectangle")
    args = parser.parse_args(argv)
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    if args.duration_seconds is not None and (not np.isfinite(args.duration_seconds) or args.duration_seconds <= 0.0):
        parser.error("--duration-seconds must be finite and positive")
    if args.duration_seconds is not None and args.input != "webcam":
        parser.error("--duration-seconds is only supported with --input webcam")
    if args.duration_seconds is not None and args.benchmark:
        parser.error("use --frames or --duration-seconds without --benchmark")
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
    try:
        lighting_config = LightingConfig(
            ambient_strength=args.ambient_strength,
            specular_strength=args.specular_strength,
            shininess=args.shininess,
            attenuation_k=args.attenuation_k,
        )
        shadow_config = ShadowConfig(
            shadow_enabled=args.shadow_enabled,
            shadow_resolution_scale=args.shadow_resolution_scale,
            shadow_steps=args.shadow_steps,
            shadow_bias_m=args.shadow_bias_m,
            shadow_thickness_m=args.shadow_thickness_m,
            ray_start_offset=args.ray_start_offset,
        )
    except ValueError as exc:
        parser.error(str(exc))
    foreground_bounds = None if args.no_occluder else tuple(args.foreground_bounds)
    if foreground_bounds is not None:
        left, top, right, bottom = foreground_bounds
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            parser.error("--foreground-bounds must be ordered values within [0, 1]")
    return run_demo(
        args.width,
        args.height,
        args.frames,
        args.duration_seconds,
        args.light_speed,
        benchmark=args.benchmark,
        benchmark_frames=args.benchmark_frames,
        warmup_frames=args.warmup_frames,
        vsync_on=args.vsync == "on",
        input_name=args.input,
        camera_index=args.camera_index,
        initial_mode=_MODE_NAMES[args.mode],
        lighting_config=lighting_config,
        shadow_config=shadow_config,
        benchmark_mode=_MODE_NAMES[args.benchmark_mode],
        foreground_bounds=foreground_bounds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
