"""Interactive P0/P1 renderer demo using generated RGB, depth, and normals."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contracts.render_types import DepthFrame, Light, LightState, NormalFrame, RenderPacket
from renderer.renderer import DebugMode, Renderer


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


def run_demo(width: int, height: int, max_frames: int | None, light_speed: float) -> int:
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
    try:
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        window = glfw.create_window(width, height, "Person 4 Synthetic Renderer", None, None)
        if not window:
            print(f"GLFW window creation failed for {width}x{height} OpenGL 3.3.", file=sys.stderr)
            return 1

        glfw.make_context_current(window)
        glfw.swap_interval(1)
        context = moderngl.create_context(require=330)
        packet = make_synthetic_packet(width=width, height=height)
        renderer = Renderer(context, packet)
        state = {"mode": DebugMode.RGB}

        def on_key(callback_window, key, _scancode, action, _mods):
            if action not in (glfw.PRESS, glfw.REPEAT):
                return
            if key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3):
                state["mode"] = DebugMode(key - glfw.KEY_0)
            elif key == glfw.KEY_ESCAPE and action == glfw.PRESS:
                glfw.set_window_should_close(callback_window, True)

        glfw.set_key_callback(window, on_key)
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
    parser.add_argument("--light-speed", type=float, default=0.75, help="light movement speed in meters/second")
    args = parser.parse_args(argv)
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    if args.width <= 0 or args.height <= 0 or args.light_speed < 0.0:
        parser.error("width/height must be positive and light-speed must be non-negative")
    return run_demo(args.width, args.height, args.frames, args.light_speed)


if __name__ == "__main__":
    raise SystemExit(main())
