#!/usr/bin/env python3
"""CLI entry point for the native live competition application.

Usage:
    python -m tools.native_live_app               # full native live app
    python -m tools.native_live_app --camera 0
    python -m tools.native_live_app --quality high --fullscreen
    python -m tools.native_live_app --smoke        # headless smoke test
    python -m tools.native_live_app --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="native_live_app",
        description="NRW Native Live Computer-Vision Competition Application",
    )
    p.add_argument("--camera", default=0,
                   help="Camera device index or /dev/videoN path (default: 0)")
    p.add_argument("--width", type=int, default=640,
                   help="Requested camera width (default: 640)")
    p.add_argument("--height", type=int, default=480,
                   help="Requested camera height (default: 480)")
    p.add_argument("--fps", type=int, default=30,
                   help="Requested camera FPS (default: 30)")
    p.add_argument("--calibration", default=None,
                   help="Path to camera calibration JSON")
    p.add_argument("--depth-backend", choices=["local", "colleague"], default=None,
                   help="Depth provider override (default: from NRW_DEPTH_SOURCE env)")
    p.add_argument("--depth-model", default="models/depth-anything-v2-small",
                   help="Path to Depth Anything V2 model checkpoint directory")
    p.add_argument("--hand-backend", choices=["auto", "tasks", "legacy", "colleague"],
                   default="auto",
                   help="Hand tracking backend (default: auto)")
    p.add_argument("--quality", choices=["low", "balanced", "high"],
                   default="balanced",
                   help="Quality profile (default: balanced)")
    p.add_argument("--mode", type=int, choices=range(1, 10), default=8,
                   help="Initial display mode 1-9 (default: 8 multi-light)")
    p.add_argument("--fullscreen", action="store_true",
                   help="Launch in fullscreen mode")
    p.add_argument("--no-persistent", action="store_true",
                   help="Disable Phase 5 persistent geometry sidecar")
    p.add_argument("--headless", action="store_true",
                   help="Run without display window (headless verification only)")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic animated camera instead of physical camera")
    p.add_argument("--smoke", action="store_true",
                   help="Run headless smoke test (10 frames) and exit")
    p.add_argument("--frames", type=int, default=None,
                   help="Stop after N rendered frames (default: run forever)")
    p.add_argument("--timeout", type=float, default=None,
                   help="Stop after N seconds (default: run forever)")
    p.add_argument("--debug", action="store_true",
                   help="Enable verbose debug output")
    return p.parse_args()


def _load_calibration(path: str, width: int, height: int):
    """Load and scale camera calibration from JSON."""
    from geometry import CameraModel
    try:
        model = CameraModel.load_json(path)
        if model.width != width or model.height != height:
            model = model.scaled_intrinsics(width, height)
        return model
    except Exception as exc:
        print(f"[WARNING] Could not load calibration {path}: {exc}", file=sys.stderr)
        print("[WARNING]  Using approximate intrinsics (UNCALIBRATED)", file=sys.stderr)
        return None


def run_smoke_test(args: argparse.Namespace) -> int:
    """Run headless 10-frame smoke test. Returns exit code."""
    print("═══════════════════════════════════════════")
    print("  NRW Native Live App  —  SMOKE TEST")
    print("═══════════════════════════════════════════")
    from geometry.native_app import AppMode, NativeLiveApp, QualityProfile

    quality_map = {"low": QualityProfile.LOW, "balanced": QualityProfile.BALANCED, "high": QualityProfile.HIGH}
    app = NativeLiveApp(
        headless=True,
        use_synthetic_camera=True,
        quality_profile=quality_map[args.quality],
        initial_mode=AppMode.MULTILIGHT,
    )
    try:
        app.start()
        for i in range(10):
            frame = app.step()
            if frame is None:
                print(f"  FAIL: step {i} returned None")
                return 1
            print(f"  frame {i+1:2d}/10  shape={frame.shape}  fps={app.render_fps:.1f}")
        print()
        print("  Smoke test passed — 10 frames rendered successfully.")
        return 0
    except Exception as exc:
        print(f"  FAIL: {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return 1
    finally:
        app.stop()


def main() -> int:
    args = parse_args()

    if args.smoke:
        return run_smoke_test(args)

    # Override depth source from CLI if requested
    if args.depth_backend is not None:
        os.environ["NRW_DEPTH_SOURCE"] = args.depth_backend

    # Parse camera device (support int or string path)
    try:
        camera_device = int(args.camera)
    except (ValueError, TypeError):
        camera_device = str(args.camera)

    from geometry.native_app import AppMode, NativeLiveApp, QualityProfile

    quality_map = {"low": QualityProfile.LOW, "balanced": QualityProfile.BALANCED, "high": QualityProfile.HIGH}

    print("═══════════════════════════════════════════════════════")
    print("  NRW COMPUTER VISION — NATIVE LIVE APPLICATION")
    print("═══════════════════════════════════════════════════════")
    print(f"  Camera    : {camera_device}  ({args.width}×{args.height} @ {args.fps} FPS)")
    print(f"  Quality   : {args.quality.upper()}")
    print(f"  Mode      : {args.mode}")
    print(f"  Depth     : {args.depth_model}")
    print(f"  Fullscreen: {args.fullscreen}")
    print(f"  Headless  : {args.headless or args.synthetic}")
    print("───────────────────────────────────────────────────────")
    print("  Controls  : [1-9] Mode  [F] Fullscreen  [D] HUD")
    print("              [P] Profile  [V] Volumetrics  [S] Shadows")
    print("              [H] Skeleton  [R] Reset  [Q/ESC] Quit")
    print("═══════════════════════════════════════════════════════")

    app = NativeLiveApp(
        camera_device=camera_device,
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=args.fps,
        depth_model_path=args.depth_model,
        quality_profile=quality_map[args.quality],
        initial_mode=AppMode(args.mode),
        headless=args.headless,
        fullscreen=args.fullscreen,
        use_synthetic_camera=args.synthetic,
    )

    try:
        app.start()

        # Override calibration if provided
        if args.calibration and Path(args.calibration).exists():
            model = _load_calibration(args.calibration, app.camera_width, app.camera_height)
            if model is not None:
                app.camera = model
                print(f"  Calibration loaded: {args.calibration}")
            else:
                print("  INTRINSICS: APPROXIMATE (no calibration file)")
        else:
            print("  INTRINSICS: APPROXIMATE (run python tools/calibrate_camera.py to calibrate)")

        app.run(
            max_frames=args.frames,
            timeout_sec=args.timeout,
        )

        # Print final summary
        print()
        print("═══════════════════════════════════════════════════════")
        print("  SESSION SUMMARY")
        print("═══════════════════════════════════════════════════════")
        print(f"  Frames rendered : {app.frames_rendered}")
        print(f"  Render FPS      : {app.render_fps:.1f}")
        print(f"  Render latency  : {app.render_latency_ms:.1f} ms (last frame)")
        if app.camera_worker:
            print(f"  Camera FPS      : {app.camera_worker.actual_fps:.1f}")
            print(f"  Dropped frames  : {app.camera_worker.dropped_frames}")
        if app.depth_worker:
            print(f"  Depth latency   : {app.depth_worker.last_inference_ms:.1f} ms")
        if app.persistent_worker:
            print(f"  Surfels         : {app.persistent_worker.surfel_count}")
        print("═══════════════════════════════════════════════════════")

        return 0

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
        return 0
    except Exception as exc:
        print(f"\n  FATAL ERROR: {exc}", file=sys.stderr)
        if args.debug:
            import traceback; traceback.print_exc()
        return 1
    finally:
        app.stop()


if __name__ == "__main__":
    sys.exit(main())
