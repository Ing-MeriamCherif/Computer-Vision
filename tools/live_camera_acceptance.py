#!/usr/bin/env python3
"""Continuous camera soak tool (physical camera default, optional --synthetic offline mode).

Runs the full live pipeline for a configurable duration and records telemetry:
FPS, latency, worker health, and RSS memory growth.

NOTE: For authoritative fail-closed physical hardware validation with zero
synthetic fallbacks, use `python -m tools.physical_camera_gate`.

Usage:
    python -m tools.live_camera_acceptance --duration 60
    python -m tools.live_camera_acceptance --soak --duration 600 --min-render-fps 28
    python -m tools.live_camera_acceptance --synthetic --duration 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import threading
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NRW Live Camera Acceptance Test"
    )
    p.add_argument("--camera", default=0, help="Camera device (default: 0)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--duration", type=float, default=30.0,
                   help="Test duration in seconds (default: 30)")
    p.add_argument("--frames", type=int, default=None,
                   help="Stop after N rendered frames")
    p.add_argument("--quality", choices=["low", "balanced", "high"], default="balanced")
    p.add_argument("--mode", type=int, default=8, choices=range(1, 10))
    p.add_argument("--calibration", default=None)
    p.add_argument("--json-report", default=None,
                   help="Write JSON report to path")
    p.add_argument("--soak", action="store_true",
                   help="Strict soak mode: fail on any metric regression")
    p.add_argument("--min-render-fps", type=float, default=20.0,
                   help="Minimum acceptable render FPS (default: 20)")
    p.add_argument("--max-p95-latency-ms", type=float, default=150.0,
                   help="Maximum p95 render latency in ms (default: 150)")
    p.add_argument("--max-rss-growth-mb", type=float, default=200.0,
                   help="Maximum RSS growth allowed (MB, default: 200)")
    p.add_argument("--no-persistent", action="store_true",
                   help="Disable persistent geometry sidecar")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic camera (offline testing only)")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-second metrics during soak")
    return p.parse_args()


def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (pct / 100.0) * (len(s) - 1)
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def _gpu_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1048576.0
    except Exception:
        pass
    return 0.0


def main() -> int:
    args = parse_args()

    from geometry.native_app import AppMode, NativeLiveApp, QualityProfile

    quality_map = {"low": QualityProfile.LOW, "balanced": QualityProfile.BALANCED, "high": QualityProfile.HIGH}

    try:
        camera_device = int(args.camera)
    except (ValueError, TypeError):
        camera_device = str(args.camera)

    print("═══════════════════════════════════════════════════════")
    print("  NRW LIVE CAMERA ACCEPTANCE TEST")
    print("═══════════════════════════════════════════════════════")
    print(f"  Camera     : {camera_device} ({args.width}×{args.height} @ {args.fps} FPS)")
    print(f"  Duration   : {args.duration:.0f}s")
    print(f"  Quality    : {args.quality.upper()}")
    print(f"  Min FPS    : {args.min_render_fps}")
    print(f"  Max p95    : {args.max_p95_latency_ms} ms")
    print(f"  Max RSS Δ  : {args.max_rss_growth_mb} MB")
    print(f"  Soak mode  : {args.soak}")
    print("═══════════════════════════════════════════════════════")

    app = NativeLiveApp(
        camera_device=camera_device,
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=args.fps,
        quality_profile=quality_map[args.quality],
        initial_mode=AppMode(args.mode),
        headless=True,               # acceptance runs headless to avoid display deps
        use_synthetic_camera=args.synthetic,
    )

    # Metrics collectors
    render_fps_samples: list[float] = []
    render_latency_samples: list[float] = []
    capture_depth_age_samples: list[float] = []
    worker_failures: list[str] = []
    hand_stale_events: int = 0
    depth_stale_events: int = 0
    depth_none_events: int = 0

    rss_start = _rss_mb()
    gpu_start = _gpu_mb()
    rss_peak = rss_start

    app.start()

    # Load calibration if provided
    if args.calibration and Path(args.calibration).exists():
        from geometry import CameraModel
        try:
            model = CameraModel.load_json(args.calibration)
            if model.width != app.camera_width or model.height != app.camera_height:
                model = model.scaled_intrinsics(app.camera_width, app.camera_height)
            app.camera = model
            print(f"  Calibration  : {args.calibration}")
        except Exception as exc:
            print(f"  Calibration WARN: {exc}")

    test_start = time.monotonic()
    last_print_ts = test_start
    frames_rendered = 0
    errors = []

    print("  Running acceptance…  (press Ctrl-C to abort)")
    print()

    try:
        while True:
            elapsed = time.monotonic() - test_start
            if elapsed >= args.duration:
                break
            if args.frames is not None and frames_rendered >= args.frames:
                break

            frame = app.step()
            if frame is None:
                time.sleep(0.005)
                continue

            frames_rendered += 1
            render_fps_samples.append(app.render_fps)
            render_latency_samples.append(app.render_latency_ms)

            rss_now = _rss_mb()
            if rss_now > rss_peak:
                rss_peak = rss_now

            # Check worker health
            if app.depth_worker and app.depth_worker.last_error:
                if not worker_failures or worker_failures[-1] != app.depth_worker.last_error:
                    worker_failures.append(f"depth: {app.depth_worker.last_error}")

            # Print heartbeat every second
            if time.monotonic() - last_print_ts >= 1.0:
                cam_fps = app.camera_worker.actual_fps if app.camera_worker else 0
                drops = app.camera_worker.dropped_frames if app.camera_worker else 0
                depth_lat = app.depth_worker.last_inference_ms if app.depth_worker else 0
                hand_fps = app.hand_worker.tracking_fps if app.hand_worker else 0
                surfel_n = app.persistent_worker.surfel_count if app.persistent_worker else 0
                pct = elapsed / args.duration * 100
                print(
                    f"  [{pct:5.1f}%  {elapsed:5.0f}s]  "
                    f"Render={app.render_fps:5.1f}FPS  lat={app.render_latency_ms:4.1f}ms  "
                    f"Camera={cam_fps:.0f}FPS  drops={drops}  "
                    f"Depth={depth_lat:.0f}ms  Hand={hand_fps:.0f}Hz  "
                    f"Surfels={surfel_n}  RSS={rss_now:.0f}MB"
                )
                last_print_ts = time.monotonic()

    except KeyboardInterrupt:
        print("\n  Aborted by user.")

    app.stop()

    # Compute final metrics
    actual_duration = time.monotonic() - test_start
    gpu_end = _gpu_mb()
    rss_end = _rss_mb()

    fps_mean = sum(render_fps_samples) / max(len(render_fps_samples), 1)
    fps_p5 = _percentile(render_fps_samples, 5)
    fps_p50 = _percentile(render_fps_samples, 50)

    lat_p50 = _percentile(render_latency_samples, 50)
    lat_p95 = _percentile(render_latency_samples, 95)
    lat_p99 = _percentile(render_latency_samples, 99)

    rss_growth = rss_end - rss_start

    # Evaluate pass/fail
    fail_reasons = []

    if fps_p5 < args.min_render_fps:
        fail_reasons.append(f"Render FPS p5={fps_p5:.1f} < {args.min_render_fps}")

    if lat_p95 > args.max_p95_latency_ms:
        fail_reasons.append(f"Render latency p95={lat_p95:.1f}ms > {args.max_p95_latency_ms}ms")

    if rss_growth > args.max_rss_growth_mb:
        fail_reasons.append(f"RSS growth={rss_growth:.1f} MB > {args.max_rss_growth_mb} MB")

    if args.soak and worker_failures:
        fail_reasons.append(f"Worker failures: {len(worker_failures)}")

    passed = len(fail_reasons) == 0

    # Print results
    print()
    print("═══════════════════════════════════════════════════════")
    print("  ACCEPTANCE TEST RESULTS")
    print("═══════════════════════════════════════════════════════")
    print(f"  Camera       : {camera_device} ({app.camera_width}×{app.camera_height})")
    print(f"  Duration     : {actual_duration:.1f}s")
    print(f"  Frames       : {frames_rendered}")
    print(f"  Render FPS   : mean={fps_mean:.1f}  p5={fps_p5:.1f}  p50={fps_p50:.1f}")
    print(f"  Latency      : p50={lat_p50:.1f}ms  p95={lat_p95:.1f}ms  p99={lat_p99:.1f}ms")
    print(f"  RSS          : start={rss_start:.0f} end={rss_end:.0f} peak={rss_peak:.0f} growth={rss_growth:.1f} MB")
    print(f"  GPU alloc    : start={gpu_start:.0f} end={gpu_end:.0f} MB")
    print(f"  Worker fails : {len(worker_failures)}")
    if worker_failures:
        for f in worker_failures[:5]:
            print(f"    - {f}")
    print()

    if passed:
        print("  RESULT: ✓ PASS")
    else:
        print("  RESULT: ✗ FAIL")
        for reason in fail_reasons:
            print(f"    - {reason}")

    print("═══════════════════════════════════════════════════════")

    # Write JSON report
    report: dict[str, Any] = {
        "passed": passed,
        "duration_s": actual_duration,
        "frames_rendered": frames_rendered,
        "camera": {
            "device": str(camera_device),
            "width": app.camera_width,
            "height": app.camera_height,
            "synthetic": args.synthetic,
        },
        "render_fps": {"mean": fps_mean, "p5": fps_p5, "p50": fps_p50},
        "latency_ms": {"p50": lat_p50, "p95": lat_p95, "p99": lat_p99},
        "resources": {
            "rss_start_mb": rss_start,
            "rss_end_mb": rss_end,
            "rss_peak_mb": rss_peak,
            "rss_growth_mb": rss_growth,
            "gpu_start_mb": gpu_start,
            "gpu_end_mb": gpu_end,
        },
        "worker_failures": worker_failures,
        "fail_reasons": fail_reasons,
        "quality": args.quality,
    }

    if args.json_report:
        out_path = Path(args.json_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"  JSON report  : {out_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
