#!/usr/bin/env python3
"""Repeatable CPU benchmark for the Phase 1-4 geometry stages."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry import (  # noqa: E402
    CameraModel,
    DepthState,
    MotionState,
    NormalMode,
    TemporalGeometryEngine,
    align_inverse_depth,
    backproject_depth,
    estimate_normals,
    summarize_timings,
    warp_depth_backward,
    warp_field_backward,
)


def _run(fn, warmups: int, iterations: int) -> dict[str, float | int]:
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return {**summarize_timings(samples), "iterations": iterations, "warmups": warmups}


def benchmark(width: int, height: int, warmups: int, iterations: int) -> dict:
    camera = CameraModel(width, height, width * 0.9, height * 0.9, (width - 1) / 2, (height - 1) / 2)
    rng = np.random.default_rng(4)
    depth = (2.0 + 0.15 * np.sin(np.linspace(0, 8, width))[None, :] + rng.normal(0, 0.01, (height, width))).astype(np.float32)
    points, valid = backproject_depth(depth, camera)
    flow = np.zeros((height, width, 2), dtype=np.float32)
    varied = np.linspace(1.0, 4.0, width * height, dtype=np.float32).reshape(height, width)
    stages = {
        "backprojection": lambda: backproject_depth(depth, camera),
        "normals_baseline": lambda: estimate_normals(points, valid, depth, NormalMode.BASELINE),
        "normals_edge_aware": lambda: estimate_normals(points, valid, depth, NormalMode.EDGE_AWARE),
        "normals_multi_scale": lambda: estimate_normals(points, valid, depth, NormalMode.MULTI_SCALE),
        "generic_warp": lambda: warp_field_backward(depth, flow),
        "depth_aware_warp": lambda: warp_depth_backward(depth, flow),
        "alignment_affine": lambda: align_inverse_depth(varied, varied * 1.1, valid, min_samples=64),
        "alignment_scale_only": lambda: align_inverse_depth(np.full_like(depth, 2.2), depth, valid, min_samples=64),
    }
    results = {name: _run(fn, warmups, iterations) for name, fn in stages.items()}
    motion = MotionState(0, 1, 1 / 30, flow, flow)
    fresh_engine = TemporalGeometryEngine(camera)
    history_engine = TemporalGeometryEngine(camera)
    fresh_engine.update(None, camera, 0, 0.0, DepthState(depth, 0.0, 0, "relative"))
    history_engine.update(None, camera, 0, 0.0, DepthState(depth, 0.0, 0, "relative"))
    fresh_frame = 0
    def fresh_update():
        nonlocal fresh_frame
        fresh_frame += 1
        fresh_engine.update(None, camera, fresh_frame, fresh_frame / 30.0, DepthState(depth, fresh_frame / 30.0, fresh_frame, "relative"), MotionState(fresh_frame - 1, fresh_frame, fresh_frame / 30.0, flow, flow))
    results["temporal_update_fresh_depth"] = _run(fresh_update, warmups, iterations)
    frame = 1
    def history_only():
        nonlocal frame
        frame += 1
        history_engine.update(None, camera, frame, frame / 30.0, None, MotionState(frame - 1, frame, frame / 30.0, flow, flow))
    results["temporal_update_history_only"] = _run(history_only, warmups, iterations)
    return {"resolution": [width, height], "stages": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.warmups < 0 or args.iterations < 2:
        parser.error("resolution must be positive, warmups non-negative, iterations at least 2")
    result = benchmark(args.width, args.height, args.warmups, args.iterations)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for stage, stats in result["stages"].items():
            print(f"{stage:30s} mean={stats['mean']:.3f} p50={stats['p50']:.3f} p95={stats['p95']:.3f} p99={stats['p99']:.3f} max={stats['max']:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
