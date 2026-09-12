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
    OpenCVFlowProvider,
    TemporalGeometryEngine,
    TemporalConfig,
    align_inverse_depth,
    backproject_depth,
    estimate_normals,
    summarize_timings,
    warp_depth_backward,
    warp_field_backward,
)


def _run(fn, warmups: int, iterations: int) -> dict[str, float | int | str]:
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    result = {**summarize_timings(samples), "iterations": iterations, "warmups": warmups}
    if iterations < 5:
        result["percentile_confidence"] = "limited (<5 measured samples)"
    elif iterations < 10:
        result["percentile_confidence"] = "moderate (<10 measured samples)"
    else:
        result["percentile_confidence"] = "usable"
    return result


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
        "alignment_affine_sampled": lambda: align_inverse_depth(varied, varied * 1.1, valid, min_samples=64, max_samples=5000),
        "alignment_scale_only": lambda: align_inverse_depth(np.full_like(depth, 2.2), depth, valid, min_samples=64),
    }
    texture_prev = (127.0 + 60.0 * np.sin(np.arange(width)[None, :] * 0.4) * np.cos(np.arange(height)[:, None] * 0.35)).clip(0, 255).astype(np.uint8)
    texture_cur = np.roll(texture_prev, 1, axis=1)
    optional = {}
    for method in ("dis", "farneback"):
        try:
            provider = OpenCVFlowProvider(method=method)
            # Capability is only proven by executing a small probe; constructor alone is insufficient.
            provider.compute(texture_prev, texture_cur, 0, 1, 1 / 30)
            stages[f"opencv_{method}"] = lambda provider=provider: provider.compute(texture_prev, texture_cur, 0, 1, 1 / 30)
            optional[f"opencv_{method}"] = {"status": "available"}
        except RuntimeError as exc:
            optional[f"opencv_{method}"] = {"status": "skipped", "reason": str(exc)}
    results = {name: _run(fn, warmups, iterations) for name, fn in stages.items()}
    for name, info in optional.items():
        if info["status"] == "skipped":
            results[name] = {**info, "iterations": 0, "warmups": 0}
    motion = MotionState(0, 1, 1 / 30, flow, flow)
    fresh_engine = TemporalGeometryEngine(camera)
    history_engine = TemporalGeometryEngine(camera, config=TemporalConfig(max_history_age=warmups + iterations + 4))
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
        history_engine.update(None, camera, frame, frame / 30.0, None, MotionState(frame - 1, frame, frame / 30.0, flow, flow))
        frame += 1
    results["temporal_update_history_only"] = _run(history_only, warmups, iterations)
    normal_quality = estimate_normals(points, valid, depth, NormalMode.EDGE_AWARE)
    results["normals_edge_aware"]["quality"] = {"valid_normal_percent": float(normal_quality.normal_valid_mask.mean() * 100.0)}
    for stage_name, cap in (("alignment_affine", None), ("alignment_affine_sampled", 5000), ("alignment_scale_only", None)):
        fit = align_inverse_depth(varied, varied * 1.1, valid, min_samples=64, max_samples=cap)
        results[stage_name]["quality"] = {
            "alignment_model": fit.model_used,
            "alignment_residual": fit.fit_residual,
            "alignment_sample_count": fit.sample_count,
            "alignment_input_sample_count": fit.input_sample_count,
        }
    history_quality = history_engine.current_state
    fresh_quality = fresh_engine.current_state
    if history_quality is not None:
        results["temporal_update_history_only"]["quality"] = {
            "geometry_valid_percent": float(history_quality.valid_mask.mean() * 100.0),
            "history_acceptance_percent": float(history_quality.history_valid.mean() * 100.0) if history_quality.history_valid is not None else None,
            "mean_temporal_age": float(history_quality.temporal_age[history_quality.valid_mask].mean()) if history_quality.valid_mask.any() and history_quality.temporal_age is not None else None,
        }
    if fresh_quality is not None:
        results["temporal_update_fresh_depth"]["quality"] = {
            "geometry_valid_percent": float(fresh_quality.valid_mask.mean() * 100.0),
            "mean_temporal_age": float(fresh_quality.temporal_age[fresh_quality.valid_mask].mean()) if fresh_quality.valid_mask.any() and fresh_quality.temporal_age is not None else None,
        }
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
