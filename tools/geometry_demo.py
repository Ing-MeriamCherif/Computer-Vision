#!/usr/bin/env python3
"""Run a camera-space reconstruction smoke demo without a neural depth model."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from geometry import CameraModel, backproject_depth
from geometry.debug import (
    exact_plane_depth,
    export_ply,
    fit_plane,
    fronto_parallel_plane,
    sphere_depth,
    sphere_normals,
    step_depth,
)
from geometry.normals import NormalMode, angular_metrics, estimate_normals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--shape", choices=("plane", "tilted", "sphere", "step"), default="tilted")
    parser.add_argument("--normals", action="store_true", help="estimate normals and report quality metrics")
    parser.add_argument("--mode", choices=[mode.value for mode in NormalMode], default=NormalMode.EDGE_AWARE.value)
    parser.add_argument("--benchmark", action="store_true", help="time all normal modes at this resolution")
    parser.add_argument("--ply", help="optional output PLY path")
    args = parser.parse_args()
    camera = CameraModel(args.width, args.height, args.width * 0.9, args.height * 0.9, (args.width - 1) / 2, (args.height - 1) / 2)
    depth = {
        "plane": fronto_parallel_plane,
        "tilted": lambda c: exact_plane_depth(c, normal=(-0.1, 0.05, 1.0), distance=-2.0),
        "sphere": sphere_depth,
        "step": step_depth,
    }[args.shape](camera)
    points, valid = backproject_depth(depth, camera, "relative")
    result = {"shape": args.shape, "resolution": [args.width, args.height], "valid_points": int(valid.sum())}
    if args.shape in ("plane", "tilted"):
        fit = fit_plane(points, valid)
        result.update(rms_plane_residual=fit.rms_residual, max_plane_residual=fit.max_abs_residual)
    if args.normals:
        normal_result = estimate_normals(points, valid, depth, args.mode)
        result.update(
            normal_mode=args.mode,
            valid_normal_percentage=float(normal_result.normal_valid_mask.mean() * 100.0),
            mean_confidence=float(normal_result.confidence.mean()),
        )
        if args.shape == "plane":
            expected = np.zeros_like(points)
            expected[..., 2] = -1.0
            metrics = angular_metrics(normal_result.normals, expected, normal_result.normal_valid_mask)
            result["angular_metrics_degrees"] = {"mean": metrics.mean_degrees, "median": metrics.median_degrees, "p95": metrics.p95_degrees}
        elif args.shape == "tilted":
            # The generated plane uses (-0.1, 0.05, 1.0); camera-facing
            # orientation requires the opposite sign under N dot P <= 0.
            vector = np.array([0.1, -0.05, -1.0])
            vector /= np.linalg.norm(vector)
            expected = np.broadcast_to(vector, points.shape)
            metrics = angular_metrics(normal_result.normals, expected, normal_result.normal_valid_mask)
            result["angular_metrics_degrees"] = {"mean": metrics.mean_degrees, "median": metrics.median_degrees, "p95": metrics.p95_degrees}
        elif args.shape == "sphere":
            metrics = angular_metrics(normal_result.normals, sphere_normals(points), normal_result.normal_valid_mask)
            result["angular_metrics_degrees"] = {"mean": metrics.mean_degrees, "median": metrics.median_degrees, "p95": metrics.p95_degrees}
    if args.benchmark:
        benchmark_depth = exact_plane_depth(camera, normal=(-0.1, 0.05, 1.0), distance=-2.0)
        benchmark_points, benchmark_valid = backproject_depth(benchmark_depth, camera, "relative")
        timings = {}
        for mode in NormalMode:
            start = time.perf_counter()
            estimate_normals(benchmark_points, benchmark_valid, benchmark_depth, mode)
            timings[mode.value] = (time.perf_counter() - start) * 1000.0
        result["normal_runtime_ms"] = timings
    if args.ply:
        export_ply(args.ply, points, valid)
        result["ply"] = args.ply
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
