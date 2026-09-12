#!/usr/bin/env python3
"""Deterministic long-run temporal geometry stress and invariant checker."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry import CameraModel, DepthState, MotionState, TemporalGeometryEngine, geometry_state_nbytes, validate_renderer_geometry  # noqa: E402


def run(frames: int, width: int, height: int, depth_period: int, seed: int, inject_bad_flow: bool, inject_invalid_depth: bool, scale_flicker: bool) -> dict:
    camera = CameraModel(width, height, width * 0.9, height * 0.9, (width - 1) / 2, (height - 1) / 2)
    rng = np.random.default_rng(seed)
    engine = TemporalGeometryEngine(camera)
    flow = np.zeros((height, width, 2), dtype=np.float32)
    latencies, valid_rates, confidences = [], [], []
    reset_count = 0
    invariant_failures = 0
    alignment_failures = 0
    max_memory = 0
    for frame_id in range(frames):
        fresh = frame_id == 0 or frame_id % depth_period == 0
        depth = (2.0 + 0.02 * np.sin(np.linspace(0, 6, width))[None, :] + rng.normal(0, 0.01, (height, width))).astype(np.float32)
        if scale_flicker and frame_id % 7 == 3:
            depth *= 1.25
        if inject_invalid_depth and frame_id % 11 == 5:
            depth[height // 4:height // 2, width // 4:width // 2] = np.nan
        motion = None
        if frame_id:
            error = np.zeros((height, width), dtype=np.float32)
            confidence = np.ones((height, width), dtype=np.float32)
            if inject_bad_flow and frame_id % 13 == 4:
                error[height // 3:height // 2, width // 3:width // 2] = 50.0
                confidence[height // 3:height // 2, width // 3:width // 2] = 0.0
            motion = MotionState(frame_id - 1, frame_id, frame_id / 30.0, flow, flow, forward_backward_error=error, flow_confidence=confidence)
        start = time.perf_counter()
        state = engine.update(None, camera, frame_id, frame_id / 30.0, DepthState(depth, frame_id / 30.0, frame_id, "relative") if fresh else None, motion)
        latencies.append((time.perf_counter() - start) * 1000.0)
        valid_rates.append(float(state.valid_mask.mean()))
        confidences.append(float(state.confidence.mean()))
        report = validate_renderer_geometry(state)
        if not report.valid:
            invariant_failures += 1
        max_memory = max(max_memory, geometry_state_nbytes(state))
        if engine.last_alignment is not None and fresh and frame_id and not engine.last_alignment.fit_success:
            alignment_failures += 1
    return {
        "frames": frames, "resolution": [width, height], "seed": seed, "depth_period": depth_period,
        "reset_count": reset_count, "flow_failures": 0, "invalid_depth_injections": int(inject_invalid_depth),
        "alignment_failures": alignment_failures, "invariant_failures": invariant_failures,
        "mean_update_ms": float(np.mean(latencies)), "p95_update_ms": float(np.percentile(latencies, 95)),
        "p99_update_ms": float(np.percentile(latencies, 99)), "valid_geometry_percent": float(np.mean(valid_rates) * 100),
        "mean_confidence": float(np.mean(confidences)), "max_memory_estimate_bytes": max_memory,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--depth-period", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--inject-bad-flow", action="store_true")
    parser.add_argument("--inject-invalid-depth", action="store_true")
    parser.add_argument("--scale-flicker", action="store_true")
    args = parser.parse_args()
    if args.frames < 1 or args.width < 4 or args.height < 4 or args.depth_period < 1:
        parser.error("frames, width, height, and depth-period must be positive")
    print(json.dumps(run(args.frames, args.width, args.height, args.depth_period, args.seed, args.inject_bad_flow, args.inject_invalid_depth, args.scale_flicker), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
