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

from geometry import CameraModel, DepthState, MotionState, TemporalConfig, TemporalGeometryEngine, geometry_state_nbytes, motion_state_nbytes, validate_renderer_geometry  # noqa: E402


def run(frames: int, width: int, height: int, depth_period: int, seed: int, inject_bad_flow: bool, inject_invalid_depth: bool, scale_flicker: bool, depth_schedule: list[bool] | None = None, motion_dx: float = 0.0) -> dict:
    camera = CameraModel(width, height, width * 0.9, height * 0.9, (width - 1) / 2, (height - 1) / 2)
    rng = np.random.default_rng(seed)
    engine = TemporalGeometryEngine(camera, TemporalConfig(diagnostics_level="basic"))
    flow = np.zeros((height, width, 2), dtype=np.float32)
    latencies, valid_rates, confidences = [], [], []
    reset_count = 0
    flow_failures = 0
    invalid_depth_injections = 0
    bad_flow_injections = 0
    history_expirations = 0
    invariant_failures = 0
    alignment_failures = 0
    max_memory = 0
    for frame_id in range(frames):
        fresh = (depth_schedule[frame_id % len(depth_schedule)] if depth_schedule else (frame_id == 0 or frame_id % depth_period == 0))
        x_axis = np.linspace(0, 6, width, dtype=np.float32)
        depth = (2.0 + 0.02 * np.sin(x_axis)[None, :] + rng.normal(0, 0.01, (height, width))).astype(np.float32)
        if motion_dx:
            left = int(round(width * 0.2 + frame_id * motion_dx)) % max(width, 1)
            right = min(width, left + max(2, width // 5))
            depth[:, left:right] = 1.0
        if scale_flicker and frame_id % 7 == 3:
            depth *= 1.25
        if inject_invalid_depth and frame_id % 11 == 5:
            depth[height // 4:height // 2, width // 4:width // 2] = np.nan
            invalid_depth_injections += 1
        motion = None
        if frame_id:
            error = np.zeros((height, width), dtype=np.float32)
            confidence = np.ones((height, width), dtype=np.float32)
            if motion_dx:
                flow[..., 0] = motion_dx
                backward = -flow
            else:
                backward = flow
            if inject_bad_flow and frame_id % 13 == 4:
                error[height // 3:height // 2, width // 3:width // 2] = 50.0
                confidence[height // 3:height // 2, width // 3:width // 2] = 0.0
                bad_flow_injections += 1
            motion = MotionState(frame_id - 1, frame_id, frame_id / 30.0, flow, backward, forward_backward_error=error, flow_confidence=confidence)
        start = time.perf_counter()
        state = engine.update(None, camera, frame_id, frame_id / 30.0, DepthState(depth, frame_id / 30.0, frame_id, "relative") if fresh else None, motion)
        latencies.append((time.perf_counter() - start) * 1000.0)
        valid_rates.append(float(state.valid_mask.mean()))
        confidences.append(float(state.confidence.mean()))
        report = validate_renderer_geometry(state)
        if not report.valid:
            invariant_failures += 1
        retained = geometry_state_nbytes(engine.previous_state) if engine.previous_state is not None else 0
        retained += motion_state_nbytes(engine.motion_state) if engine.motion_state is not None else 0
        retained += int(engine.previous_rgb.nbytes) if isinstance(engine.previous_rgb, np.ndarray) else 0
        max_memory = max(max_memory, retained)
        if engine.last_diagnostics is not None and engine.last_diagnostics.reset_reason not in (None, "none", "first_frame"):
            reset_count += 1
        if not state.valid_mask.any() and not fresh:
            history_expirations += 1
        if engine.last_alignment is not None and fresh and frame_id and not engine.last_alignment.fit_success:
            alignment_failures += 1
        flow_failures = int(getattr(engine, "flow_failures", 0))
    return {
        "frames": frames, "resolution": [width, height], "seed": seed, "depth_period": depth_period,
        "reset_count": reset_count, "flow_failures": flow_failures, "invalid_depth_injections": invalid_depth_injections,
        "bad_flow_injections": bad_flow_injections, "alignment_failures": alignment_failures, "history_expirations": history_expirations, "invariant_failures": invariant_failures,
        "motion_dx": motion_dx, "depth_schedule": depth_schedule,
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
    parser.add_argument("--motion-dx", type=float, default=0.0, help="deterministic horizontal translation per frame (supports subpixels)")
    parser.add_argument("--depth-schedule", type=str, default=None, help="comma-separated fresh/history schedule, e.g. fresh,history,history")
    args = parser.parse_args()
    if args.frames < 1 or args.width < 4 or args.height < 4 or args.depth_period < 1:
        parser.error("frames, width, height, and depth-period must be positive")
    schedule = None
    if args.depth_schedule:
        tokens = [token.strip().lower() for token in args.depth_schedule.split(",") if token.strip()]
        if not tokens or any(token not in {"fresh", "history"} for token in tokens):
            parser.error("depth-schedule must contain only fresh or history")
        schedule = [token == "fresh" for token in tokens]
    print(json.dumps(run(args.frames, args.width, args.height, args.depth_period, args.seed, args.inject_bad_flow, args.inject_invalid_depth, args.scale_flicker, schedule, args.motion_dx), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
