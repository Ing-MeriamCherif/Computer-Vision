#!/usr/bin/env python3
"""Deterministic temporal geometry smoke demo with synthetic flow."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from geometry import CameraModel, DepthState, MotionState, TemporalConfig, TemporalGeometryEngine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.03)
    args = parser.parse_args()
    if args.frames < 2 or args.noise < 0:
        raise SystemExit("--frames must be at least 2 and --noise must be non-negative")
    camera = CameraModel(args.width, args.height, args.width * 0.9, args.height * 0.9, (args.width - 1) / 2, (args.height - 1) / 2)
    base = np.full((args.height, args.width), 2.0, dtype=np.float32)
    flow = np.zeros((args.height, args.width, 2), dtype=np.float32)
    engine = TemporalGeometryEngine(camera, TemporalConfig(history_weight_max=0.9, history_decay=0.98))
    rng = np.random.default_rng(17)
    raw_values: list[float] = []
    stable_values: list[float] = []
    update_ms: list[float] = []
    history_acceptance: list[float] = []
    sample = (args.height // 2, args.width // 2)
    for frame_id in range(args.frames):
        depth = base + rng.normal(0.0, args.noise, base.shape).astype(np.float32)
        raw_values.append(float(depth[sample]))
        motion = None if frame_id == 0 else MotionState(frame_id - 1, frame_id, frame_id / 30.0, flow, flow)
        start = time.perf_counter()
        state = engine.update(None, camera, frame_id, frame_id / 30.0, DepthState(depth, frame_id / 30.0, frame_id, "relative"), motion)
        update_ms.append((time.perf_counter() - start) * 1000.0)
        stable_values.append(float(state.depth[sample]))
        if frame_id:
            history_acceptance.append(float(state.history_valid.mean() * 100.0) if state.history_valid is not None else 0.0)
    raw = np.asarray(raw_values[1:])
    stable = np.asarray(stable_values[1:])
    result = {
        "resolution": [args.width, args.height],
        "frames": args.frames,
        "raw_depth_std": float(raw.std()),
        "stabilized_depth_std": float(stable.std()),
        "jitter_reduction_percent": float((1.0 - stable.std() / max(raw.std(), 1e-12)) * 100.0),
        "update_ms_mean": float(np.mean(update_ms[1:])),
        "update_ms_p95": float(np.percentile(update_ms[1:], 95)),
        "history_acceptance_percent": float(np.mean(history_acceptance)) if history_acceptance else 0.0,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
