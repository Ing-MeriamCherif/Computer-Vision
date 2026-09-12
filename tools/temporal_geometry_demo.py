#!/usr/bin/env python3
"""Deterministic temporal geometry smoke demo with synthetic flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry import (
    CameraModel,
    DepthState,
    MotionState,
    TemporalConfig,
    TemporalGeometryEngine,
    align_inverse_depth,
    warp_depth_backward,
)


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
    warp_ms: list[float] = []
    alignment_ms: list[float] = []
    history_acceptance: list[float] = []
    history_rejection: list[float] = []
    occlusion_rates: list[float] = []
    disocclusion_rates: list[float] = []
    temporal_ages: list[float] = []
    temporal_confs: list[float] = []
    previous_depth: np.ndarray | None = None
    sample = (args.height // 2, args.width // 2)
    for frame_id in range(args.frames):
        depth = base + rng.normal(0.0, args.noise, base.shape).astype(np.float32)
        raw_values.append(float(depth[sample]))
        motion = None if frame_id == 0 else MotionState(frame_id - 1, frame_id, frame_id / 30.0, flow, flow)
        if previous_depth is not None:
            component_start = time.perf_counter()
            warped, warped_valid, _ = warp_depth_backward(previous_depth, flow)
            warp_ms.append((time.perf_counter() - component_start) * 1000.0)
            component_start = time.perf_counter()
            align_inverse_depth(depth, warped, warped_valid, min_samples=64)
            alignment_ms.append((time.perf_counter() - component_start) * 1000.0)
        start = time.perf_counter()
        state = engine.update(None, camera, frame_id, frame_id / 30.0, DepthState(depth, frame_id / 30.0, frame_id, "relative"), motion)
        update_ms.append((time.perf_counter() - start) * 1000.0)
        stable_values.append(float(state.depth[sample]))
        if frame_id:
            history_acceptance.append(float(state.history_valid.mean() * 100.0) if state.history_valid is not None else 0.0)
            history_rejection.append(float(state.history_rejection_mask.mean() * 100.0) if state.history_rejection_mask is not None else 0.0)
            occlusion_rates.append(float(state.occlusion_mask.mean() * 100.0) if state.occlusion_mask is not None else 0.0)
            disocclusion_rates.append(float(state.disocclusion_mask.mean() * 100.0) if state.disocclusion_mask is not None else 0.0)
            temporal_ages.append(float(state.temporal_age.mean()) if state.temporal_age is not None else 0.0)
            temporal_confs.append(float(state.confidence.mean()) if state.confidence is not None else 0.0)
        previous_depth = depth
    raw = np.asarray(raw_values[1:])
    stable = np.asarray(stable_values[1:])
    raw_jitter = float(np.mean(np.abs(np.diff(raw)))) if len(raw) > 1 else 0.0
    stable_jitter = float(np.mean(np.abs(np.diff(stable)))) if len(stable) > 1 else 0.0
    last_align = engine.last_alignment
    result = {
        "resolution": [args.width, args.height],
        "frames": args.frames,
        "raw_depth_jitter": raw_jitter,
        "stabilized_depth_jitter": stable_jitter,
        "raw_depth_std": float(raw.std()),
        "stabilized_depth_std": float(stable.std()),
        "jitter_reduction_percent": float((1.0 - stable.std() / max(raw.std(), 1e-12)) * 100.0),
        "history_acceptance_percent": float(np.mean(history_acceptance)) if history_acceptance else 0.0,
        "history_rejection_percent": float(np.mean(history_rejection)) if history_rejection else 0.0,
        "occlusion_percent": float(np.mean(occlusion_rates)) if occlusion_rates else 0.0,
        "disocclusion_percent": float(np.mean(disocclusion_rates)) if disocclusion_rates else 0.0,
        "mean_temporal_age": float(np.mean(temporal_ages)) if temporal_ages else 0.0,
        "alignment_model_used": last_align.model_used if last_align is not None else "none",
        "normalized_alignment_residual": float(last_align.normalized_fit_residual) if last_align is not None else 0.0,
        "mean_temporal_confidence": float(np.mean(temporal_confs)) if temporal_confs else 0.0,
        "update_ms_mean": float(np.mean(update_ms[1:])),
        "update_ms_p50": float(np.percentile(update_ms[1:], 50)),
        "update_ms_p95": float(np.percentile(update_ms[1:], 95)),
        "depth_aware_warp_ms_mean": float(np.mean(warp_ms)) if warp_ms else 0.0,
        "depth_aware_warp_ms_p50": float(np.percentile(warp_ms, 50)) if warp_ms else 0.0,
        "depth_aware_warp_ms_p95": float(np.percentile(warp_ms, 95)) if warp_ms else 0.0,
        "alignment_ms_mean": float(np.mean(alignment_ms)) if alignment_ms else 0.0,
        "alignment_ms_p50": float(np.percentile(alignment_ms, 50)) if alignment_ms else 0.0,
        "alignment_ms_p95": float(np.percentile(alignment_ms, 95)) if alignment_ms else 0.0,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
