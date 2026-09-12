#!/usr/bin/env python3
"""Dedicated Phase 5 persistent geometry performance and memory benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry import (  # noqa: E402
    CameraModel,
    CameraPoseState,
    DepthState,
    GeometryState,
    MotionState,
    PersistentGeometryConfig,
    PersistentGeometryMapper,
    PoseEstimator,
    SurfelMap,
    TemporalGeometryEngine,
    classify_static_geometry,
    extract_pose_correspondences,
)


def _benchmark_resolution(
    width: int,
    height: int,
    warmup: int = 2,
    repeats: int = 5,
) -> dict[str, float]:
    fx = fy = float(width) * 0.85
    cx, cy = width / 2.0 - 0.5, height / 2.0 - 0.5
    camera = CameraModel(width, height, fx, fy, cx, cy)

    # Synthetic plane at 2.5m
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    depth = np.full((height, width), 2.5, dtype=np.float32)
    pos_3d = np.zeros((height, width, 3), dtype=np.float32)
    pos_3d[..., 0] = (u - cx) * depth / fx
    pos_3d[..., 1] = (v - cy) * depth / fy
    pos_3d[..., 2] = depth
    normals = np.zeros_like(pos_3d)
    normals[..., 2] = -1.0
    valid_mask = np.ones((height, width), dtype=bool)
    confidence = np.ones((height, width), dtype=np.float32)

    geom = GeometryState(
        0.0,
        0,
        depth,
        pos_3d,
        valid_mask,
        camera,
        "metric",
        normals=normals,
        normal_valid_mask=valid_mask,
        confidence=confidence,
    )

    # 2-pixel translation flow
    flow = np.zeros((height, width, 2), dtype=np.float32)
    flow[..., 0] = 2.0
    motion = MotionState(
        source_frame_id=0,
        target_frame_id=1,
        timestamp=1.0 / 30.0,
        forward_flow=flow,
        backward_flow=-flow,
        valid_mask=valid_mask,
    )

    estimator = PoseEstimator(camera, max_samples=800, min_inliers=15)
    config = PersistentGeometryConfig(voxel_size=0.03, max_surfels=50000, mapping_stride=4)
    mapper = PersistentGeometryMapper(camera, config=config)
    init_pose = CameraPoseState(0.0, 0, np.eye(4, dtype=np.float32), True, 1.0, 100, 0.0)
    mapper.update(geom, init_pose)

    corr_times: list[float] = []
    pose_times: list[float] = []
    static_times: list[float] = []
    mapping_times: list[float] = []
    reproj_times: list[float] = []

    for i in range(warmup + repeats):
        # 1. Correspondence extraction
        t0 = time.perf_counter()
        corr = extract_pose_correspondences(geom, motion, current_camera=camera, max_samples=800)
        t1 = time.perf_counter()

        # 2. PnP Pose estimation
        pose_res = estimator.estimate_from_correspondences(corr, camera=camera)
        t2 = time.perf_counter()

        # 3. Static classification
        static_res = classify_static_geometry(geom, motion, pose_res, camera=camera)
        t3 = time.perf_counter()

        # 4. Mapping / insertion
        persistent = mapper.update(geom, pose_res, static_confidence=static_res.static_confidence)
        t4 = time.perf_counter()

        # 5. Reprojection
        pose_state = mapper.last_pose or init_pose
        mapper.map.reproject(camera, pose_state)
        t5 = time.perf_counter()

        if i >= warmup:
            corr_times.append((t1 - t0) * 1000.0)
            pose_times.append((t2 - t1) * 1000.0)
            static_times.append((t3 - t2) * 1000.0)
            mapping_times.append((t4 - t3) * 1000.0)
            reproj_times.append((t5 - t4) * 1000.0)

    med_corr = float(np.median(corr_times))
    med_pose = float(np.median(pose_times))
    med_stat = float(np.median(static_times))
    med_map = float(np.median(mapping_times))
    med_rep = float(np.median(reproj_times))
    total_ms = med_corr + med_pose + med_stat + med_map + med_rep

    return {
        "correspondence_ms": med_corr,
        "pose_ms": med_pose,
        "static_classification_ms": med_stat,
        "mapping_ms": med_map,
        "reprojection_ms": med_rep,
        "total_ms": total_ms,
    }


def _benchmark_map_reprojection(
    map_size: int,
    width: int = 640,
    height: int = 360,
    warmup: int = 2,
    repeats: int = 5,
) -> tuple[float, int, int]:
    camera = CameraModel(width, height, 540.0, 540.0, 319.5, 179.5)
    surfel_map = SurfelMap(voxel_size=0.02, max_surfels=map_size + 1000)

    # Populate synthetic surfels in front of camera
    rng = np.random.default_rng(42)
    pts = np.column_stack((
        rng.uniform(-1.5, 1.5, map_size),
        rng.uniform(-1.0, 1.0, map_size),
        rng.uniform(1.5, 4.5, map_size),
    )).astype(np.float32)
    nrms = np.zeros_like(pts)
    nrms[..., 2] = -1.0
    confs = rng.uniform(0.5, 1.0, map_size).astype(np.float32)

    from geometry.persistent import Surfel
    surfel_map._surfels = [
        Surfel(pt, norm, float(conf), 1, 0)
        for pt, norm, conf in zip(pts, nrms, confs)
    ]
    surfel_map._rebuild_buckets()
    surfel_map.revision = 1
    pose = CameraPoseState(0.0, 0, np.eye(4, dtype=np.float32), True, 1.0, 100, 0.0)

    times: list[float] = []
    for i in range(warmup + repeats):
        t0 = time.perf_counter()
        surfel_map.reproject(camera, pose)
        t1 = time.perf_counter()
        if i >= warmup:
            times.append((t1 - t0) * 1000.0)

    packed_bytes = surfel_map.packed_array_bytes()
    approx_heap = surfel_map.approx_python_bytes()
    return float(np.median(times)), packed_bytes, approx_heap


def _benchmark_insertion_strides(repeats: int = 2) -> dict[str, float]:
    """Benchmark surfel insertion at 720p resolution across strides 2, 4, 8."""
    w, h = 1280, 720
    fx, fy = 1000.0, 1000.0
    cx, cy = w / 2.0, h / 2.0
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.zeros((h, w, 3), dtype=np.float32)
    pts[..., 2] = 2.5
    pts[..., 0] = (u - cx) * pts[..., 2] / fx
    pts[..., 1] = (v - cy) * pts[..., 2] / fy
    nrms = np.zeros_like(pts)
    nrms[..., 2] = -1.0
    confs = np.ones((h, w), dtype=np.float32)

    stride_res: dict[str, float] = {}
    for stride in (2, 4, 8):
        times = []
        for _ in range(repeats):
            smap = SurfelMap(voxel_size=0.03, max_surfels=50000)
            t0 = time.perf_counter()
            smap.insert(pts, nrms, confs, 0, stride=stride)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        stride_res[f"stride_{stride}"] = float(np.median(times))
    return stride_res


def run_benchmark(quick: bool = False) -> dict:
    repeats = 2 if quick else 3

    # 1. Pipeline benchmarks across resolutions
    resolutions = {
        "320x180": (320, 180),
        "640x360": (640, 360),
        "1280x720": (1280, 720),
    }
    pipeline_res: dict[str, dict[str, float]] = {}
    for name, (w, h) in resolutions.items():
        pipeline_res[name] = _benchmark_resolution(w, h, warmup=1, repeats=repeats)

    # 2. Reprojection & Memory benchmarks across map sizes
    map_sizes = [10_000, 50_000, 100_000] if not quick else [10_000, 50_000]
    map_res: dict[str, dict[str, float | int]] = {}
    for size in map_sizes:
        size_k = f"{size // 1000}k"
        reproj_ms, packed_b, heap_b = _benchmark_map_reprojection(size, repeats=repeats)
        bytes_per_surfel = packed_b / max(1, size)
        map_res[size_k] = {
            "reprojection_ms": reproj_ms,
            "packed_array_bytes": packed_b,
            "approx_python_bytes": heap_b,
            "bytes_per_surfel": bytes_per_surfel,
        }

    # 3. Insertion stride benchmarks at 720p
    stride_res = _benchmark_insertion_strides(repeats=repeats)

    return {
        "pipeline_by_resolution": pipeline_res,
        "reprojection_and_memory_by_size": map_res,
        "insertion_by_stride_720p": stride_res,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run faster benchmark")
    parser.add_argument("--output", type=str, default=None, help="optional path to save JSON results")
    args = parser.parse_args()

    results = run_benchmark(quick=args.quick)
    json_str = json.dumps(results, indent=2)
    print(json_str)
    if args.output:
        Path(args.output).write_text(json_str + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
