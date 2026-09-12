#!/usr/bin/env python3
"""Deterministic, headless Phase 5 pose + persistent-surfel demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry import CameraModel, DepthState, PersistentGeometryMapper, PoseEstimator, SurfelMap, TemporalGeometryEngine  # noqa: E402


def run(frames: int = 5, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    camera = CameraModel(320, 240, 280.0, 275.0, 157.5, 118.5)
    points = np.column_stack((rng.uniform(-1.2, 1.2, 600), rng.uniform(-.8, .8, 600), rng.uniform(3, 6, 600))).astype(np.float32)
    # A dense deterministic plane gives the mapper visible geometry while the
    # independent point cloud supplies well-conditioned PnP correspondences.
    depth = np.full((camera.height, camera.width), 4.0, np.float32)
    pixels = camera.project(points)
    valid = np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height)
    engine = TemporalGeometryEngine(camera)
    mapper = PersistentGeometryMapper(camera, SurfelMap(voxel_size=.04, max_surfels=50000), mapping_stride=2)
    pose_estimator = PoseEstimator(camera, max_samples=1000, min_inliers=20)
    stats = []
    previous_points = points
    previous_pixels = camera.project(previous_points).astype(np.float32)
    geometry = engine.update(None, camera, 0, 0.0, DepthState(depth, 0.0, 0, "metric"))
    identity = np.eye(4, dtype=np.float32)
    from geometry import CameraPoseState
    persistent = mapper.update(geometry, CameraPoseState(0.0, 0, identity, True, 1.0, len(points), 0.0))
    stats.append({"frame": 0, "pose_valid": True, "surfel_count": mapper.map.count, "coverage_percent": float(persistent.projected_valid.mean() * 100) if persistent else 0.0})
    for frame_id in range(1, frames):
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = (0.02 * frame_id, 0.0, 0.0)
        current_points = (transform[:3, :3] @ points.T).T + transform[:3, 3]
        current_pixels = camera.project(current_points).astype(np.float32)
        result = pose_estimator.estimate(previous_points, current_pixels)
        geometry = engine.update(None, camera, frame_id, frame_id / 30.0, None, None) if False else geometry
        if result.valid:
            persistent = mapper.update(geometry, result)
        stats.append({"frame": frame_id, "pose_valid": result.valid, "pose_confidence": result.confidence, "inliers": result.inlier_count, "reprojection_error": result.reprojection_error, "surfel_count": mapper.map.count, "coverage_percent": float(persistent.projected_valid.mean() * 100) if persistent else 0.0})
        previous_points, previous_pixels = current_points, current_pixels
    return {"frames": frames, "seed": seed, "map_count": mapper.map.count, "map_memory_bytes": mapper.map.nbytes(), "stats": stats}


def export_ply(path: str, mapper: PersistentGeometryMapper) -> None:
    arrays = mapper.map.arrays()
    lines = ["ply", "format ascii 1.0", f"element vertex {len(arrays['positions_world'])}", "property float x", "property float y", "property float z", "property float nx", "property float ny", "property float nz", "property float confidence", "end_header"]
    lines.extend(" ".join(f"{value:.7g}" for value in (*point, *normal, float(conf))) for point, normal, conf in zip(arrays["positions_world"], arrays["normals_world"], arrays["confidence"]))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("frames must be positive")
    print(json.dumps(run(args.frames, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
