#!/usr/bin/env python3
"""Deterministic, physically coherent Phase 5 pose + persistent-surfel demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
    PoseConfig,
    PoseEstimator,
    SurfelMap,
    TemporalGeometryEngine,
    classify_static_geometry,
    compute_dynamic_contamination,
    extract_pose_correspondences,
    pose_error,
)


def _generate_synthetic_world_scene() -> tuple[np.ndarray, np.ndarray]:
    """Generate a coherent static 3D world scene (back wall + side wall + floor)."""
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []

    # Back wall: Z = 3.5, X in [-1.5, 1.5], Y in [-1.0, 1.0]
    xw, yw = np.meshgrid(np.linspace(-1.5, 1.5, 60), np.linspace(-1.0, 1.0, 40))
    zw = np.full_like(xw, 3.5)
    wall_pts = np.column_stack((xw.ravel(), yw.ravel(), zw.ravel()))
    wall_nrm = np.tile([0.0, 0.0, -1.0], (len(wall_pts), 1))
    points.append(wall_pts)
    normals.append(wall_nrm)

    # Floor: Y = 1.0, X in [-1.5, 1.5], Z in [1.5, 3.5]
    xf, zf = np.meshgrid(np.linspace(-1.5, 1.5, 60), np.linspace(1.5, 3.5, 40))
    yf = np.full_like(xf, 1.0)
    floor_pts = np.column_stack((xf.ravel(), yf.ravel(), zf.ravel()))
    floor_nrm = np.tile([0.0, -1.0, 0.0], (len(floor_pts), 1))
    points.append(floor_pts)
    normals.append(floor_nrm)

    all_pts = np.vstack(points).astype(np.float32)
    all_nrms = np.vstack(normals).astype(np.float32)
    return all_pts, all_nrms


def _render_frame(
    camera: CameraModel,
    world_points: np.ndarray,
    T_world_from_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project static world points into camera frame, returning depth map and 3D positions."""
    h, w = camera.height, camera.width
    T_cam_from_world = np.linalg.inv(np.asarray(T_world_from_camera, dtype=np.float64))
    pts_h = np.concatenate((world_points.astype(np.float64), np.ones((len(world_points), 1))), axis=1)
    pts_cam = (T_cam_from_world @ pts_h.T).T[:, :3]

    in_front = pts_cam[:, 2] > 0.1
    pts_cam_front = pts_cam[in_front]

    pixels = camera.project(pts_cam_front)
    px = np.rint(pixels).astype(np.int64)
    inside = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)

    depth_map = np.full((h, w), np.nan, dtype=np.float32)
    pos_3d = np.full((h, w, 3), np.nan, dtype=np.float32)

    if inside.any():
        px_valid = px[inside]
        pts_valid = pts_cam_front[inside]
        linear = px_valid[:, 1] * w + px_valid[:, 0]
        order = np.lexsort((pts_valid[:, 2], linear))
        sorted_linear = linear[order]
        first = np.r_[True, sorted_linear[1:] != sorted_linear[:-1]]
        winners = order[first]
        wy, wx = px_valid[winners, 1], px_valid[winners, 0]
        depth_map[wy, wx] = pts_valid[winners, 2].astype(np.float32)
        pos_3d[wy, wx] = pts_valid[winners].astype(np.float32)

    # Infill small planar gaps for realistic dense depth
    valid = np.isfinite(depth_map)
    if not valid.all() and valid.any():
        med_z = float(np.nanmedian(depth_map))
        u_g, v_g = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        missing = ~valid
        depth_map[missing] = med_z
        pos_3d[missing, 0] = (u_g[missing] - camera.cx) * med_z / camera.fx
        pos_3d[missing, 1] = (v_g[missing] - camera.cy) * med_z / camera.fy
        pos_3d[missing, 2] = med_z

    return depth_map, pos_3d


def run(frames: int = 5, seed: int = 7, ply: str | None = None, dynamic: bool = True) -> dict:
    camera = CameraModel(320, 240, 280.0, 275.0, 157.5, 118.5)
    world_points, world_normals = _generate_synthetic_world_scene()

    config = PersistentGeometryConfig(
        voxel_size=0.03,
        max_surfels=50000,
        mapping_stride=2,
        min_static_confidence=0.5,
    )
    mapper = PersistentGeometryMapper(camera, config=config)
    pose_estimator = PoseEstimator(camera, max_samples=800, min_inliers=15, reprojection_error=3.0)
    base_engine = TemporalGeometryEngine(camera)

    stats: list[dict] = []
    prev_geom: GeometryState | None = None
    prev_T_gt: np.ndarray | None = None

    # Dynamic object world bounding box for contamination check
    dyn_bounds_min = np.array([-0.3, -0.3, 1.8], dtype=np.float32)
    dyn_bounds_max = np.array([0.3, 0.3, 2.4], dtype=np.float32)

    for frame_id in range(frames):
        timestamp = frame_id / 30.0

        # Smooth camera trajectory: translation in X/Z and yaw
        t_x = float(0.02 * np.sin(0.2 * frame_id))
        t_y = float(-0.005 * frame_id)
        t_z = float(0.015 * frame_id)
        yaw = float(0.01 * frame_id)

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        T_w_c = np.eye(4, dtype=np.float32)
        T_w_c[:3, :3] = [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]]
        T_w_c[:3, 3] = [t_x, t_y, t_z]

        depth, positions = _render_frame(camera, world_points, T_w_c)
        depth_state = DepthState(depth, timestamp, frame_id, "metric")
        geom = base_engine.update(None, camera, frame_id, timestamp, depth_state)

        if frame_id == 0:
            init_pose = CameraPoseState(timestamp, frame_id, np.eye(4, dtype=np.float32), True, 1.0, 100, 0.0)
            persistent = mapper.update(geom, init_pose)
            rot_err, trans_err = 0.0, 0.0
            stats.append({
                "frame": 0,
                "pose_valid": True,
                "pose_confidence": 1.0,
                "pose_inliers": 100,
                "pose_reprojection_error": 0.0,
                "rotation_error_deg": 0.0,
                "translation_error": 0.0,
                "static_pixel_percent": 100.0,
                "dynamic_pixel_percent": 0.0,
                "surfel_count": mapper.map.count,
                "coverage_percent": float(persistent.projected_valid.mean() * 100.0) if persistent else 0.0,
                "dynamic_contamination_percent": 0.0,
            })
        else:
            assert prev_geom is not None and prev_T_gt is not None
            # Ground truth relative motion T_current_from_previous
            T_c_p_gt = (np.linalg.inv(T_w_c) @ prev_T_gt).astype(np.float32)

            # Compute optical flow from previous 3D points to current camera
            pts_prev = prev_geom.positions_3d
            pts_prev_flat = pts_prev.reshape(-1, 3)
            pts_curr_pred = (T_c_p_gt[:3, :3] @ pts_prev_flat.T).T + T_c_p_gt[:3, 3]
            px_curr = camera.project(pts_curr_pred).reshape(camera.height, camera.width, 2)
            u_g, v_g = np.meshgrid(np.arange(camera.width, dtype=np.float32), np.arange(camera.height, dtype=np.float32))

            flow = np.zeros((camera.height, camera.width, 2), dtype=np.float32)
            flow[..., 0] = px_curr[..., 0] - u_g
            flow[..., 1] = px_curr[..., 1] - v_g

            # If dynamic enabled, add moving foreground object in center 20%
            if dynamic:
                flow[90:130, 130:170, 0] += 8.0
                flow[90:130, 130:170, 1] -= 5.0

            motion = MotionState(
                source_frame_id=frame_id - 1,
                target_frame_id=frame_id,
                timestamp=timestamp,
                forward_flow=flow,
                backward_flow=-flow,
                valid_mask=np.ones((camera.height, camera.width), dtype=bool),
            )

            pose_res = pose_estimator.estimate_from_states(prev_geom, motion)
            static_res = classify_static_geometry(prev_geom, motion, pose_res, camera=camera)

            persistent = None
            if pose_res.valid:
                persistent = mapper.update(geom, pose_res, static_confidence=static_res.static_confidence)

            # Trajectory error vs ground truth
            est_world = mapper.world_from_camera if mapper.world_from_camera is not None else np.eye(4, dtype=np.float32)
            rot_err, trans_err = pose_error(est_world, T_w_c)

            arrays = mapper.map.arrays()
            contam = compute_dynamic_contamination(arrays["positions_world"], dyn_bounds_min, dyn_bounds_max)

            stats.append({
                "frame": frame_id,
                "pose_valid": bool(pose_res.valid),
                "pose_confidence": float(pose_res.confidence),
                "pose_inliers": int(pose_res.inlier_count),
                "pose_reprojection_error": float(pose_res.reprojection_error),
                "rotation_error_deg": float(rot_err),
                "translation_error": float(trans_err),
                "static_pixel_percent": float(static_res.static_pixel_percent),
                "dynamic_pixel_percent": float(static_res.dynamic_pixel_percent),
                "surfel_count": int(mapper.map.count),
                "coverage_percent": float(persistent.projected_valid.mean() * 100.0) if persistent else 0.0,
                "dynamic_contamination_percent": float(contam),
            })

        prev_geom = geom
        prev_T_gt = T_w_c

    if ply:
        export_ply(ply, mapper)

    return {
        "frames": frames,
        "seed": seed,
        "map_count": mapper.map.count,
        "map_memory_bytes": mapper.map.nbytes(),
        "stats": stats,
    }


def export_ply(path: str, mapper: PersistentGeometryMapper) -> None:
    arrays = mapper.map.arrays()
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(arrays['positions_world'])}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float confidence",
        "end_header",
    ]
    lines.extend(
        f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} {normal[0]:.7g} {normal[1]:.7g} {normal[2]:.7g} {float(conf):.7g}"
        for point, normal, conf in zip(arrays["positions_world"], arrays["normals_world"], arrays["confidence"])
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-dynamic", dest="dynamic", action="store_false", default=True, help="disable moving object injection")
    parser.add_argument("--ply", type=str, default=None, help="optional path to export surfel map PLY")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("frames must be positive")
    print(json.dumps(run(args.frames, args.seed, ply=args.ply, dynamic=args.dynamic), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
