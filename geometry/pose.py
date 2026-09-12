"""Lightweight camera-pose estimation for optional persistent geometry.

The estimator is deliberately independent of the Phase 1-4 temporal engine. It
accepts compact 3D-to-2D correspondences and returns an invalid result instead
of guessing when OpenCV or the scene geometry is unsuitable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .camera import CameraModel


@dataclass(frozen=True, slots=True)
class CameraPoseState:
    timestamp: float
    source_frame_id: int | str
    T_world_from_camera: np.ndarray
    valid: bool
    confidence: float
    inlier_count: int = 0
    reprojection_error: float = float("inf")

    def __post_init__(self) -> None:
        transform = np.asarray(self.T_world_from_camera, dtype=np.float32)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("T_world_from_camera must be a finite 4x4 matrix")
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("pose confidence must be in [0, 1]")
        object.__setattr__(self, "T_world_from_camera", transform)


@dataclass(frozen=True, slots=True)
class PoseEstimateResult:
    valid: bool
    T_current_from_previous: np.ndarray
    confidence: float
    inlier_mask: np.ndarray
    inlier_count: int
    reprojection_error: float
    median_reprojection_error: float = float("inf")
    p95_reprojection_error: float = float("inf")
    correspondence_count: int = 0
    spatial_coverage: float = 0.0
    reason: str = "none"

    def __post_init__(self) -> None:
        transform = np.asarray(self.T_current_from_previous, dtype=np.float32)
        mask = np.asarray(self.inlier_mask, dtype=bool)
        if transform.shape != (4, 4) or mask.ndim != 1:
            raise ValueError("pose result has invalid transform or inlier mask")
        object.__setattr__(self, "T_current_from_previous", transform)
        object.__setattr__(self, "inlier_mask", mask)
        object.__setattr__(self, "confidence", float(np.clip(self.confidence, 0.0, 1.0)))


def _invalid(count: int, reason: str, coverage: float = 0.0) -> PoseEstimateResult:
    return PoseEstimateResult(False, np.eye(4, dtype=np.float32), 0.0, np.zeros(count, dtype=bool), 0, float("inf"), correspondence_count=count, spatial_coverage=coverage, reason=reason)


class PoseEstimator:
    """PnP-RANSAC pose estimator using ``T_current_from_previous`` semantics."""

    def __init__(self, camera: CameraModel, *, max_samples: int = 2000, min_correspondences: int = 6, min_inliers: int = 6, reprojection_error: float = 3.0, iterations: int = 100, confidence: float = 0.995) -> None:
        if max_samples < min_correspondences or min_correspondences < 4 or min_inliers < 4:
            raise ValueError("pose correspondence limits are invalid")
        self.camera = camera
        self.max_samples = int(max_samples)
        self.min_correspondences = int(min_correspondences)
        self.min_inliers = int(min_inliers)
        self.reprojection_error = float(reprojection_error)
        self.iterations = int(iterations)
        self.ransac_confidence = float(confidence)

    @staticmethod
    def _sample(object_points: np.ndarray, image_points: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if len(object_points) <= max_samples:
            return object_points, image_points
        # Deterministic spatial ordering of image correspondences, then evenly sample.
        order = np.lexsort((image_points[:, 0], image_points[:, 1]))
        take = np.linspace(0, len(order) - 1, max_samples, dtype=np.int64)
        selected = order[take]
        return object_points[selected], image_points[selected]

    def estimate(self, object_points: np.ndarray, image_points: np.ndarray, *, camera: CameraModel | None = None) -> PoseEstimateResult:
        points3d = np.asarray(object_points, dtype=np.float32)
        pixels = np.asarray(image_points, dtype=np.float32)
        if points3d.ndim != 2 or points3d.shape[1] != 3 or pixels.shape != (len(points3d), 2):
            raise ValueError("object_points must be (N,3) and image_points must be (N,2)")
        finite = np.isfinite(points3d).all(axis=1) & np.isfinite(pixels).all(axis=1)
        points3d, pixels = points3d[finite], pixels[finite]
        if len(points3d) < self.min_correspondences:
            return _invalid(len(points3d), "too_few_correspondences")
        coverage = float(np.prod(np.clip((pixels.max(axis=0) - pixels.min(axis=0)) / np.array([self.camera.width, self.camera.height]), 0.0, 1.0)))
        if coverage < 1e-4:
            return _invalid(len(points3d), "insufficient_spatial_coverage", coverage)
        centered = pixels - pixels.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        if len(singular) < 2 or singular[1] / max(singular[0], 1e-6) < 0.01:
            return _invalid(len(points3d), "near_collinear_correspondences", coverage)
        points3d, pixels = self._sample(points3d, pixels, self.max_samples)
        try:
            import cv2
        except ImportError:
            return _invalid(len(points3d), "opencv_unavailable", coverage)
        cam = camera or self.camera
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                points3d.astype(np.float64), pixels.astype(np.float64), cam.camera_matrix, None,
                iterationsCount=self.iterations, reprojectionError=self.reprojection_error,
                confidence=self.ransac_confidence, flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return _invalid(len(points3d), "pnp_failed", coverage)
        if not ok or inliers is None or len(inliers) < self.min_inliers:
            return _invalid(len(points3d), "insufficient_inliers", coverage)
        rotation, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation.astype(np.float32)
        transform[:3, 3] = np.asarray(tvec, dtype=np.float32).reshape(3)
        projected, _ = cv2.projectPoints(points3d.astype(np.float64), rvec, tvec, cam.camera_matrix, None)
        errors = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
        inlier_mask = np.zeros(len(points3d), dtype=bool)
        inlier_mask[np.asarray(inliers).reshape(-1)] = True
        inlier_errors = errors[inlier_mask]
        median = float(np.median(inlier_errors))
        p95 = float(np.percentile(inlier_errors, 95))
        inlier_ratio = float(inlier_mask.mean())
        count_score = min(1.0, len(inlier_errors) / 100.0)
        residual_score = math.exp(-median / max(self.reprojection_error, 1e-6))
        confidence = float(np.clip(inlier_ratio * (0.35 + 0.35 * count_score + 0.30 * residual_score) * min(1.0, 10.0 * coverage), 0.0, 1.0))
        return PoseEstimateResult(True, transform, confidence, inlier_mask, int(inlier_mask.sum()), float(np.mean(inlier_errors)), median, p95, len(points3d), coverage)

    def estimate_from_flow(
        self,
        previous_geometry,
        motion,
        *,
        min_geometry_confidence: float = 0.35,
        min_motion_confidence: float = 0.35,
    ) -> PoseEstimateResult:
        """Estimate camera motion from previous GeometryState and MotionState forward flow."""
        cam = previous_geometry.camera or self.camera
        h, w = cam.height, cam.width
        pts_3d = np.asarray(previous_geometry.positions_3d, dtype=np.float32)
        flow = np.asarray(motion.forward_flow, dtype=np.float32)
        if pts_3d.shape != (h, w, 3) or flow.ndim != 3:
            return _invalid(0, "dimension_mismatch")

        if flow.shape == (h, w, 2):
            flow_u = flow[..., 0]
            flow_v = flow[..., 1]
            finite_flow = np.isfinite(flow).all(axis=-1)
        elif flow.shape == (2, h, w):
            flow_u = flow[0]
            flow_v = flow[1]
            finite_flow = np.isfinite(flow).all(axis=0)
        else:
            return _invalid(0, "dimension_mismatch")

        mask = np.asarray(previous_geometry.valid_mask, dtype=bool) & np.asarray(motion.valid_mask, dtype=bool)
        if getattr(previous_geometry, "normal_valid_mask", None) is not None:
            mask &= np.asarray(previous_geometry.normal_valid_mask, dtype=bool)
        if getattr(previous_geometry, "confidence", None) is not None:
            mask &= np.asarray(previous_geometry.confidence, dtype=np.float32) >= float(min_geometry_confidence)
        if getattr(motion, "confidence", None) is not None and motion.confidence is not None:
            mask &= np.asarray(motion.confidence, dtype=np.float32) >= float(min_motion_confidence)

        finite_pts = np.isfinite(pts_3d).all(axis=-1) & (pts_3d[..., 2] > 0)
        mask &= finite_pts & finite_flow

        valid_count = int(mask.sum())
        if valid_count < self.min_correspondences:
            return _invalid(valid_count, "too_few_reliable_correspondences")

        u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        u_sel = u_grid[mask]
        v_sel = v_grid[mask]
        u_flow_sel = flow_u[mask]
        v_flow_sel = flow_v[mask]

        p_prev = np.column_stack((u_sel, v_sel))
        p_curr = np.column_stack((u_sel + u_flow_sel, v_sel + v_flow_sel))
        obj_pts = pts_3d[mask]

        inside = (p_curr[:, 0] >= 0) & (p_curr[:, 0] < w) & (p_curr[:, 1] >= 0) & (p_curr[:, 1] < h)
        obj_pts = obj_pts[inside]
        p_curr = p_curr[inside]

        if len(obj_pts) < self.min_correspondences:
            return _invalid(len(obj_pts), "correspondences_out_of_frame")

        return self.estimate(obj_pts, p_curr, camera=cam)


def compose_world_pose(previous_world_from_camera: np.ndarray, current_from_previous: np.ndarray) -> np.ndarray:
    """Compose poses using ``T_world_from_current = T_world_from_previous @ inv(T_current_from_previous)``."""
    previous = np.asarray(previous_world_from_camera, dtype=np.float64)
    relative = np.asarray(current_from_previous, dtype=np.float64)
    if previous.shape != (4, 4) or relative.shape != (4, 4):
        raise ValueError("poses must be 4x4")
    return (previous @ np.linalg.inv(relative)).astype(np.float32)


def compute_rigid_flow_residual(
    camera: CameraModel,
    previous_positions: np.ndarray,
    forward_flow: np.ndarray,
    T_current_from_previous: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-pixel discrepancy between observed optical flow and rigid camera motion."""
    h, w = camera.height, camera.width
    pts = np.asarray(previous_positions, dtype=np.float32)
    flow = np.asarray(forward_flow, dtype=np.float32)
    t_mat = np.asarray(T_current_from_previous, dtype=np.float32)
    residual = np.full((h, w), np.nan, dtype=np.float32)

    if flow.ndim != 3:
        return residual
    if flow.shape == (h, w, 2):
        flow_u = flow[..., 0]
        flow_v = flow[..., 1]
        finite_flow = np.isfinite(flow).all(axis=-1)
    elif flow.shape == (2, h, w):
        flow_u = flow[0]
        flow_v = flow[1]
        finite_flow = np.isfinite(flow).all(axis=0)
    else:
        return residual

    valid = np.isfinite(pts).all(axis=-1) & (pts[..., 2] > 0) & finite_flow
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)

    if not valid.any():
        return residual

    rot = t_mat[:3, :3]
    trans = t_mat[:3, 3]
    pts_valid = pts[valid]
    curr_pts = (rot @ pts_valid.T).T + trans
    in_front = curr_pts[:, 2] > 1e-4

    if not in_front.any():
        return residual

    valid_indices = np.flatnonzero(valid)
    front_indices = valid_indices[in_front]

    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    u_obs = (u_grid + flow_u).reshape(-1)[front_indices]
    v_obs = (v_grid + flow_v).reshape(-1)[front_indices]
    obs_pixels = np.column_stack((u_obs, v_obs))

    rigid_pixels = camera.project(curr_pts[in_front])
    diff = np.linalg.norm(rigid_pixels - obs_pixels, axis=1)

    res_flat = residual.reshape(-1)
    res_flat[front_indices] = diff.astype(np.float32)
    return residual


def compute_static_confidence(
    rigid_residual: np.ndarray,
    spatial_confidence: np.ndarray | None = None,
    *,
    threshold: float = 1.5,
    tau: float = 2.0,
) -> np.ndarray:
    """Calculate static confidence in [0, 1] using rigid motion flow residual."""
    res = np.asarray(rigid_residual, dtype=np.float32)
    static = np.zeros_like(res, dtype=np.float32)
    finite = np.isfinite(res)
    if finite.any():
        excess = np.maximum(0.0, res[finite] - float(threshold))
        static[finite] = np.exp(-excess / max(float(tau), 1e-6))
    if spatial_confidence is not None:
        spatial = np.clip(np.asarray(spatial_confidence, dtype=np.float32), 0.0, 1.0)
        static *= spatial
    return np.clip(static, 0.0, 1.0).astype(np.float32)

