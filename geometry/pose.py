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


def _validate_rigid_transform(transform: np.ndarray, name: str = "transform") -> None:
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    bottom = transform[3, :]
    if not np.allclose(bottom, [0.0, 0.0, 0.0, 1.0], atol=1e-2):
        raise ValueError(f"{name} bottom row must be approximately [0, 0, 0, 1]")
    r = transform[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3, dtype=r.dtype), atol=5e-2):
        raise ValueError(f"{name} rotation matrix must be approximately orthonormal")
    det = float(np.linalg.det(r))
    if not math.isclose(det, 1.0, abs_tol=5e-2):
        raise ValueError(f"{name} rotation determinant must be approximately +1, got {det:.4f}")


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
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("pose confidence must be in [0, 1]")
        if self.valid:
            _validate_rigid_transform(transform, "T_world_from_camera")
        else:
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError("T_world_from_camera must be a finite 4x4 matrix")
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
    source_frame_id: int | str | None = None
    target_frame_id: int | str | None = None

    def __post_init__(self) -> None:
        transform = np.asarray(self.T_current_from_previous, dtype=np.float32)
        mask = np.asarray(self.inlier_mask, dtype=bool)
        if transform.shape != (4, 4) or mask.ndim != 1:
            raise ValueError("pose result has invalid transform or inlier mask")
        if not np.isfinite(transform).all():
            raise ValueError("pose transform must be finite")
        if self.valid:
            _validate_rigid_transform(transform, "T_current_from_previous")
        if self.inlier_count < 0:
            raise ValueError("inlier_count cannot be negative")
        if self.correspondence_count > 0 and self.inlier_count > self.correspondence_count:
            raise ValueError("inlier_count cannot exceed correspondence_count")
        if self.reprojection_error < 0.0:
            raise ValueError("reprojection_error cannot be negative")
        object.__setattr__(self, "T_current_from_previous", transform)
        object.__setattr__(self, "inlier_mask", mask)
        object.__setattr__(self, "confidence", float(np.clip(self.confidence, 0.0, 1.0)))


@dataclass(frozen=True, slots=True)
class PoseCorrespondences:
    object_points: np.ndarray  # (N, 3) 3D camera coordinates in previous frame
    current_pixels: np.ndarray  # (N, 2) 2D pixel coordinates in current frame
    previous_pixels: np.ndarray  # (N, 2) 2D pixel coordinates in previous frame
    weights: np.ndarray  # (N,) quality weights in [0, 1]
    source_indices: tuple[np.ndarray, np.ndarray]  # (rows, cols) in previous frame
    correspondence_count: int
    spatial_coverage: float  # fraction of image area covered by bounding box


@dataclass(frozen=True, slots=True)
class PoseConfig:
    """Centralized governor for pose estimation."""
    max_samples: int = 1000
    min_correspondences: int = 6
    min_inliers: int = 6
    reprojection_error: float = 3.0
    iterations: int = 100
    confidence: float = 0.995
    min_spatial_coverage: float = 1e-4
    min_geometry_confidence: float = 0.35
    min_motion_confidence: float = 0.35


def _invalid(
    count: int,
    reason: str,
    coverage: float = 0.0,
    source_frame_id: int | str | None = None,
    target_frame_id: int | str | None = None,
) -> PoseEstimateResult:
    return PoseEstimateResult(
        False,
        np.eye(4, dtype=np.float32),
        0.0,
        np.zeros(count, dtype=bool),
        0,
        float("inf"),
        correspondence_count=count,
        spatial_coverage=coverage,
        reason=reason,
        source_frame_id=source_frame_id,
        target_frame_id=target_frame_id,
    )


def extract_pose_correspondences(
    previous_geometry,
    motion,
    *,
    current_camera: CameraModel | None = None,
    min_geometry_confidence: float = 0.35,
    min_motion_confidence: float = 0.35,
    max_samples: int = 1000,
    grid_cells: tuple[int, int] = (16, 16),
) -> PoseCorrespondences:
    """Extract spatially stratified, confidence-gated 3D-2D correspondences from states."""
    if motion.source_frame_id != previous_geometry.source_frame_id:
        raise ValueError(
            f"Frame contract mismatch: motion.source_frame_id ({motion.source_frame_id}) "
            f"!= previous_geometry.source_frame_id ({previous_geometry.source_frame_id})"
        )

    cam = current_camera or previous_geometry.camera
    h, w = cam.height, cam.width
    pts_3d = np.asarray(previous_geometry.positions_3d, dtype=np.float32)
    flow = np.asarray(motion.forward_flow, dtype=np.float32)

    if pts_3d.shape != (h, w, 3):
        raise ValueError(f"Geometry positions_3d shape {pts_3d.shape} does not match camera ({h}, {w}, 3)")

    if flow.ndim != 3:
        raise ValueError(f"Flow ndim {flow.ndim} must be 3")

    if flow.shape == (h, w, 2):
        flow_u = flow[..., 0]
        flow_v = flow[..., 1]
        finite_flow = np.isfinite(flow).all(axis=-1)
    elif flow.shape == (2, h, w):
        flow_u = flow[0]
        flow_v = flow[1]
        finite_flow = np.isfinite(flow).all(axis=0)
    else:
        raise ValueError(f"Flow shape {flow.shape} does not match resolution ({h}, {w})")

    mask = np.asarray(previous_geometry.valid_mask, dtype=bool) & np.asarray(motion.valid_mask, dtype=bool)
    if getattr(previous_geometry, "normal_valid_mask", None) is not None:
        mask &= np.asarray(previous_geometry.normal_valid_mask, dtype=bool)

    geom_conf = (
        np.asarray(previous_geometry.confidence, dtype=np.float32)
        if getattr(previous_geometry, "confidence", None) is not None
        else np.ones((h, w), dtype=np.float32)
    )
    mask &= geom_conf >= float(min_geometry_confidence)

    motion_conf = None
    if getattr(motion, "flow_confidence", None) is not None and motion.flow_confidence is not None:
        motion_conf = np.asarray(motion.flow_confidence, dtype=np.float32)
    elif getattr(motion, "confidence", None) is not None and motion.confidence is not None:
        motion_conf = np.asarray(motion.confidence, dtype=np.float32)
    else:
        motion_conf = np.ones((h, w), dtype=np.float32)
    mask &= motion_conf >= float(min_motion_confidence)

    finite_pts = np.isfinite(pts_3d).all(axis=-1) & (pts_3d[..., 2] > 0)
    mask &= finite_pts & finite_flow

    # Target pixel coordinates
    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    u_curr = u_grid + flow_u
    v_curr = v_grid + flow_v

    # Filter target within image bounds
    inside = (u_curr >= 0) & (u_curr < w) & (v_curr >= 0) & (v_curr < h)
    mask &= inside

    valid_indices = np.flatnonzero(mask)
    if len(valid_indices) == 0:
        return PoseCorrespondences(
            object_points=np.empty((0, 3), dtype=np.float32),
            current_pixels=np.empty((0, 2), dtype=np.float32),
            previous_pixels=np.empty((0, 2), dtype=np.float32),
            weights=np.empty((0,), dtype=np.float32),
            source_indices=(np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)),
            correspondence_count=0,
            spatial_coverage=0.0,
        )

    # Calculate quality weights
    qual_weights = (geom_conf * motion_conf).reshape(-1)

    # Spatial stratification via image grid
    n_rows, n_cols = grid_cells
    cell_h = max(1, h // n_rows)
    cell_w = max(1, w // n_cols)

    row_indices = (valid_indices // w)
    col_indices = (valid_indices % w)

    grid_y = np.clip(row_indices // cell_h, 0, n_rows - 1)
    grid_x = np.clip(col_indices // cell_w, 0, n_cols - 1)
    cell_ids = grid_y * n_cols + grid_x

    # Select best candidate per cell first, then fill remaining quota
    unique_cells = np.unique(cell_ids)
    selected_indices: list[int] = []

    for cell_id in unique_cells:
        cell_mask = cell_ids == cell_id
        cell_cand_indices = valid_indices[cell_mask]
        cell_cand_weights = qual_weights[cell_cand_indices]
        best_cand = cell_cand_indices[np.argmax(cell_cand_weights)]
        selected_indices.append(int(best_cand))

    selected_set = set(selected_indices)
    if len(selected_indices) < max_samples:
        remaining = [idx for idx in valid_indices if idx not in selected_set]
        if remaining:
            rem_arr = np.asarray(remaining, dtype=np.int64)
            rem_weights = qual_weights[rem_arr]
            top_rem = rem_arr[np.argsort(-rem_weights)[: max_samples - len(selected_indices)]]
            selected_indices.extend(top_rem.tolist())
    elif len(selected_indices) > max_samples:
        sel_arr = np.asarray(selected_indices, dtype=np.int64)
        sel_weights = qual_weights[sel_arr]
        selected_indices = sel_arr[np.argsort(-sel_weights)[:max_samples]].tolist()

    sel_arr = np.asarray(selected_indices, dtype=np.int64)
    sel_rows = sel_arr // w
    sel_cols = sel_arr % w

    obj_pts = pts_3d[sel_rows, sel_cols]
    prev_px = np.column_stack((sel_cols.astype(np.float32), sel_rows.astype(np.float32)))
    curr_px = np.column_stack((u_curr[sel_rows, sel_cols], v_curr[sel_rows, sel_cols]))
    weights = qual_weights[sel_arr].astype(np.float32)

    dx = float(curr_px[:, 0].max() - curr_px[:, 0].min()) / w if len(curr_px) > 0 else 0.0
    dy = float(curr_px[:, 1].max() - curr_px[:, 1].min()) / h if len(curr_px) > 0 else 0.0
    coverage = float(np.clip(dx * dy, 0.0, 1.0))

    return PoseCorrespondences(
        object_points=obj_pts.astype(np.float32),
        current_pixels=curr_px.astype(np.float32),
        previous_pixels=prev_px.astype(np.float32),
        weights=weights,
        source_indices=(sel_rows, sel_cols),
        correspondence_count=len(sel_arr),
        spatial_coverage=coverage,
    )


class PoseEstimator:
    """PnP-RANSAC pose estimator using ``T_current_from_previous`` semantics."""

    def __init__(
        self,
        camera: CameraModel,
        *,
        config: PoseConfig | None = None,
        max_samples: int = 1000,
        min_correspondences: int = 6,
        min_inliers: int = 6,
        reprojection_error: float = 3.0,
        iterations: int = 100,
        confidence: float = 0.995,
        min_spatial_coverage: float = 1e-4,
    ) -> None:
        self.camera = camera
        if config is not None:
            self.config = config
        else:
            self.config = PoseConfig(
                max_samples=max_samples,
                min_correspondences=min_correspondences,
                min_inliers=min_inliers,
                reprojection_error=reprojection_error,
                iterations=iterations,
                confidence=confidence,
                min_spatial_coverage=min_spatial_coverage,
            )
        if (
            self.config.max_samples < self.config.min_correspondences
            or self.config.min_correspondences < 4
            or self.config.min_inliers < 4
        ):
            raise ValueError("pose correspondence limits are invalid")
        self.max_samples = self.config.max_samples
        self.min_correspondences = self.config.min_correspondences
        self.min_inliers = self.config.min_inliers
        self.reprojection_error = self.config.reprojection_error
        self.iterations = self.config.iterations
        self.ransac_confidence = self.config.confidence
        self.min_spatial_coverage = self.config.min_spatial_coverage

    @staticmethod
    def _sample(object_points: np.ndarray, image_points: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if len(object_points) <= max_samples:
            return object_points, image_points
        order = np.lexsort((image_points[:, 0], image_points[:, 1]))
        take = np.linspace(0, len(order) - 1, max_samples, dtype=np.int64)
        selected = order[take]
        return object_points[selected], image_points[selected]

    def estimate(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        *,
        camera: CameraModel | None = None,
        source_frame_id: int | str | None = None,
        target_frame_id: int | str | None = None,
    ) -> PoseEstimateResult:
        points3d = np.asarray(object_points, dtype=np.float32)
        pixels = np.asarray(image_points, dtype=np.float32)
        if points3d.ndim != 2 or points3d.shape[1] != 3 or pixels.shape != (len(points3d), 2):
            raise ValueError("object_points must be (N,3) and image_points must be (N,2)")
        finite = np.isfinite(points3d).all(axis=1) & np.isfinite(pixels).all(axis=1)
        points3d, pixels = points3d[finite], pixels[finite]
        if len(points3d) < self.min_correspondences:
            return _invalid(len(points3d), "too_few_correspondences", source_frame_id=source_frame_id, target_frame_id=target_frame_id)
        cam = camera or self.camera
        coverage = float(np.prod(np.clip((pixels.max(axis=0) - pixels.min(axis=0)) / np.array([cam.width, cam.height]), 0.0, 1.0)))
        if coverage < self.min_spatial_coverage:
            return _invalid(len(points3d), "insufficient_spatial_coverage", coverage, source_frame_id=source_frame_id, target_frame_id=target_frame_id)
        centered = pixels - pixels.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        if len(singular) < 2 or singular[1] / max(singular[0], 1e-6) < 0.01:
            return _invalid(len(points3d), "near_collinear_correspondences", coverage, source_frame_id=source_frame_id, target_frame_id=target_frame_id)
        points3d, pixels = self._sample(points3d, pixels, self.max_samples)
        try:
            import cv2
        except ImportError:
            return _invalid(len(points3d), "opencv_unavailable", coverage, source_frame_id=source_frame_id, target_frame_id=target_frame_id)
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                points3d.astype(np.float64),
                pixels.astype(np.float64),
                cam.camera_matrix,
                None,
                iterationsCount=self.iterations,
                reprojectionError=self.reprojection_error,
                confidence=self.ransac_confidence,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return _invalid(len(points3d), "pnp_failed", coverage, source_frame_id=source_frame_id, target_frame_id=target_frame_id)
        if not ok or inliers is None or len(inliers) < self.min_inliers:
            return _invalid(len(points3d), "insufficient_inliers", coverage, source_frame_id=source_frame_id, target_frame_id=target_frame_id)
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
        return PoseEstimateResult(
            True,
            transform,
            confidence,
            inlier_mask,
            int(inlier_mask.sum()),
            float(np.mean(inlier_errors)),
            median,
            p95,
            len(points3d),
            coverage,
            source_frame_id=source_frame_id,
            target_frame_id=target_frame_id,
        )

    def estimate_from_correspondences(
        self,
        correspondences: PoseCorrespondences,
        *,
        camera: CameraModel | None = None,
        source_frame_id: int | str | None = None,
        target_frame_id: int | str | None = None,
    ) -> PoseEstimateResult:
        """Estimate camera motion from precomputed PoseCorrespondences."""
        if correspondences.correspondence_count < self.min_correspondences:
            return _invalid(correspondences.correspondence_count, "too_few_correspondences", correspondences.spatial_coverage, source_frame_id=source_frame_id, target_frame_id=target_frame_id)
        return self.estimate(correspondences.object_points, correspondences.current_pixels, camera=camera, source_frame_id=source_frame_id, target_frame_id=target_frame_id)

    def estimate_from_states(
        self,
        previous_geometry,
        motion,
        *,
        current_camera: CameraModel | None = None,
        min_geometry_confidence: float | None = None,
        min_motion_confidence: float | None = None,
    ) -> PoseEstimateResult:
        """High-level pose estimation directly from previous GeometryState and MotionState."""
        min_geom = self.config.min_geometry_confidence if min_geometry_confidence is None else min_geometry_confidence
        min_mot = self.config.min_motion_confidence if min_motion_confidence is None else min_motion_confidence
        corr = extract_pose_correspondences(
            previous_geometry,
            motion,
            current_camera=current_camera or self.camera,
            min_geometry_confidence=min_geom,
            min_motion_confidence=min_mot,
            max_samples=self.max_samples,
        )
        source_id = getattr(previous_geometry, "source_frame_id", None)
        target_id = getattr(motion, "target_frame_id", None)
        return self.estimate_from_correspondences(
            corr,
            camera=current_camera or self.camera,
            source_frame_id=source_id,
            target_frame_id=target_id,
        )

    def estimate_from_flow(
        self,
        previous_geometry,
        motion,
        *,
        min_geometry_confidence: float = 0.35,
        min_motion_confidence: float = 0.35,
    ) -> PoseEstimateResult:
        """Backward-compatible alias for estimate_from_states."""
        return self.estimate_from_states(
            previous_geometry,
            motion,
            min_geometry_confidence=min_geometry_confidence,
            min_motion_confidence=min_motion_confidence,
        )


def compose_world_pose(previous_world_from_camera: np.ndarray, current_from_previous: np.ndarray) -> np.ndarray:
    """Compose poses using ``T_world_from_current = T_world_from_previous @ inv(T_current_from_previous)``."""
    previous = np.asarray(previous_world_from_camera, dtype=np.float64)
    relative = np.asarray(current_from_previous, dtype=np.float64)
    if previous.shape != (4, 4) or relative.shape != (4, 4):
        raise ValueError("poses must be 4x4")
    composed = previous @ np.linalg.inv(relative)
    return composed.astype(np.float32)


def pose_error(
    estimated_world_from_camera: np.ndarray,
    ground_truth_world_from_camera: np.ndarray,
) -> tuple[float, float]:
    """Compute rotation error in degrees and translation error norm between two 4x4 poses."""
    T_est = np.asarray(estimated_world_from_camera, dtype=np.float64)
    T_gt = np.asarray(ground_truth_world_from_camera, dtype=np.float64)
    R_rel = T_est[:3, :3] @ T_gt[:3, :3].T
    tr = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_deg = float(np.rad2deg(np.arccos(tr)))
    trans_err = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
    return rot_err_deg, trans_err


def compute_rigid_flow_residual(
    camera: CameraModel,
    previous_positions: np.ndarray,
    forward_flow: np.ndarray,
    T_current_from_previous: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    current_domain: bool = False,
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

    if current_domain:
        res_curr = np.full((h, w), np.nan, dtype=np.float32)
        uc = np.rint(obs_pixels[:, 0]).astype(np.int32)
        vc = np.rint(obs_pixels[:, 1]).astype(np.int32)
        inside_curr = (uc >= 0) & (uc < w) & (vc >= 0) & (vc < h)
        if inside_curr.any():
            uc_in = uc[inside_curr]
            vc_in = vc[inside_curr]
            diff_in = diff[inside_curr].astype(np.float32)
            for idx in range(len(uc_in)):
                x, y, d = uc_in[idx], vc_in[idx], diff_in[idx]
                if np.isnan(res_curr[y, x]) or d > res_curr[y, x]:
                    res_curr[y, x] = d
        return res_curr

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


@dataclass(frozen=True, slots=True)
class StaticGeometryResult:
    """Per-pixel static vs dynamic classification outcome."""
    static_confidence: np.ndarray  # (H, W) in [0, 1]
    rigid_residual: np.ndarray  # (H, W) flow residual in pixels
    static_mask: np.ndarray  # (H, W) bool
    dynamic_mask: np.ndarray  # (H, W) bool
    valid_mask: np.ndarray  # (H, W) bool
    static_pixel_percent: float
    dynamic_pixel_percent: float


def classify_static_geometry(
    previous_geometry,
    motion,
    pose_result: PoseEstimateResult | CameraPoseState | np.ndarray,
    camera: CameraModel | None = None,
    *,
    current_geometry=None,
    current_domain: bool = False,
    threshold: float = 1.5,
    tau: float = 2.0,
    min_static_confidence: float = 0.5,
) -> StaticGeometryResult:
    """Classify per-pixel static vs dynamic geometry using rigid motion residual."""
    cam = camera or previous_geometry.camera
    h, w = cam.height, cam.width
    if isinstance(pose_result, PoseEstimateResult):
        t_mat = pose_result.T_current_from_previous
    elif isinstance(pose_result, CameraPoseState):
        if hasattr(pose_result, "T_current_from_previous"):
            t_mat = getattr(pose_result, "T_current_from_previous")
        else:
            raise ValueError(
                "classify_static_geometry requires relative transform T_current_from_previous, not T_world_from_camera"
            )
    elif isinstance(pose_result, np.ndarray) and pose_result.shape == (4, 4):
        t_mat = pose_result
    else:
        raise TypeError(f"Unsupported pose_result type: {type(pose_result).__name__}")

    use_curr = current_domain or (current_geometry is not None)
    res = compute_rigid_flow_residual(
        cam,
        previous_geometry.positions_3d,
        motion.forward_flow,
        t_mat,
        valid_mask=previous_geometry.valid_mask,
        current_domain=use_curr,
    )
    if use_curr and current_geometry is not None:
        spatial_conf = getattr(current_geometry, "spatial_confidence", None)
        if spatial_conf is None:
            spatial_conf = getattr(current_geometry, "confidence", None)
    else:
        spatial_conf = getattr(previous_geometry, "spatial_confidence", None)
        if spatial_conf is None:
            spatial_conf = getattr(previous_geometry, "confidence", None)

    conf = compute_static_confidence(
        res,
        spatial_conf,
        threshold=threshold,
        tau=tau,
    )
    finite = np.isfinite(res)
    static_mask = finite & (conf >= float(min_static_confidence))
    dynamic_mask = finite & (conf < float(min_static_confidence))
    total_pixels = h * w
    static_pct = float(static_mask.sum() / max(1, total_pixels) * 100.0)
    dynamic_pct = float(dynamic_mask.sum() / max(1, total_pixels) * 100.0)

    return StaticGeometryResult(
        static_confidence=conf,
        rigid_residual=res,
        static_mask=static_mask,
        dynamic_mask=dynamic_mask,
        valid_mask=finite,
        static_pixel_percent=static_pct,
        dynamic_pixel_percent=dynamic_pct,
    )
