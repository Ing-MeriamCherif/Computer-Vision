"""Bounded world-space surfel memory for the optional Phase 5 path.

The map is intentionally a separate, single-owner stateful object. The base
camera-space :class:`TemporalGeometryEngine` remains authoritative and can be
used without importing or enabling this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .camera import CameraModel
from .pose import CameraPoseState, PoseEstimateResult, compose_world_pose
from .state import GeometryState


@dataclass(frozen=True, slots=True)
class PersistentGeometryState:
    timestamp: float
    source_frame_id: int | str
    pose: CameraPoseState
    projected_depth: np.ndarray
    projected_positions: np.ndarray
    projected_normals: np.ndarray
    projected_confidence: np.ndarray
    projected_valid: np.ndarray
    surfel_count: int
    map_revision: int
    map_stats: dict[str, float | int]


@dataclass(slots=True)
class Surfel:
    position_world: np.ndarray
    normal_world: np.ndarray
    confidence: float
    observation_count: int
    last_seen_frame: int
    color: np.ndarray | None = None


class SurfelMap:
    """A deterministic voxel-bucketed, bounded surfel map."""

    def __init__(self, *, voxel_size: float = 0.02, max_surfels: int = 100_000, max_age_frames: int = 300, min_confidence: float = 0.05, merge_distance: float | None = None, normal_merge_cos: float = 0.8660254) -> None:
        if voxel_size <= 0 or max_surfels < 1 or max_age_frames < 1 or not 0 <= min_confidence <= 1:
            raise ValueError("invalid surfel map limits")
        self.voxel_size = float(voxel_size)
        self.max_surfels = int(max_surfels)
        self.max_age_frames = int(max_age_frames)
        self.min_confidence = float(min_confidence)
        self.merge_distance = float(merge_distance if merge_distance is not None else voxel_size * 0.75)
        self.normal_merge_cos = float(normal_merge_cos)
        self._surfels: list[Surfel] = []
        self._buckets: dict[tuple[int, int, int], list[int]] = {}
        self.revision = 0
        self.last_stats: dict[str, float | int] = {"new_surfels": 0, "fused_surfels": 0, "evicted_surfels": 0}

    def _key(self, point: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.floor(np.asarray(point) / self.voxel_size).astype(np.int64).tolist())

    def _rebuild_buckets(self) -> None:
        self._buckets = {}
        for index, surfel in enumerate(self._surfels):
            self._buckets.setdefault(self._key(surfel.position_world), []).append(index)

    def _evict(self, frame_id: int) -> int:
        evicted = 0
        while len(self._surfels) > self.max_surfels:
            scores = np.asarray([s.confidence * math.exp(-max(0, frame_id - s.last_seen_frame) / max(self.max_age_frames, 1)) for s in self._surfels])
            del self._surfels[int(np.argmin(scores))]
            evicted += 1
            self._rebuild_buckets()
        return evicted

    def insert(self, positions_world: np.ndarray, normals_world: np.ndarray, confidence: np.ndarray, frame_id: int, *, stride: int = 4, static_confidence: np.ndarray | None = None) -> dict[str, int]:
        points = np.asarray(positions_world, dtype=np.float32)
        normals = np.asarray(normals_world, dtype=np.float32)
        conf = np.asarray(confidence, dtype=np.float32)
        if points.shape != (*conf.shape, 3) or normals.shape != points.shape:
            raise ValueError("surfel inputs must have matching (...,3) and (...) shapes")
        if stride < 1:
            raise ValueError("stride must be positive")
        static = np.ones_like(conf) if static_confidence is None else np.asarray(static_confidence, dtype=np.float32)
        if static.shape != conf.shape:
            raise ValueError("static_confidence must match confidence shape")
        valid = np.isfinite(points).all(axis=-1) & np.isfinite(normals).all(axis=-1) & np.isfinite(conf) & (conf >= self.min_confidence) & (static >= self.min_confidence)
        sample = np.zeros(valid.shape, dtype=bool)
        sample[::stride, ::stride] = True
        valid &= sample
        new_count = fused_count = 0
        for point, normal, weight in zip(points[valid], normals[valid], (conf * static)[valid]):
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1e-6:
                continue
            normal = normal / normal_norm
            key = self._key(point)
            match = None
            for index in self._buckets.get(key, []):
                candidate = self._surfels[index]
                if np.linalg.norm(candidate.position_world - point) <= self.merge_distance and float(np.dot(candidate.normal_world, normal)) >= self.normal_merge_cos:
                    match = candidate
                    break
            if match is None:
                self._surfels.append(Surfel(point.copy(), normal.astype(np.float32), float(np.clip(weight, 0, 1)), 1, int(frame_id)))
                self._buckets.setdefault(key, []).append(len(self._surfels) - 1)
                new_count += 1
            else:
                old_weight = max(match.confidence, 1e-4)
                alpha = float(np.clip(weight / (old_weight + weight), 0.05, 0.5))
                match.position_world = ((1 - alpha) * match.position_world + alpha * point).astype(np.float32)
                match.normal_world = ((1 - alpha) * match.normal_world + alpha * normal).astype(np.float32)
                match.normal_world /= max(float(np.linalg.norm(match.normal_world)), 1e-6)
                match.confidence = float(np.clip(match.confidence + alpha * (weight - match.confidence), 0.0, 1.0))
                match.observation_count = min(match.observation_count + 1, 65535)
                match.last_seen_frame = int(frame_id)
                fused_count += 1
        evicted = self._evict(frame_id)
        self.revision += 1
        self.last_stats = {"new_surfels": new_count, "fused_surfels": fused_count, "evicted_surfels": evicted}
        return {key: int(value) for key, value in self.last_stats.items()}

    def age_evict(self, frame_id: int) -> int:
        before = len(self._surfels)
        self._surfels = [s for s in self._surfels if frame_id - s.last_seen_frame <= self.max_age_frames and s.confidence >= self.min_confidence]
        self._rebuild_buckets()
        removed = before - len(self._surfels)
        if removed:
            self.revision += 1
        return removed

    @property
    def count(self) -> int:
        return len(self._surfels)

    def arrays(self) -> dict[str, np.ndarray]:
        if not self._surfels:
            return {"positions_world": np.empty((0, 3), np.float32), "normals_world": np.empty((0, 3), np.float32), "confidence": np.empty((0,), np.float32), "last_seen_frame": np.empty((0,), np.int32), "observation_count": np.empty((0,), np.int32)}
        return {
            "positions_world": np.stack([s.position_world for s in self._surfels]).astype(np.float32),
            "normals_world": np.stack([s.normal_world for s in self._surfels]).astype(np.float32),
            "confidence": np.asarray([s.confidence for s in self._surfels], dtype=np.float32),
            "last_seen_frame": np.asarray([s.last_seen_frame for s in self._surfels], dtype=np.int32),
            "observation_count": np.asarray([s.observation_count for s in self._surfels], dtype=np.int32),
        }

    def nbytes(self) -> int:
        arrays = self.arrays()
        return int(sum(array.nbytes for array in arrays.values()))

    def reproject(self, camera: CameraModel, pose: CameraPoseState) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shape = (camera.height, camera.width)
        depth = np.full(shape, np.nan, np.float32)
        positions = np.full((*shape, 3), np.nan, np.float32)
        normals = np.full((*shape, 3), np.nan, np.float32)
        confidence = np.zeros(shape, np.float32)
        if not pose.valid or not self._surfels:
            return depth, positions, normals, confidence
        arrays = self.arrays()
        world_to_camera = np.linalg.inv(np.asarray(pose.T_world_from_camera, dtype=np.float64))
        points_h = np.concatenate((arrays["positions_world"].astype(np.float64), np.ones((self.count, 1))), axis=1)
        camera_points = (world_to_camera @ points_h.T).T[:, :3]
        valid = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 0)
        pixels = camera.project(camera_points)
        inside = valid & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height)
        indices = np.flatnonzero(inside)
        if not len(indices):
            return depth, positions, normals, confidence
        px = np.rint(pixels[indices]).astype(np.int64)
        inside_round = (px[:, 0] >= 0) & (px[:, 0] < camera.width) & (px[:, 1] >= 0) & (px[:, 1] < camera.height)
        indices, px = indices[inside_round], px[inside_round]
        linear = px[:, 1] * camera.width + px[:, 0]
        order = np.lexsort((camera_points[indices, 2], linear))
        sorted_linear = linear[order]
        first = np.r_[True, sorted_linear[1:] != sorted_linear[:-1]]
        winners = indices[order[first]]
        winner_px = px[order[first]]
        yi, xi = winner_px[:, 1], winner_px[:, 0]
        depth[yi, xi] = camera_points[winners, 2].astype(np.float32)
        positions[yi, xi] = camera_points[winners].astype(np.float32)
        rotation = world_to_camera[:3, :3]
        transformed_normals = arrays["normals_world"] @ rotation.T
        transformed_normals /= np.maximum(np.linalg.norm(transformed_normals, axis=1, keepdims=True), 1e-6)
        normals[yi, xi] = transformed_normals[winners].astype(np.float32)
        age = np.maximum(0, int(pose.source_frame_id) - arrays["last_seen_frame"]) if isinstance(pose.source_frame_id, (int, np.integer)) else np.zeros(self.count)
        confidence[yi, xi] = (arrays["confidence"][winners] * np.exp(-age[winners] / max(self.max_age_frames, 1)) * float(pose.confidence)).astype(np.float32)
        return depth, positions, normals, confidence


class PersistentGeometryMapper:
    """Single-owner adapter from stable GeometryState to a SurfelMap."""

    def __init__(self, camera: CameraModel, surfel_map: SurfelMap | None = None, *, mapping_stride: int = 4, min_insert_confidence: float = 0.35, min_static_confidence: float = 0.5) -> None:
        self.camera = camera
        self.map = surfel_map or SurfelMap()
        self.mapping_stride = int(mapping_stride)
        self.min_insert_confidence = float(min_insert_confidence)
        self.min_static_confidence = float(min_static_confidence)
        self.world_from_camera: np.ndarray | None = None
        self.last_pose: CameraPoseState | None = None

    def reset(self) -> None:
        self.map = SurfelMap(voxel_size=self.map.voxel_size, max_surfels=self.map.max_surfels, max_age_frames=self.map.max_age_frames, min_confidence=self.map.min_confidence, merge_distance=self.map.merge_distance, normal_merge_cos=self.map.normal_merge_cos)
        self.world_from_camera = None
        self.last_pose = None

    def _pose_state(self, geometry: GeometryState, pose: PoseEstimateResult | CameraPoseState) -> CameraPoseState | None:
        if isinstance(pose, CameraPoseState):
            if pose.source_frame_id != geometry.source_frame_id:
                return None
            return pose if pose.valid else None
        if not pose.valid:
            return None
        if self.world_from_camera is None:
            world = np.eye(4, dtype=np.float32)
        else:
            world = compose_world_pose(self.world_from_camera, pose.T_current_from_previous)
        return CameraPoseState(geometry.timestamp, geometry.source_frame_id, world, True, pose.confidence, pose.inlier_count, pose.reprojection_error)

    def update(self, geometry: GeometryState, pose: PoseEstimateResult | CameraPoseState, *, static_confidence: np.ndarray | None = None) -> PersistentGeometryState | None:
        if geometry.camera != self.camera:
            self.camera = geometry.camera
        pose_state = self._pose_state(geometry, pose)
        if pose_state is None:
            return None
        self.world_from_camera = pose_state.T_world_from_camera.copy()
        self.last_pose = pose_state
        valid = geometry.valid_mask & (geometry.normal_valid_mask if geometry.normal_valid_mask is not None else False)
        confidence = np.asarray(geometry.confidence if geometry.confidence is not None else np.zeros_like(valid, dtype=np.float32), dtype=np.float32)
        static = np.ones_like(confidence) if static_confidence is None else np.asarray(static_confidence, dtype=np.float32)
        valid &= confidence >= self.min_insert_confidence
        valid &= static >= self.min_static_confidence
        points = np.asarray(geometry.positions_3d, dtype=np.float32)
        normals = np.asarray(geometry.normals, dtype=np.float32)
        world = np.asarray(pose_state.T_world_from_camera, dtype=np.float32)
        points_world = (world[:3, :3] @ points.reshape(-1, 3).T).T + world[:3, 3]
        normals_world = (world[:3, :3] @ normals.reshape(-1, 3).T).T
        points_world = points_world.reshape(points.shape)
        normals_world = normals_world.reshape(normals.shape)
        stats = self.map.insert(points_world, normals_world, confidence, int(geometry.source_frame_id) if isinstance(geometry.source_frame_id, (int, np.integer)) else 0, stride=self.mapping_stride, static_confidence=static)
        if isinstance(geometry.source_frame_id, (int, np.integer)):
            stats["age_evicted_surfels"] = self.map.age_evict(int(geometry.source_frame_id))
        depth, projected_positions, projected_normals, projected_confidence = self.map.reproject(self.camera, pose_state)
        projected_valid = np.isfinite(depth) & (depth > 0) & (projected_confidence > 0)
        return PersistentGeometryState(geometry.timestamp, geometry.source_frame_id, pose_state, depth, projected_positions, projected_normals, projected_confidence, projected_valid, self.map.count, self.map.revision, {**stats, "map_memory_bytes": self.map.nbytes()})


class AdvancedGeometryEngine:
    """Optional wrapper preserving the base update API and fail-safe fallback."""

    def __init__(self, base_engine, *, advanced_geometry_enabled: bool = False, mapper: PersistentGeometryMapper | None = None) -> None:
        self.base_engine = base_engine
        self.advanced_geometry_enabled = bool(advanced_geometry_enabled)
        self.mapper = mapper or PersistentGeometryMapper(base_engine.camera)
        self.last_persistent_state: PersistentGeometryState | None = None

    def update(self, *args, pose: PoseEstimateResult | CameraPoseState | None = None, static_confidence: np.ndarray | None = None, **kwargs) -> GeometryState:
        state = self.base_engine.update(*args, **kwargs)
        self.last_persistent_state = None
        if self.advanced_geometry_enabled and pose is not None:
            try:
                self.last_persistent_state = self.mapper.update(state, pose, static_confidence=static_confidence)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                self.last_persistent_state = None
        return state

    def update_with_persistent(self, *args, pose: PoseEstimateResult | CameraPoseState | None = None, static_confidence: np.ndarray | None = None, **kwargs) -> tuple[GeometryState, PersistentGeometryState | None]:
        state = self.update(*args, pose=pose, static_confidence=static_confidence, **kwargs)
        return state, self.last_persistent_state
