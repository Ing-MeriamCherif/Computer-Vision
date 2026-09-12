"""Bounded world-space surfel memory for the optional Phase 5 path.

The map is intentionally a separate, single-owner stateful object. The base
camera-space :class:`TemporalGeometryEngine` remains authoritative and can be
used without importing or enabling this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys

import numpy as np

from .camera import CameraModel
from .pose import (
    CameraPoseState,
    PoseConfig,
    PoseEstimateResult,
    PoseEstimator,
    classify_static_geometry,
    compose_world_pose,
)
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
    tracking_state: str = "TRACKING"
    map_updated: bool = False
    persistent_projection_valid: bool = False

    def __post_init__(self) -> None:
        shape = self.projected_depth.shape
        if len(shape) != 2:
            raise ValueError("projected_depth must have shape (H, W)")
        if self.projected_positions.shape != (*shape, 3):
            raise ValueError("projected_positions must have shape (H, W, 3)")
        if self.projected_normals.shape != (*shape, 3):
            raise ValueError("projected_normals must have shape (H, W, 3)")
        if self.projected_confidence.shape != shape:
            raise ValueError("projected_confidence must have shape (H, W)")
        if self.projected_valid.shape != shape:
            raise ValueError("projected_valid must have shape (H, W)")
        conf = np.asarray(self.projected_confidence, dtype=np.float32)
        if conf.size > 0:
            c_min = float(np.nanmin(conf))
            c_max = float(np.nanmax(conf))
            if c_min < -1e-5 or c_max > 1.0 + 1e-5:
                raise ValueError(f"projected_confidence must be in [0, 1], got [{c_min}, {c_max}]")


@dataclass(frozen=True, slots=True)
class PersistentGeometryConfig:
    """Tunable governor and parameters for Phase 5 persistent geometry."""
    voxel_size: float = 0.02
    relative_voxel_fraction: float = 0.01
    max_surfels: int = 100_000
    max_age_frames: int = 300
    min_confidence: float = 0.05
    merge_distance: float | None = None
    merge_distance_factor: float = 0.75
    normal_merge_cos: float = 0.8660254  # 30 degrees
    mapping_stride: int = 4
    min_insert_confidence: float = 0.35
    min_static_confidence: float = 0.5
    pose_max_samples: int = 1000
    rigid_residual_threshold: float = 1.5
    persistent_hole_fill_enabled: bool = False
    min_hole_fill_confidence: float = 0.5
    map_update_period: int = 1


@dataclass(slots=True)
class Surfel:
    position_world: np.ndarray
    normal_world: np.ndarray
    confidence: float
    observation_count: int
    last_seen_frame: int
    color: np.ndarray | None = None


# Precomputed 27-cell neighbor offsets
_NEIGHBOR_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
)


class SurfelMap:
    """A deterministic voxel-bucketed, bounded surfel map with 27-cell neighbor search."""

    def __init__(
        self,
        *,
        voxel_size: float = 0.02,
        max_surfels: int = 100_000,
        max_age_frames: int = 300,
        min_confidence: float = 0.05,
        merge_distance: float | None = None,
        merge_distance_factor: float = 0.75,
        normal_merge_cos: float = 0.8660254,
    ) -> None:
        if voxel_size <= 0 or max_surfels < 1 or max_age_frames < 1 or not 0 <= min_confidence <= 1:
            raise ValueError("invalid surfel map limits")
        self.voxel_size = float(voxel_size)
        self.resolved_voxel_size = float(voxel_size)
        self.scene_scale: float | None = None
        self.scale_mode: str = "metric"
        self.max_surfels = int(max_surfels)
        self.max_age_frames = int(max_age_frames)
        self.min_confidence = float(min_confidence)
        self.merge_distance_factor = float(merge_distance_factor)
        self.merge_distance = float(merge_distance if merge_distance is not None else voxel_size * self.merge_distance_factor)
        self.normal_merge_cos = float(normal_merge_cos)
        self._surfels: list[Surfel] = []
        self._buckets: dict[tuple[int, int, int], list[int]] = {}
        self.revision = 0
        self._cached_arrays: dict[str, np.ndarray] | None = None
        self._cached_revision: int = -1
        self.last_stats: dict[str, float | int] = {"new_surfels": 0, "fused_surfels": 0, "evicted_surfels": 0}

    def _key(self, point: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.floor(np.asarray(point) / self.voxel_size).astype(np.int64).tolist())

    def _invalidate_cache(self) -> None:
        self._cached_arrays = None
        self._cached_revision = -1

    def _rebuild_buckets(self) -> None:
        self._buckets = {}
        for index, surfel in enumerate(self._surfels):
            self._buckets.setdefault(self._key(surfel.position_world), []).append(index)

    def validate_index(self) -> bool:
        """Verify bucket index consistency and bidirectional integrity."""
        all_indices: list[int] = []
        n_surfels = len(self._surfels)
        for key, indices in self._buckets.items():
            for idx in indices:
                if not (0 <= idx < n_surfels):
                    raise ValueError(f"Bucket {key} contains invalid index {idx} (total surfels {n_surfels})")
                expected_key = self._key(self._surfels[idx].position_world)
                if expected_key != key:
                    raise ValueError(
                        f"Surfel {idx} at {self._surfels[idx].position_world} has key {expected_key} but is in bucket {key}"
                    )
                all_indices.append(idx)
        if len(all_indices) != n_surfels:
            raise ValueError(f"Total bucket indices {len(all_indices)} does not match surfel count {n_surfels}")
        if len(set(all_indices)) != n_surfels:
            raise ValueError("Duplicate surfel indices found across buckets")
        return True

    def _evict(self, frame_id: int) -> int:
        """Batch-evict lowest-scoring excess surfels with a single bucket rebuild."""
        excess = len(self._surfels) - self.max_surfels
        if excess <= 0:
            return 0
        ages = np.maximum(0, frame_id - np.asarray([s.last_seen_frame for s in self._surfels], dtype=np.float32))
        confs = np.asarray([s.confidence for s in self._surfels], dtype=np.float32)
        tau = max(float(self.max_age_frames), 1.0)
        scores = confs * np.exp(-ages / tau)
        # argpartition finds the 'excess' smallest score indices in O(N)
        evict_indices = set(np.argpartition(scores, excess)[:excess].tolist())
        self._surfels = [s for i, s in enumerate(self._surfels) if i not in evict_indices]
        self._rebuild_buckets()
        self._invalidate_cache()
        return excess

    def insert(
        self,
        positions_world: np.ndarray,
        normals_world: np.ndarray,
        confidence: np.ndarray,
        frame_id: int,
        *,
        stride: int = 4,
        static_confidence: np.ndarray | None = None,
        min_static_confidence: float = 0.5,
        valid_mask: np.ndarray | None = None,
    ) -> dict[str, int]:
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
        valid = (
            np.isfinite(points).all(axis=-1)
            & np.isfinite(normals).all(axis=-1)
            & np.isfinite(conf)
            & (conf >= self.min_confidence)
            & (static >= float(min_static_confidence))
        )
        if valid_mask is not None:
            valid &= np.asarray(valid_mask, dtype=bool)
        sample = np.zeros(valid.shape, dtype=bool)
        if valid.ndim == 2:
            sample[::stride, ::stride] = True
        else:
            sample[::stride] = True
        valid &= sample
        new_count = fused_count = 0

        for point, normal, weight in zip(points[valid], normals[valid], (conf * static)[valid]):
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1e-6:
                continue
            normal = normal / normal_norm
            base_key = self._key(point)

            # 27-cell neighborhood search
            best_match: Surfel | None = None
            best_idx: int = -1
            best_score = float("inf")
            best_old_key: tuple[int, int, int] | None = None

            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                neigh_key = (base_key[0] + dx, base_key[1] + dy, base_key[2] + dz)
                bucket_indices = self._buckets.get(neigh_key)
                if not bucket_indices:
                    continue
                for index in bucket_indices:
                    candidate = self._surfels[index]
                    dist = float(np.linalg.norm(candidate.position_world - point))
                    if dist <= self.merge_distance:
                        cos_sim = float(np.dot(candidate.normal_world, normal))
                        if cos_sim >= self.normal_merge_cos:
                            # Primary: minimum spatial distance. Secondary tie-breaker: highest normal similarity
                            score = dist - 1e-4 * cos_sim
                            if score < best_score:
                                best_score = score
                                best_match = candidate
                                best_idx = index
                                best_old_key = neigh_key

            if best_match is None:
                new_surfel = Surfel(
                    point.copy(),
                    normal.astype(np.float32),
                    float(np.clip(weight, 0.0, 1.0)),
                    1,
                    int(frame_id),
                )
                self._surfels.append(new_surfel)
                self._buckets.setdefault(base_key, []).append(len(self._surfels) - 1)
                new_count += 1
            else:
                old_count = best_match.observation_count
                alpha = float(np.clip(weight / (max(best_match.confidence, 1e-4) * old_count + weight), 0.01, 0.5))
                old_pos_key = self._key(best_match.position_world)

                # Fuse position and normal
                best_match.position_world = ((1.0 - alpha) * best_match.position_world + alpha * point).astype(np.float32)
                best_match.normal_world = ((1.0 - alpha) * best_match.normal_world + alpha * normal).astype(np.float32)
                best_match.normal_world /= max(float(np.linalg.norm(best_match.normal_world)), 1e-6)

                # Bounded confidence accumulation (saturates towards 1.0, never unexpected drop)
                w_clip = float(np.clip(weight, 0.0, 1.0))
                new_conf = 1.0 - (1.0 - best_match.confidence) * (1.0 - 0.5 * w_clip)
                best_match.confidence = float(np.clip(new_conf, 0.0, 1.0))
                best_match.observation_count = min(best_match.observation_count + 1, 65535)
                best_match.last_seen_frame = int(frame_id)
                fused_count += 1

                # Bucket migration if fused position crossed a voxel boundary
                new_pos_key = self._key(best_match.position_world)
                if new_pos_key != old_pos_key:
                    if old_pos_key in self._buckets and best_idx in self._buckets[old_pos_key]:
                        self._buckets[old_pos_key].remove(best_idx)
                        if not self._buckets[old_pos_key]:
                            del self._buckets[old_pos_key]
                    self._buckets.setdefault(new_pos_key, []).append(best_idx)

        evicted = self._evict(frame_id)
        if new_count > 0 or fused_count > 0 or evicted > 0:
            self.revision += 1
            self._invalidate_cache()

        self.last_stats = {"new_surfels": new_count, "fused_surfels": fused_count, "evicted_surfels": evicted}
        return {key: int(value) for key, value in self.last_stats.items()}

    def age_evict(self, frame_id: int) -> int:
        before = len(self._surfels)
        self._surfels = [
            s for s in self._surfels
            if frame_id - s.last_seen_frame <= self.max_age_frames and s.confidence >= self.min_confidence
        ]
        removed = before - len(self._surfels)
        if removed > 0:
            self._rebuild_buckets()
            self.revision += 1
            self._invalidate_cache()
        return removed

    @property
    def count(self) -> int:
        return len(self._surfels)

    def arrays(self) -> dict[str, np.ndarray]:
        """Return packed NumPy arrays of map contents, cached by revision."""
        if self._cached_arrays is not None and self._cached_revision == self.revision:
            return self._cached_arrays

        if not self._surfels:
            res = {
                "positions_world": np.empty((0, 3), np.float32),
                "normals_world": np.empty((0, 3), np.float32),
                "confidence": np.empty((0,), np.float32),
                "last_seen_frame": np.empty((0,), np.int32),
                "observation_count": np.empty((0,), np.int32),
            }
        else:
            res = {
                "positions_world": np.stack([s.position_world for s in self._surfels]).astype(np.float32),
                "normals_world": np.stack([s.normal_world for s in self._surfels]).astype(np.float32),
                "confidence": np.asarray([s.confidence for s in self._surfels], dtype=np.float32),
                "last_seen_frame": np.asarray([s.last_seen_frame for s in self._surfels], dtype=np.int32),
                "observation_count": np.asarray([s.observation_count for s in self._surfels], dtype=np.int32),
            }
        self._cached_arrays = res
        self._cached_revision = self.revision
        return res

    def packed_array_bytes(self) -> int:
        """Packed NumPy byte size of all map arrays."""
        arrs = self.arrays()
        return int(sum(array.nbytes for array in arrs.values()))

    def approx_python_bytes(self) -> int:
        """Estimated total process memory including Python object overhead."""
        # ~160 bytes per Surfel object + dict/list references
        per_surfel = 160 + sys.getsizeof(0) * 2
        heap = len(self._surfels) * per_surfel + sys.getsizeof(self._buckets) + sys.getsizeof(self._surfels)
        return int(self.packed_array_bytes() + heap)

    def nbytes(self) -> int:
        return self.packed_array_bytes()

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

    def __init__(
        self,
        camera: CameraModel,
        surfel_map: SurfelMap | None = None,
        *,
        config: PersistentGeometryConfig | None = None,
        mapping_stride: int = 4,
        min_insert_confidence: float = 0.35,
        min_static_confidence: float = 0.5,
    ) -> None:
        self.camera = camera
        self.config = config or PersistentGeometryConfig()
        stride = self.config.mapping_stride if config is not None else mapping_stride
        min_insert = self.config.min_insert_confidence if config is not None else min_insert_confidence
        min_static = self.config.min_static_confidence if config is not None else min_static_confidence
        self.map = surfel_map or SurfelMap(
            voxel_size=self.config.voxel_size,
            max_surfels=self.config.max_surfels,
            max_age_frames=self.config.max_age_frames,
            min_confidence=self.config.min_confidence,
            merge_distance=self.config.merge_distance,
            merge_distance_factor=self.config.merge_distance_factor,
            normal_merge_cos=self.config.normal_merge_cos,
        )
        self.mapping_stride = int(stride)
        self.min_insert_confidence = float(min_insert)
        self.min_static_confidence = float(min_static)
        self.world_from_camera: np.ndarray | None = None
        self.last_pose: CameraPoseState | None = None
        self.tracking_state: str = "UNINITIALIZED"
        self.scale_mode: str | None = None

    def reset(self) -> None:
        self.map = SurfelMap(
            voxel_size=self.map.voxel_size,
            max_surfels=self.map.max_surfels,
            max_age_frames=self.map.max_age_frames,
            min_confidence=self.map.min_confidence,
            merge_distance=self.map.merge_distance,
            merge_distance_factor=self.map.merge_distance_factor,
            normal_merge_cos=self.map.normal_merge_cos,
        )
        self.world_from_camera = None
        self.last_pose = None
        self.tracking_state = "UNINITIALIZED"
        self.scale_mode = None

    def _init_relative_scale(self, geometry: GeometryState) -> None:
        """Derive and freeze relative-mode scene scale once on session start."""
        if self.scale_mode is not None:
            return
        geom_mode = getattr(geometry.scale_mode, "value", str(geometry.scale_mode))
        if geom_mode == "relative":
            depths = geometry.positions_3d[..., 2]
            valid = geometry.valid_mask & np.isfinite(depths) & (depths > 0)
            if valid.any():
                med_depth = float(np.nanmedian(depths[valid]))
                if med_depth > 0:
                    self.map.scene_scale = med_depth
                    self.map.voxel_size = med_depth * self.config.relative_voxel_fraction
                    self.map.merge_distance = self.map.voxel_size * self.config.merge_distance_factor
                    self.map.resolved_voxel_size = self.map.voxel_size
                    self.map.scale_mode = "relative"
                    self.scale_mode = "relative"
                    return
        self.map.scale_mode = "metric"
        self.map.resolved_voxel_size = self.config.voxel_size
        self.scale_mode = "metric"

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
        return CameraPoseState(
            geometry.timestamp,
            geometry.source_frame_id,
            world,
            True,
            pose.confidence,
            pose.inlier_count,
            pose.reprojection_error,
        )

    def update(
        self,
        geometry: GeometryState,
        pose: PoseEstimateResult | CameraPoseState,
        *,
        static_confidence: np.ndarray | None = None,
    ) -> PersistentGeometryState | None:
        if geometry.camera != self.camera:
            self.camera = geometry.camera

        self._init_relative_scale(geometry)
        geom_mode = getattr(geometry.scale_mode, "value", str(geometry.scale_mode))
        if self.scale_mode is not None and geom_mode != self.scale_mode:
            # Scale mode contract: refuse mixed metric/relative updates
            self.tracking_state = "LOST"
            return None

        pose_state = self._pose_state(geometry, pose)
        if pose_state is None:
            self.tracking_state = "LOST"
            return None

        self.tracking_state = "TRACKING"
        self.world_from_camera = pose_state.T_world_from_camera.copy()
        self.last_pose = pose_state
        valid = geometry.valid_mask & (geometry.normal_valid_mask if geometry.normal_valid_mask is not None else False)
        confidence = np.asarray(
            geometry.confidence if geometry.confidence is not None else np.zeros_like(valid, dtype=np.float32),
            dtype=np.float32,
        )
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

        stats = self.map.insert(
            points_world,
            normals_world,
            confidence,
            int(geometry.source_frame_id) if isinstance(geometry.source_frame_id, (int, np.integer)) else 0,
            stride=self.mapping_stride,
            static_confidence=static,
            min_static_confidence=self.min_static_confidence,
            valid_mask=valid,
        )
        if isinstance(geometry.source_frame_id, (int, np.integer)):
            stats["age_evicted_surfels"] = self.map.age_evict(int(geometry.source_frame_id))

        depth, projected_positions, projected_normals, projected_confidence = self.map.reproject(self.camera, pose_state)
        projected_valid = np.isfinite(depth) & (depth > 0) & (projected_confidence > 0)

        map_updated = bool(stats.get("new_surfels", 0) > 0 or stats.get("fused_surfels", 0) > 0 or stats.get("evicted_surfels", 0) > 0)
        proj_valid = bool(projected_valid.any())

        return PersistentGeometryState(
            geometry.timestamp,
            geometry.source_frame_id,
            pose_state,
            depth,
            projected_positions,
            projected_normals,
            projected_confidence,
            projected_valid,
            self.map.count,
            self.map.revision,
            {**stats, "map_memory_bytes": self.map.nbytes()},
            tracking_state=self.tracking_state,
            map_updated=map_updated,
            persistent_projection_valid=proj_valid,
        )


def persistent_hole_fill(
    geometry: GeometryState,
    persistent: PersistentGeometryState,
    *,
    min_persistent_confidence: float = 0.5,
) -> GeometryState:
    """Optionally fill missing/invalid geometry using confident persistent reprojection."""
    if persistent is None or not persistent.pose.valid:
        return geometry

    geom_valid = np.asarray(geometry.valid_mask, dtype=bool)
    p_valid = np.asarray(persistent.projected_valid, dtype=bool)
    p_conf = np.asarray(persistent.projected_confidence, dtype=np.float32)

    fill_mask = (~geom_valid) & p_valid & (p_conf >= float(min_persistent_confidence))
    if not fill_mask.any():
        return geometry

    positions = np.array(geometry.positions_3d, copy=True)
    normals = np.array(geometry.normals, copy=True)
    valid_mask = np.array(geom_valid, copy=True)
    normal_valid_mask = (
        np.array(geometry.normal_valid_mask, copy=True)
        if geometry.normal_valid_mask is not None
        else np.zeros_like(geom_valid)
    )
    confidence = (
        np.array(geometry.confidence, copy=True)
        if geometry.confidence is not None
        else np.zeros_like(geom_valid, dtype=np.float32)
    )
    depth = np.array(geometry.depth, copy=True) if geometry.depth is not None else positions[..., 2].copy()

    positions[fill_mask] = persistent.projected_positions[fill_mask]
    normals[fill_mask] = persistent.projected_normals[fill_mask]
    depth[fill_mask] = persistent.projected_depth[fill_mask]
    confidence[fill_mask] = persistent.projected_confidence[fill_mask] * 0.9
    valid_mask[fill_mask] = True
    normal_valid_mask[fill_mask] = True

    return GeometryState(
        timestamp=geometry.timestamp,
        source_frame_id=geometry.source_frame_id,
        camera=geometry.camera,
        positions_3d=positions,
        valid_mask=valid_mask,
        scale_mode=geometry.scale_mode,
        normals=normals,
        normal_valid_mask=normal_valid_mask,
        selected_radius=geometry.selected_radius,
        spatial_confidence=geometry.spatial_confidence,
        confidence=confidence,
        temporal_confidence=geometry.temporal_confidence,
        temporal_age=geometry.temporal_age,
        depth=depth,
    )


class AdvancedGeometryEngine:
    """Optional wrapper providing end-to-end automatic flow→pose→map with fail-safe fallback."""

    def __init__(
        self,
        base_engine,
        *,
        advanced_geometry_enabled: bool = False,
        config: PersistentGeometryConfig | None = None,
        mapper: PersistentGeometryMapper | None = None,
        persistent_hole_fill_enabled: bool = False,
    ) -> None:
        self.base_engine = base_engine
        self.advanced_geometry_enabled = bool(advanced_geometry_enabled)
        self.config = config or PersistentGeometryConfig(
            persistent_hole_fill_enabled=persistent_hole_fill_enabled
        )
        self.mapper = mapper or PersistentGeometryMapper(base_engine.camera, config=self.config)
        self.persistent_hole_fill_enabled = bool(
            self.config.persistent_hole_fill_enabled or persistent_hole_fill_enabled
        )
        self.pose_estimator = PoseEstimator(
            base_engine.camera,
            max_samples=self.config.pose_max_samples,
            reprojection_error=3.0,
        )
        self.last_persistent_state: PersistentGeometryState | None = None

    def update(
        self,
        *args,
        pose: PoseEstimateResult | CameraPoseState | None = None,
        static_confidence: np.ndarray | None = None,
        **kwargs,
    ) -> GeometryState:
        # Cache previous geometry before base engine overwrites it
        prev_geom = getattr(self.base_engine, "previous_state", None)

        state = self.base_engine.update(*args, **kwargs)
        self.last_persistent_state = None

        if not self.advanced_geometry_enabled:
            return state

        # Case 1: Manual pose injection for debug/tests
        if pose is not None:
            try:
                self.last_persistent_state = self.mapper.update(state, pose, static_confidence=static_confidence)
                if self.persistent_hole_fill_enabled and self.last_persistent_state is not None:
                    state = persistent_hole_fill(
                        state,
                        self.last_persistent_state,
                        min_persistent_confidence=self.config.min_hole_fill_confidence,
                    )
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                self.last_persistent_state = None
            return state

        # Case 2: Automatic pipeline
        try:
            # First frame with valid geometry: initialize world pose at origin
            if self.mapper.world_from_camera is None:
                if state.valid_mask.any():
                    init_pose = CameraPoseState(
                        timestamp=state.timestamp,
                        source_frame_id=state.source_frame_id,
                        T_world_from_camera=np.eye(4, dtype=np.float32),
                        valid=True,
                        confidence=1.0,
                        inlier_count=100,
                        reprojection_error=0.0,
                    )
                    self.last_persistent_state = self.mapper.update(state, init_pose)
                return state

            # Subsequent frames: extract correspondences and estimate pose
            motion = getattr(self.base_engine, "motion_state", None)
            if (
                prev_geom is not None
                and motion is not None
                and motion.source_frame_id == prev_geom.source_frame_id
                and prev_geom.valid_mask.any()
            ):
                pose_result = self.pose_estimator.estimate_from_states(prev_geom, motion)
                if pose_result.valid:
                    static_res = classify_static_geometry(
                        prev_geom,
                        motion,
                        pose_result,
                        camera=self.base_engine.camera,
                        threshold=self.config.rigid_residual_threshold,
                        min_static_confidence=self.config.min_static_confidence,
                    )
                    self.last_persistent_state = self.mapper.update(
                        state,
                        pose_result,
                        static_confidence=static_res.static_confidence,
                    )
                    if self.persistent_hole_fill_enabled and self.last_persistent_state is not None:
                        state = persistent_hole_fill(
                            state,
                            self.last_persistent_state,
                            min_persistent_confidence=self.config.min_hole_fill_confidence,
                        )
                else:
                    self.mapper.tracking_state = "LOST"
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            self.last_persistent_state = None

        return state

    def update_with_persistent(
        self,
        *args,
        pose: PoseEstimateResult | CameraPoseState | None = None,
        static_confidence: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[GeometryState, PersistentGeometryState | None]:
        state = self.update(*args, pose=pose, static_confidence=static_confidence, **kwargs)
        return state, self.last_persistent_state


def compute_dynamic_contamination(
    positions_world: np.ndarray,
    dynamic_bounds_min: np.ndarray,
    dynamic_bounds_max: np.ndarray,
) -> float:
    """Calculate percentage of surfels residing inside a forbidden dynamic bounding box."""
    if len(positions_world) == 0:
        return 0.0
    pts = np.asarray(positions_world, dtype=np.float32)
    b_min = np.asarray(dynamic_bounds_min, dtype=np.float32)
    b_max = np.asarray(dynamic_bounds_max, dtype=np.float32)
    inside = (pts >= b_min).all(axis=1) & (pts <= b_max).all(axis=1)
    return float(inside.mean() * 100.0)


def compute_reprojection_metrics(
    projected_depth: np.ndarray,
    ground_truth_depth: np.ndarray,
    eval_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate mean, median, p95 relative depth error and coverage %."""
    proj = np.asarray(projected_depth, dtype=np.float32)
    gt = np.asarray(ground_truth_depth, dtype=np.float32)
    valid = np.isfinite(proj) & (proj > 0) & np.isfinite(gt) & (gt > 0)
    if eval_mask is not None:
        valid &= np.asarray(eval_mask, dtype=bool)
    if not valid.any():
        return {
            "mean_error": float("inf"),
            "median_error": float("inf"),
            "p95_error": float("inf"),
            "coverage_percent": 0.0,
        }
    rel_err = np.abs(proj[valid] - gt[valid]) / gt[valid]
    return {
        "mean_error": float(np.mean(rel_err)),
        "median_error": float(np.median(rel_err)),
        "p95_error": float(np.percentile(rel_err, 95)),
        "coverage_percent": float(valid.mean() * 100.0),
    }
