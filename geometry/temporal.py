"""Stateful short-term temporal reconstruction for camera-space geometry.

This module stabilizes geometry in image/camera space only. It does not assume
camera pose, persistent world coordinates, or a renderer temporal accumulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .alignment import DepthAlignmentResult, align_history_depth, align_inverse_depth
from .backproject import DepthScaleMode, backproject_depth
from .camera import CameraModel
from .diagnostics import GeometryDiagnostics
from .motion import MotionState, OpticalFlowProvider
from .normals import NormalConfig, NormalMode, estimate_normals
from .state import DepthState, GeometryState
from .warp import warp_depth_backward, warp_field_backward


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    flow_fb_threshold: float = 1.5
    photometric_threshold: float = 0.25
    depth_disagreement_threshold: float = 0.25
    depth_warp_discontinuity_threshold: float = 0.15
    history_weight_max: float = 0.85
    history_decay: float = 0.92
    max_history_age: int = 8
    history_min_confidence: float = 0.05
    alignment_min_samples: int = 64
    alignment_max_samples: int | None = 10000
    alignment_residual_threshold: float = 0.04
    alignment_epsilon: float = 1e-6
    timestamp_gap_reset: float = 0.5
    depth_timestamp_tolerance: float = 0.1
    diagnostics_level: str = "none"  # none, basic, timing
    normal_mode: NormalMode | str = NormalMode.EDGE_AWARE

    def __post_init__(self) -> None:
        if self.flow_fb_threshold <= 0:
            raise ValueError("flow_fb_threshold must be positive")
        if self.photometric_threshold <= 0 or self.depth_disagreement_threshold <= 0:
            raise ValueError("consistency thresholds must be positive")
        if self.depth_warp_discontinuity_threshold <= 0:
            raise ValueError("depth_warp_discontinuity_threshold must be positive")
        if not 0 <= self.history_weight_max <= 1 or not 0 < self.history_decay <= 1:
            raise ValueError("history weights/decay must be in valid ranges")
        if self.max_history_age < 1 or not 0 <= self.history_min_confidence <= 1:
            raise ValueError("history age/confidence settings are invalid")
        if self.alignment_min_samples < 2 or (self.alignment_max_samples is not None and self.alignment_max_samples < self.alignment_min_samples) or self.alignment_residual_threshold <= 0 or self.alignment_epsilon <= 0:
            raise ValueError("alignment settings are invalid")
        if self.timestamp_gap_reset <= 0:
            raise ValueError("timestamp_gap_reset must be positive")
        if self.depth_timestamp_tolerance < 0:
            raise ValueError("depth_timestamp_tolerance must be non-negative")
        if self.diagnostics_level not in {"none", "basic", "timing"}:
            raise ValueError("diagnostics_level must be 'none', 'basic', or 'timing'")
        object.__setattr__(self, "normal_mode", NormalMode(self.normal_mode))


def _gray(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 3:
        array = array.astype(np.float32).mean(axis=-1)
    if array.ndim != 2:
        raise ValueError("RGB frames must have shape (H, W) or (H, W, C)")
    array = array.astype(np.float32)
    if array.size and np.nanmax(array) > 1.0:
        array /= 255.0
    return np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)


def photometric_error(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    backward_flow: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute brightness-robust normalized RGB/grayscale mismatch."""

    previous = _gray(previous_frame)
    current = _gray(current_frame)
    if previous.shape != current.shape or previous.shape != backward_flow.shape[:2]:
        raise ValueError("frames and backward_flow must share resolution")
    warped, valid = warp_field_backward(previous, backward_flow, interpolation="bilinear", fill_value=np.nan)
    difference = np.abs(current - warped)
    # Remove a global exposure offset before measuring residual mismatch.
    offset = np.nanmedian(np.where(valid, current - warped, np.nan))
    difference = np.abs(current - (warped + (0.0 if not np.isfinite(offset) else offset)))
    return np.where(valid, difference, np.inf).astype(np.float32), valid


def _camera_compatible(a: CameraModel, b: CameraModel) -> bool:
    return (a.width, a.height) == (b.width, b.height) and np.allclose(
        [a.fx, a.fy, a.cx, a.cy], [b.fx, b.fy, b.cx, b.cy], rtol=1e-5, atol=1e-5
    )


class TemporalGeometryEngine:
    """Fuse fresh or absent depth observations into short-lived stable geometry."""

    def __init__(
        self,
        camera: CameraModel,
        config: TemporalConfig | None = None,
        flow_provider: OpticalFlowProvider | None = None,
        normal_config: NormalConfig | None = None,
    ) -> None:
        self.camera = camera
        self.config = config or TemporalConfig()
        self.flow_provider = flow_provider
        self.normal_config = normal_config
        self.previous_state: GeometryState | None = None
        self.previous_rgb: np.ndarray | None = None
        self.previous_timestamp: float | None = None
        self.previous_frame_id: int | str | None = None
        self.previous_processing_frame_id: int | str | None = None
        self.motion_state: MotionState | None = None
        self.last_alignment: DepthAlignmentResult | None = None
        self.last_diagnostics: GeometryDiagnostics | None = None
        self._pending_reset_reason: str | None = None
        self.flow_failures: int = 0

    @property
    def current_state(self) -> GeometryState | None:
        return self.previous_state

    def reset(self, reason: str = "manual") -> None:
        self.previous_state = None
        self.previous_rgb = None
        self.previous_timestamp = None
        self.previous_frame_id = None
        self.previous_processing_frame_id = None
        self.motion_state = None
        self.last_alignment = None
        self._pending_reset_reason = reason

    def _needs_reset(
        self,
        camera: CameraModel,
        frame_id: int | str,
        timestamp: float,
        scale_mode: DepthScaleMode | None,
        processing_frame_id: int | str | None = None,
    ) -> str | None:
        if self.previous_state is None:
            return None
        if not _camera_compatible(self.camera, camera):
            return "camera_changed"
        current_processing = frame_id if processing_frame_id is None else processing_frame_id
        previous_processing = self.previous_processing_frame_id
        if isinstance(current_processing, int) and isinstance(previous_processing, int) and current_processing != previous_processing + 1:
            return "frame_discontinuity"
        if self.previous_timestamp is not None and (
            timestamp <= self.previous_timestamp or timestamp - self.previous_timestamp > self.config.timestamp_gap_reset
        ):
            return "timestamp_reversal" if timestamp <= self.previous_timestamp else "timestamp_gap"
        if scale_mode is not None and scale_mode != self.previous_state.scale_mode:
            return "scale_mode_changed"
        return None

    def _empty_state(
        self,
        frame_id: int | str,
        timestamp: float,
        camera: CameraModel,
        scale_mode: DepthScaleMode,
        processing_frame_id: int | str | None = None,
    ) -> GeometryState:
        shape = (camera.height, camera.width)
        depth = np.full(shape, np.nan, dtype=np.float32)
        positions = np.full((*shape, 3), np.nan, dtype=np.float32)
        zeros = np.zeros(shape, dtype=np.float32)
        return GeometryState(
            timestamp=timestamp,
            source_frame_id=frame_id,
            depth=depth,
            positions_3d=positions,
            valid_mask=np.zeros(shape, dtype=bool),
            camera=camera,
            scale_mode=scale_mode,
            normals=np.full((*shape, 3), np.nan, dtype=np.float32),
            confidence=zeros,
            temporal_age=np.zeros(shape, dtype=np.uint16),
            history_valid=np.zeros(shape, dtype=bool),
            occlusion_mask=np.zeros(shape, dtype=bool),
            normal_valid_mask=np.zeros(shape, dtype=bool),
            normal_confidence=zeros,
            selected_radius=np.zeros(shape, dtype=np.int16),
            spatial_confidence=zeros,
            history_confidence=zeros,
            temporal_confidence=zeros,
            depth_alignment_residual=np.full(shape, np.inf, dtype=np.float32),
            history_rejection_mask=np.zeros(shape, dtype=bool),
            disocclusion_mask=np.zeros(shape, dtype=bool),
            processing_frame_id=frame_id if processing_frame_id is None else processing_frame_id,
        )

    def _get_motion(
        self,
        rgb_frame: np.ndarray | None,
        frame_id: int | str,
        timestamp: float,
        injected_motion: MotionState | None,
    ) -> MotionState | None:
        if self.previous_state is None:
            return None
        if injected_motion is not None:
            if injected_motion.source_frame_id != self.previous_frame_id or injected_motion.target_frame_id != frame_id:
                raise ValueError("MotionState frame IDs must match previous and current update IDs")
            return injected_motion
        if self.flow_provider is None or self.previous_rgb is None or rgb_frame is None:
            return None
        try:
            return self.flow_provider.compute(self.previous_rgb, rgb_frame, self.previous_frame_id, frame_id, timestamp)
        except RuntimeError:
            # Catastrophic provider failure safely rejects history for this update.
            self.flow_failures += 1
            return None

    def _history_inputs(
        self,
        motion: MotionState,
        current_rgb: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.previous_state is not None
        previous = self.previous_state

        # 1. Depth-aware warping across discontinuities
        warped_depth, warped_valid, depth_warp_conf = warp_depth_backward(
            previous.depth,
            motion.backward_flow,
            previous.valid_mask,
            discontinuity_threshold=self.config.depth_warp_discontinuity_threshold,
            epsilon=self.config.alignment_epsilon,
        )

        previous_confidence = previous.temporal_confidence
        if previous_confidence is None:
            previous_confidence = previous.confidence
        if previous_confidence is None:
            previous_confidence = np.ones_like(previous.valid_mask, dtype=np.float32)

        warped_confidence, confidence_valid = warp_field_backward(
            previous_confidence, motion.backward_flow, previous.valid_mask, "bilinear", 0.0
        )
        warped_age, _ = warp_field_backward(
            previous.temporal_age.astype(np.float32) if previous.temporal_age is not None else np.zeros_like(previous.valid_mask, dtype=np.float32),
            motion.backward_flow,
            previous.valid_mask,
            "nearest",
            0.0,
        )

        # 2. Flow confidence: single source of truth (no duplicate exponential)
        flow_confidence, flow_valid = warp_field_backward(
            motion.flow_confidence, motion.backward_flow, interpolation="bilinear", fill_value=0.0
        )
        flow_error, flow_error_valid = warp_field_backward(
            motion.forward_backward_error, motion.backward_flow, interpolation="bilinear", fill_value=np.inf
        )
        flow_valid &= flow_error_valid & (flow_error <= self.config.flow_fb_threshold)

        # Combine previous confidence, flow confidence, and depth warp confidence
        history_confidence = np.clip(
            np.nan_to_num(warped_confidence) * np.nan_to_num(flow_confidence) * np.nan_to_num(depth_warp_conf),
            0.0,
            1.0,
        )
        history_valid = warped_valid & confidence_valid & flow_valid & (history_confidence >= self.config.history_min_confidence)

        if motion.occlusion_mask is not None:
            history_valid &= ~motion.occlusion_mask

        if current_rgb is not None and self.previous_rgb is not None:
            photo, photo_valid = photometric_error(self.previous_rgb, current_rgb, motion.backward_flow)
            history_confidence *= np.where(photo_valid, np.exp(-((photo / self.config.photometric_threshold) ** 2)), 0.0)
            history_valid &= (
                photo_valid
                & (photo <= self.config.photometric_threshold * 3.0)
                & (history_confidence >= self.config.history_min_confidence)
            )

        raw_age = np.nan_to_num(warped_age).astype(np.int32) + 1
        return warped_depth, history_valid, history_confidence.astype(np.float32), raw_age.astype(np.uint16), flow_valid

    def update(
        self,
        rgb_frame: np.ndarray | None,
        camera: CameraModel,
        frame_id: int | str,
        timestamp: float,
        depth_state: DepthState | None = None,
        motion_state: MotionState | None = None,
        processing_frame_id: int | str | None = None,
    ) -> GeometryState:
        """Process one camera frame; ``depth_state=None`` propagates valid history."""
        timing_enabled = self.config.diagnostics_level == "timing"
        processing_id = frame_id if processing_frame_id is None else processing_frame_id
        total_start = time.perf_counter() if timing_enabled else 0.0
        stage_ms: dict[str, float] = {}
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if depth_state is not None:
            if depth_state.source_frame_id != frame_id:
                raise ValueError(
                    f"DepthState.source_frame_id {depth_state.source_frame_id!r} does not match current frame_id {frame_id!r}"
                )
            if not np.isfinite(depth_state.timestamp):
                raise ValueError("DepthState.timestamp must be finite")
            if abs(float(depth_state.timestamp) - float(timestamp)) > self.config.depth_timestamp_tolerance:
                raise ValueError("DepthState.timestamp is stale relative to the current frame timestamp")
        current_mode = None if depth_state is None else DepthScaleMode(depth_state.scale_mode)
        had_previous = self.previous_state is not None
        previous_mode = None if self.previous_state is None else self.previous_state.scale_mode
        reset_reason = self._needs_reset(camera, frame_id, timestamp, current_mode, processing_id)
        if reset_reason is not None:
            self.reset(reset_reason)
        elif self.previous_state is None:
            reset_reason = self._pending_reset_reason or "first_frame"
        self.camera = camera
        if depth_state is None and self.previous_state is None:
            if not had_previous:
                raise ValueError("first temporal update requires a DepthState")
            state = self._empty_state(frame_id, timestamp, camera, previous_mode or DepthScaleMode.RELATIVE, processing_id)
            self.previous_state = state
            self.previous_rgb = None if rgb_frame is None else np.asarray(rgb_frame).copy()
            self.previous_timestamp = timestamp
            self.previous_frame_id = frame_id
            self.previous_processing_frame_id = processing_id
            self.last_diagnostics = GeometryDiagnostics(
                total_ms=(time.perf_counter() - total_start) * 1000.0 if timing_enabled else None,
                valid_geometry_percent=0.0,
                valid_normal_percent=0.0,
                mean_temporal_age=0.0,
                mean_confidence=0.0,
                reset_reason=reset_reason,
                depth_source_frame_delta=None,
            )
            self._pending_reset_reason = None
            return state

        motion_start = time.perf_counter() if timing_enabled else 0.0
        motion = self._get_motion(rgb_frame, processing_id, timestamp, motion_state)
        if timing_enabled:
            stage_ms["flow_ms"] = (time.perf_counter() - motion_start) * 1000.0
        self.motion_state = motion
        backprojection_start = time.perf_counter() if timing_enabled else 0.0
        if depth_state is not None:
            mode = DepthScaleMode(depth_state.scale_mode)
            raw_positions, current_valid = backproject_depth(depth_state.depth, camera, mode, depth_state.valid_mask)
            spatial = estimate_normals(
                raw_positions, current_valid, depth_state.depth, self.config.normal_mode, self.normal_config, depth_state.confidence
            )
            current_depth = np.asarray(depth_state.depth, dtype=np.float32)
        else:
            mode = self.previous_state.scale_mode
            raw_positions = None
            current_valid = np.zeros((camera.height, camera.width), dtype=bool)
            spatial = None
            current_depth = np.full(current_valid.shape, np.nan, dtype=np.float32)
        if timing_enabled:
            stage_ms["backprojection_ms"] = (time.perf_counter() - backprojection_start) * 1000.0

        has_reprojected_history = self.previous_state is not None and motion is not None

        if not has_reprojected_history:
            history_depth = np.full(current_valid.shape, np.nan, dtype=np.float32)
            history_valid = np.zeros(current_valid.shape, dtype=bool)
            history_confidence = np.zeros(current_valid.shape, dtype=np.float32)
            age = np.zeros(current_valid.shape, dtype=np.uint16)
            flow_valid = np.zeros(current_valid.shape, dtype=bool)
        else:
            warp_start = time.perf_counter() if timing_enabled else 0.0
            history_depth, history_valid, history_confidence, age, flow_valid = self._history_inputs(motion, rgb_frame)
            if timing_enabled:
                stage_ms["warp_ms"] = (time.perf_counter() - warp_start) * 1000.0

        alignment = DepthAlignmentResult(1.0, 0.0, 0, float("inf"), False, model_used="none", normalized_fit_residual=float("inf"))
        aligned_history = history_depth
        aligned_valid = history_valid.copy()
        rel_diff = np.zeros(current_valid.shape, dtype=np.float32)
        disagreement = np.zeros(current_valid.shape, dtype=np.float32)

        if depth_state is not None and history_valid.any():
            alignment_start = time.perf_counter() if timing_enabled else 0.0
            if mode is DepthScaleMode.RELATIVE:
                alignment = align_inverse_depth(
                    current_depth,
                    history_depth,
                    current_valid & history_valid,
                    weights=history_confidence,
                    min_samples=self.config.alignment_min_samples,
                    max_samples=self.config.alignment_max_samples,
                    residual_threshold=self.config.alignment_residual_threshold,
                    epsilon=self.config.alignment_epsilon,
                )
                aligned_history, aligned_valid = align_history_depth(
                    history_depth, alignment, mode, self.config.alignment_epsilon
                )
            else:
                aligned_valid = np.isfinite(history_depth) & (history_depth > self.config.alignment_epsilon)
            aligned_valid &= history_valid
            if timing_enabled:
                stage_ms["alignment_ms"] = (time.perf_counter() - alignment_start) * 1000.0

            # Dimensionless relative depth difference
            denominator = np.maximum(np.minimum(np.abs(current_depth), np.abs(aligned_history)), self.config.alignment_epsilon)
            rel_diff = np.where(aligned_valid & current_valid, (current_depth - aligned_history) / denominator, 0.0)
            disagreement = np.abs(rel_diff)
            agreement = np.exp(-((disagreement / self.config.depth_disagreement_threshold) ** 2))
            history_confidence *= agreement.astype(np.float32)
            history_valid &= aligned_valid & (disagreement <= self.config.depth_disagreement_threshold * 3.0)
        elif depth_state is None:
            # History-only propagation: stale age enforcement applies here
            history_valid &= (age <= self.config.max_history_age)

        # Distinct mask semantics
        occlusion = np.zeros(current_valid.shape, dtype=bool)
        disocclusion = np.zeros(current_valid.shape, dtype=bool)
        history_rejection = np.zeros(current_valid.shape, dtype=bool)

        if depth_state is not None:
            fusion_start = time.perf_counter() if timing_enabled else 0.0
            both_tested = current_valid & aligned_valid
            # Geometric occlusion: new foreground surface moved in front
            occlusion = both_tested & (rel_diff < -self.config.depth_disagreement_threshold)
            # Geometric disocclusion: foreground surface moved away revealing background
            disocclusion = both_tested & (rel_diff > self.config.depth_disagreement_threshold)
            history_rejection = has_reprojected_history & ~history_valid

            current_confidence = spatial.confidence if spatial is not None else np.zeros_like(current_valid, dtype=np.float32)
            current_weight = np.where(current_valid, np.maximum(current_confidence, 0.05), 0.0)
            history_weight = np.where(history_valid, np.minimum(history_confidence, self.config.history_weight_max), 0.0)
            both = current_valid & history_valid
            fused_depth = np.full(current_depth.shape, np.nan, dtype=np.float32)
            denominator = current_weight + history_weight
            fused_depth[current_valid & ~history_valid] = current_depth[current_valid & ~history_valid]
            fused_depth[~current_valid & history_valid] = aligned_history[~current_valid & history_valid]
            fused_depth[both] = (
                (current_weight[both] * current_depth[both] + history_weight[both] * aligned_history[both])
                / np.maximum(denominator[both], 1e-6)
            ).astype(np.float32)
            final_valid = np.isfinite(fused_depth) & (fused_depth > self.config.alignment_epsilon)
            temporal_confidence = np.where(
                both,
                (current_weight * current_confidence + history_weight * history_confidence) / np.maximum(denominator, 1e-6),
                np.where(current_valid, current_confidence, history_confidence),
            )

            # Temporal age: 0 when fresh depth observation is accepted; increment only for history-only fill
            age = np.where(current_valid, 0, np.where(history_valid, age, 0)).astype(np.uint16)
            spatial_confidence = current_confidence
            if timing_enabled:
                stage_ms["fusion_ms"] = (time.perf_counter() - fusion_start) * 1000.0
        else:
            history_rejection = has_reprojected_history & ~history_valid
            fused_depth = np.where(history_valid, aligned_history, np.nan).astype(np.float32)
            final_valid = history_valid & np.isfinite(fused_depth) & (fused_depth > self.config.alignment_epsilon)
            temporal_confidence = np.where(final_valid, history_confidence * self.config.history_decay, 0.0)
            temporal_confidence = np.where(age <= self.config.max_history_age, temporal_confidence, 0.0)
            final_valid &= temporal_confidence >= self.config.history_min_confidence
            age = np.where(final_valid, age, 0).astype(np.uint16)
            spatial_confidence = np.zeros_like(temporal_confidence)

        post_start = time.perf_counter() if timing_enabled else 0.0
        positions, _ = backproject_depth(fused_depth, camera, mode, final_valid)
        if timing_enabled:
            stage_ms["backprojection_ms"] = stage_ms.get("backprojection_ms", 0.0) + (time.perf_counter() - post_start) * 1000.0
        normal_start = time.perf_counter() if timing_enabled else 0.0
        normal_result = estimate_normals(positions, final_valid, fused_depth, self.config.normal_mode, self.normal_config)
        if timing_enabled:
            stage_ms["normals_ms"] = (time.perf_counter() - normal_start) * 1000.0
        history_confidence = np.where(history_valid, history_confidence, 0.0).astype(np.float32)
        final_confidence = np.clip(np.nan_to_num(temporal_confidence), 0.0, 1.0).astype(np.float32)
        final_confidence = np.where(final_valid, final_confidence, 0.0).astype(np.float32)
        alignment_map = np.full(final_valid.shape, alignment.fit_residual, dtype=np.float32)

        return_state = GeometryState(
            timestamp=timestamp,
            source_frame_id=frame_id,
            depth=fused_depth,
            positions_3d=positions,
            valid_mask=final_valid,
            camera=camera,
            scale_mode=mode,
            normals=normal_result.normals,
            confidence=final_confidence,
            temporal_age=age,
            history_valid=history_valid,
            occlusion_mask=occlusion,
            normal_valid_mask=normal_result.normal_valid_mask,
            normal_confidence=normal_result.confidence,
            selected_radius=normal_result.selected_radius,
            spatial_confidence=spatial_confidence.astype(np.float32),
            history_confidence=history_confidence,
            temporal_confidence=final_confidence,
            depth_alignment_residual=alignment_map,
            history_rejection_mask=history_rejection,
            disocclusion_mask=disocclusion,
            processing_frame_id=processing_id,
        )
        self.previous_state = return_state
        self.previous_rgb = None if rgb_frame is None else np.asarray(rgb_frame).copy()
        self.previous_timestamp = timestamp
        self.previous_frame_id = frame_id
        self.previous_processing_frame_id = processing_id
        self.last_alignment = alignment
        if self.config.diagnostics_level != "none":
            total = float(final_valid.size)
            history_tested = float(has_reprojected_history and motion is not None)
            accepted = float(history_valid.mean() * 100.0) if history_tested else None
            rejected = float(history_rejection.mean() * 100.0) if history_tested else None
            self.last_diagnostics = GeometryDiagnostics(
                total_ms=(time.perf_counter() - total_start) * 1000.0 if timing_enabled else None,
                flow_ms=stage_ms.get("flow_ms"), warp_ms=stage_ms.get("warp_ms"), alignment_ms=stage_ms.get("alignment_ms"),
                fusion_ms=stage_ms.get("fusion_ms"), backprojection_ms=stage_ms.get("backprojection_ms"), normals_ms=stage_ms.get("normals_ms"),
                valid_geometry_percent=float(final_valid.mean() * 100.0),
                valid_normal_percent=float(normal_result.normal_valid_mask.mean() * 100.0),
                history_acceptance_percent=accepted, history_rejection_percent=rejected,
                occlusion_percent=float(occlusion.mean() * 100.0), disocclusion_percent=float(disocclusion.mean() * 100.0),
                mean_temporal_age=float(age[final_valid].mean()) if final_valid.any() else 0.0,
                mean_confidence=float(final_confidence[final_valid].mean()) if final_valid.any() else 0.0,
                alignment_success=alignment.fit_success, alignment_model=alignment.model_used,
                alignment_residual=alignment.fit_residual, reset_reason=reset_reason or self._pending_reset_reason or "none",
                depth_source_frame_delta=(int(frame_id - depth_state.source_frame_id) if depth_state is not None and isinstance(frame_id, int) and isinstance(depth_state.source_frame_id, int) else None),
            )
        self._pending_reset_reason = None
        return return_state
