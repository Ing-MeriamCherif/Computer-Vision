"""Asynchronous physical-camera runtime ending at the P123 handoff contract.

This module deliberately stops before Person 4: it contains no lighting,
shadows, volumetrics, shaders, or renderer integration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Any

import numpy as np

from .async_pipeline import LatestDepthBuffer, LatestFrameBuffer
from .backproject import DepthScaleMode
from .camera import CameraModel
from .camera_worker import CameraCaptureWorker
from .depth_provider import DepthAnythingProvider
from .depth_sampling import sample_depth
from .hand_control import GestureState, HandControlEngine
from .motion import OpenCVFlowProvider
from .p123_contract import HandXYZ, P4InputState
from .state import DepthState, GeometryState
from .temporal import TemporalConfig, TemporalGeometryEngine


@dataclass(frozen=True, slots=True)
class P123Metrics:
    capture_hz: float | None
    depth_hz: float | None
    geometry_hz: float | None
    temporal_hz: float | None
    hand_hz: float | None
    xyz_hz: float | None
    depth_age_p50_ms: float | None
    depth_age_p95_ms: float | None
    geometry_age_p95_ms: float | None
    hand_age_p95_ms: float | None
    xyz_age_p95_ms: float | None
    captured: int
    overwritten_before_consumption: int
    depth_errors: int
    normal_hz: float | None = None
    normal_age_p95_ms: float | None = None


@dataclass(frozen=True, slots=True)
class P123Snapshot:
    rgb_frame: np.ndarray | None
    rgb_capture_id: int | None
    rgb_timestamp: float | None
    depth_state: DepthState | None
    geometry_state: GeometryState | None
    hand_state: GestureState | None
    xyz: tuple[HandXYZ, ...]
    contract: P4InputState | None
    metrics: P123Metrics
    fast_geometry_state: GeometryState | None = None


def _rate(timestamps) -> float | None:
    if len(timestamps) < 2:
        return None
    elapsed = timestamps[-1] - timestamps[0]
    return float((len(timestamps) - 1) / elapsed) if elapsed > 0 else None


def _percentile(values, p: float) -> float | None:
    return float(np.percentile(values, p)) if values else None


def _fast_temporal_confidence(
    previous_depth: np.ndarray | None,
    current_depth: np.ndarray,
    current_valid: np.ndarray | None,
    current_confidence: np.ndarray | None,
) -> np.ndarray:
    """Compute a cheap native-resolution depth-consistency confidence map."""
    current = np.asarray(current_depth, dtype=np.float32)
    valid = np.isfinite(current) & (current > 1e-6)
    if current_valid is not None:
        valid &= np.asarray(current_valid, dtype=bool)
    if previous_depth is None:
        confidence = np.ones(current.shape, dtype=np.float32)
    else:
        previous = np.asarray(previous_depth, dtype=np.float32)
        if previous.shape != current.shape:
            return np.zeros(current.shape, dtype=np.float32)
        overlap = valid & np.isfinite(previous) & (previous > 1e-6)
        if not overlap.any():
            return np.zeros(current.shape, dtype=np.float32)
        prev_med = float(np.median(previous[overlap]))
        curr_med = float(np.median(current[overlap]))
        aligned = current * (prev_med / max(curr_med, 1e-6))
        relative_error = np.abs(aligned - previous) / np.maximum(np.abs(previous), 1e-6)
        confidence = np.exp(-np.clip(relative_error / 0.12, 0.0, 8.0)).astype(np.float32)
        valid &= overlap
    if current_confidence is not None:
        confidence *= np.clip(np.nan_to_num(np.asarray(current_confidence, dtype=np.float32)), 0.0, 1.0)
    return np.where(valid, np.clip(confidence, 0.0, 1.0), 0.0).astype(np.float32)


def _smooth_depth(
    previous_depth: np.ndarray | None,
    current_depth: np.ndarray,
    current_valid: np.ndarray | None,
    alpha: float = 0.30,
) -> np.ndarray:
    """Adaptive robust temporal depth: preserve edges and reject invalid history."""
    current = np.asarray(current_depth, dtype=np.float32)
    valid = np.isfinite(current) & (current > 1e-6)
    if current_valid is not None:
        valid &= np.asarray(current_valid, dtype=bool)
    if previous_depth is None or np.asarray(previous_depth).shape != current.shape:
        return np.where(valid, current, np.nan).astype(np.float32)
    previous = np.asarray(previous_depth, dtype=np.float32)
    overlap = valid & np.isfinite(previous) & (previous > 1e-6)
    if not overlap.any():
        return np.where(valid, current, np.nan).astype(np.float32)
    ratio = float(np.median(previous[overlap])) / max(float(np.median(current[overlap])), 1e-6)
    aligned = current * ratio
    relative_delta = np.abs(aligned - previous) / np.maximum(np.abs(previous), 1e-6)
    # Responsive on motion/edges, smooth only where the two estimates agree.
    local_alpha = np.clip(alpha + 0.55 * np.clip(relative_delta / 0.20, 0.0, 1.0), alpha, 0.9)
    agree = relative_delta <= 0.28
    smoothed = np.where(overlap & agree, (1.0 - local_alpha) * previous + local_alpha * aligned, aligned)
    return np.where(valid, smoothed, np.nan).astype(np.float32)


def _talel_depth_from_palm_size(
    fx: float, palm_width_px: float | None, *, real_palm_m: float = 0.085,
    z_min: float = 0.20, z_max: float = 3.0, fallback: float = 0.50,
) -> tuple[float, bool]:
    """Talel hand branch's stable metric Z proxy from apparent palm width."""
    if palm_width_px is None or palm_width_px < 8.0:
        return float(fallback), False
    z = float(fx * real_palm_m / palm_width_px)
    return float(np.clip(z, z_min, z_max)), True


def _talel_hand_xyz(camera: CameraModel, palm_uv: tuple[float, float], palm_width_px: float | None) -> tuple[np.ndarray, bool]:
    """Return Talel's camera-space hand position and whether Z was estimated."""
    z, estimated = _talel_depth_from_palm_size(camera.fx, palm_width_px)
    return np.asarray(camera.unproject(palm_uv[0], palm_uv[1], z), dtype=np.float32), estimated


class P123LiveRuntime:
    """Run camera, depth, geometry/temporal, and hands on independent workers."""

    def __init__(
        self,
        *,
        camera_device: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        depth_provider: Any | None = None,
        depth_model: str = "models/depth-anything-v2-small",
        depth_size: int | tuple[int, int] | None = None,
        depth_backend: str = "local",
        use_fp16: bool = False,
        hand_backend: str = "auto",
        calibration: CameraModel | None = None,
        max_state_age_ms: float = 200.0,
        full_temporal: bool = False,
    ) -> None:
        self.camera_worker = CameraCaptureWorker(camera_device, width, height, fps)
        self._explicit_calibration = calibration is not None
        self.camera = calibration or CameraModel(width, height, width * 0.82, width * 0.82, width / 2.0, height / 2.0)
        if depth_provider is not None:
            self.depth_provider = depth_provider
        elif depth_backend in {"colleague", "mariem"}:
            from .colleague_depth import ColleagueDepthProvider, MariemDepthProvider
            if depth_size is None:
                colleague_size = max(height, width)
            elif isinstance(depth_size, tuple):
                colleague_size = max(int(v) for v in depth_size)
            else:
                colleague_size = int(depth_size)
            provider_cls = MariemDepthProvider if depth_backend == "mariem" else ColleagueDepthProvider
            self.depth_provider = provider_cls(device="auto", input_size=colleague_size, fp16=use_fp16)
        elif depth_backend == "local":
            # Native camera dimensions preserve fine spatial detail; the
            # latest-only worker keeps capture/UI cadence independent of the
            # heavier inference cost.
            native_size = (height, width) if depth_size is None else depth_size
            self.depth_provider = DepthAnythingProvider(depth_model, device="auto", use_fp16=use_fp16, input_size=native_size)
        else:
            raise ValueError(f"unknown depth backend: {depth_backend}")
        self.hand_engine = HandControlEngine(model_path="models/hand_landmarker.task", max_hands=2, backend=hand_backend, detect_every_n=2, max_coast_frames=8)
        self.temporal = TemporalGeometryEngine(
            self.camera,
            TemporalConfig(diagnostics_level="timing"),
            OpenCVFlowProvider(method="farneback", flow_scale=0.5),
        )
        self.max_state_age_ms = float(max_state_age_ms)
        self.full_temporal = bool(full_temporal)
        self._depth_frames = LatestFrameBuffer()
        self._hand_frames = LatestFrameBuffer()
        self._depth_states = LatestDepthBuffer()
        self._rgb_lock = threading.Lock()
        self._rgb_history: dict[int | str, tuple[np.ndarray, float]] = {}
        self._rgb_order: deque[int | str] = deque()
        self._state_lock = threading.Lock()
        self._rgb: tuple[np.ndarray, int, float] | None = None
        self._depth: DepthState | None = None
        self._geometry: GeometryState | None = None
        self._fast_geometry: GeometryState | None = None
        self._normal_backend = None
        try:
            from .cuda_backend import TorchGeometryBackend
            backend = TorchGeometryBackend("auto")
            if backend.is_cuda:
                self._normal_backend = backend
        except Exception:
            self._normal_backend = None
        self._hands: GestureState | None = None
        self._running = False
        self._threads: list[threading.Thread] = []
        self._last_dispatched_id: int | None = None
        self._last_geometry_depth_id: int | str | None = None
        self._processing_id = 0
        self._depth_errors = 0
        metric = lambda: deque(maxlen=256)
        self._capture_times = metric(); self._depth_times = metric(); self._geometry_times = metric()
        self._normal_times = metric(); self._fast_temporal_times = metric(); self._hand_times = metric(); self._xyz_times = metric()
        self._depth_ages = metric(); self._geometry_ages = metric(); self._hand_ages = metric(); self._xyz_ages = metric(); self._normal_ages = metric()
        self._xyz: tuple[HandXYZ, ...] = ()
        self._xyz_smooth: dict[int, np.ndarray] = {}

    def start(self) -> None:
        if self._running:
            return
        # Keep the diagnostic UI responsive while OpenCV/MediaPipe workers run
        # concurrently on the same workstation.
        try:
            import cv2
            cv2.setNumThreads(1)
        except Exception:
            pass
        self.depth_provider.load()
        warmup = getattr(self.depth_provider, "warmup", None)
        if callable(warmup):
            warmup(iterations=1)
        if self.hand_engine.backend_name in {"unavailable", "mock"}:
            raise RuntimeError(f"hand backend unavailable: {self.hand_engine.backend_name}")
        self.camera_worker.start()
        if not self._explicit_calibration:
            self.camera = CameraModel(
                self.camera_worker.actual_width, self.camera_worker.actual_height,
                self.camera_worker.actual_width * 0.82, self.camera_worker.actual_width * 0.82,
                self.camera_worker.actual_width / 2.0, self.camera_worker.actual_height / 2.0,
            )
        elif (self.camera.width, self.camera.height) != (self.camera_worker.actual_width, self.camera_worker.actual_height):
            self.camera_worker.stop()
            raise RuntimeError(
                f"explicit calibration {self.camera.width}x{self.camera.height} mismatches negotiated camera "
                f"{self.camera_worker.actual_width}x{self.camera_worker.actual_height}"
            )
        self._running = True
        targets = [self._dispatch_loop, self._depth_loop, self._normal_loop, self._hand_loop, self._xyz_loop]
        if self.full_temporal:
            targets.insert(2, self._geometry_loop)
        self._threads = [threading.Thread(target=target, name=target.__name__, daemon=True) for target in targets]
        for thread in self._threads:
            thread.start()

    def _dispatch_loop(self) -> None:
        while self._running:
            packet = self.camera_worker.slot.get(timeout=0.2)
            if packet is None:
                continue
            frame, capture_id, timestamp = packet
            self._last_dispatched_id = capture_id
            self._capture_times.append(time.monotonic())
            with self._rgb_lock:
                self._rgb = (frame, capture_id, timestamp)
                self._rgb_history[capture_id] = (frame, timestamp)
                self._rgb_order.append(capture_id)
                while len(self._rgb_history) > 24:
                    self._rgb_history.pop(self._rgb_order.popleft(), None)
            self._depth_frames.put(frame, capture_id, timestamp)
            self._hand_frames.put(frame, capture_id, timestamp)

    def _depth_loop(self) -> None:
        version = 0
        while self._running:
            packet, version = self._depth_frames.wait_for_new(version, timeout=0.2)
            if packet is None:
                continue
            try:
                state = self.depth_provider.compute(packet.frame, packet.frame_id, packet.timestamp)
                self._depth_states.put(state)
                completed = time.monotonic()
                state.completed_timestamp = completed
                with self._state_lock:
                    self._depth = state
                self._depth_times.append(completed)
                self._depth_ages.append(max(0.0, (completed - packet.timestamp) * 1000.0))
            except Exception:
                self._depth_errors += 1

    def _geometry_loop(self) -> None:
        version = 0
        while self._running:
            state, version = self._depth_states.wait_for_new(version, timeout=0.2)
            if state is None:
                continue
            with self._rgb_lock:
                source = self._rgb_history.get(state.source_frame_id)
            if source is None:
                continue
            rgb, _ = source
            # The physical capture ID is preserved as source metadata. The
            # contiguous processing ID is used only by temporal continuity.
            processing_id = self._processing_id
            self._processing_id += 1
            try:
                geometry = self.temporal.update(
                    rgb, self.camera, state.source_frame_id, state.timestamp, state,
                    processing_frame_id=processing_id,
                )
            except Exception:
                self._last_geometry_depth_id = state.source_frame_id
                continue
            self._last_geometry_depth_id = state.source_frame_id
            completed = time.monotonic()
            geometry.completed_timestamp = completed
            with self._state_lock:
                self._geometry = geometry
            self._geometry_times.append(completed)
            self._geometry_ages.append(max(0.0, (completed - state.timestamp) * 1000.0))

    def _normal_loop(self) -> None:
        """Publish native-resolution CUDA normals without blocking temporal CPU work."""
        if self._normal_backend is None:
            return
        last_depth_id: int | str | None = None
        version = 0
        previous_depth: np.ndarray | None = None
        smoothed_depth: np.ndarray | None = None
        while self._running:
            state, version = self._depth_states.wait_for_new(version, timeout=0.2)
            if state is None:
                continue
            try:
                smoothed_depth = _smooth_depth(smoothed_depth, state.depth, state.valid_mask)
                fast = self._normal_backend.process_depth(
                    smoothed_depth,
                    self.camera,
                    frame_id=state.source_frame_id,
                    timestamp=state.timestamp,
                    scale_mode=state.scale_mode,
                    valid_mask=state.valid_mask,
                    input_confidence=state.confidence,
                )
                fast.processing_frame_id = state.source_frame_id
                fast.completed_timestamp = time.monotonic()
                fast.temporal_confidence = _fast_temporal_confidence(
                    previous_depth, smoothed_depth, state.valid_mask, state.confidence
                )
                with self._state_lock:
                    self._fast_geometry = fast
                self._normal_times.append(time.monotonic())
                self._normal_ages.append(max(0.0, (time.monotonic() - state.timestamp) * 1000.0))
                self._fast_temporal_times.append(time.monotonic())
                previous_depth = smoothed_depth.copy()
                last_depth_id = state.source_frame_id
            except Exception:
                # Keep the temporal worker/UI alive if CUDA geometry rejects a frame.
                last_depth_id = state.source_frame_id

    def _hand_loop(self) -> None:
        version = 0
        while self._running:
            packet, version = self._hand_frames.wait_for_new(version, timeout=0.2)
            if packet is None:
                continue
            try:
                state = self.hand_engine.update(packet.frame, packet.timestamp, packet.frame_id)
                with self._state_lock:
                    self._hands = state
                completed = time.monotonic()
                self._hand_times.append(completed)
                self._hand_ages.append(max(0.0, (completed - packet.timestamp) * 1000.0))
            except Exception:
                continue

    def _xyz_loop(self) -> None:
        last_key: tuple[int | str | None, int | str | None] = (None, None)
        while self._running:
            with self._state_lock:
                # Prefer the fast native CUDA geometry state. The full CPU
                # temporal state is retained for the contract but is too old
                # for interactive XYZ on this hardware.
                geometry, hands = self._fast_geometry or self._geometry, self._hands
            if geometry is None or hands is None:
                time.sleep(0.01)
                continue
            key = (geometry.source_frame_id, hands.source_frame_id)
            if key == last_key:
                time.sleep(0.01)
                continue
            last_key = key
            # Freshness is measured from geometry completion, not capture
            # time. Mariem/Talel inference can legitimately finish a frame
            # 100+ ms after capture; using the source timestamp made every
            # valid hand appear perpetually "depth pending".
            completed_at = geometry.completed_timestamp or geometry.timestamp
            age_ms = max(0.0, (time.monotonic() - completed_at) * 1000.0)
            depth_hz = self._depth_times and _rate(self._depth_times) or 8.0
            freshness_limit = min(self.max_state_age_ms, max(180.0, 1.75 * (1000.0 / max(depth_hz, 1.0))))
            if age_ms > freshness_limit:
                self._xyz = ()
                for stale_id in tuple(self._xyz_smooth):
                    self._xyz_smooth.pop(stale_id, None)
                continue
            values: list[HandXYZ] = []
            with self._state_lock:
                raw_depth_state = self._depth
            for hand in hands.hands:
                sampled_z, reliability = sample_depth(geometry.depth, geometry.valid_mask, *hand.palm_uv)
                if sampled_z <= 0.0 and raw_depth_state is not None:
                    sampled_z, reliability = sample_depth(raw_depth_state.depth, raw_depth_state.valid_mask, *hand.palm_uv)
                # Relative monocular depth has no metric unit. Use it only
                # when it lands in Talel's physically plausible working
                # volume; otherwise use his palm-size proxy so XYZ cannot
                # collapse to centimetre-scale coordinates.
                size_z, size_ok = _talel_depth_from_palm_size(self.camera.fx, hand.palm_width_px)
                mode_value = getattr(getattr(geometry, "scale_mode", "relative"), "value", getattr(geometry, "scale_mode", "relative"))
                metric_depth = str(mode_value) == "metric"
                z = sampled_z if metric_depth and 0.20 <= sampled_z <= 3.0 else size_z
                if not size_ok:
                    reliability *= 0.5
                if metric_depth and z > 0:
                    raw_xyz = np.asarray(geometry.camera.unproject(hand.palm_uv[0], hand.palm_uv[1], z), dtype=np.float32)
                elif z > 0:
                    raw_xyz, _ = _talel_hand_xyz(geometry.camera, hand.palm_uv, hand.palm_width_px)
                else:
                    raw_xyz = None
                if raw_xyz is None:
                    prior_xyz = self._xyz_smooth.get(hand.hand_id)
                    xyz = None if prior_xyz is None else tuple(float(v) for v in prior_xyz)
                    reliability = 0.0 if prior_xyz is None else reliability * 0.5
                else:
                    prior_xyz = self._xyz_smooth.get(hand.hand_id)
                    motion = float(np.linalg.norm(raw_xyz - prior_xyz)) if prior_xyz is not None else 1.0
                    blend = float(np.clip(0.25 + motion * 0.45, 0.25, 0.75))
                    smooth_xyz = raw_xyz if prior_xyz is None else blend * raw_xyz + (1.0 - blend) * prior_xyz
                    self._xyz_smooth[hand.hand_id] = smooth_xyz
                    xyz = tuple(float(v) for v in smooth_xyz)
                values.append(HandXYZ(hand.hand_id, hand.palm_uv, xyz, float(hand.confidence * reliability), hand.timestamp, hands.source_frame_id, age_ms, hand.handedness))
            self._xyz = tuple(values)
            now = time.monotonic()
            self._xyz_times.append(now)
            self._xyz_ages.append(max(0.0, (now - hands.timestamp) * 1000.0))

    def snapshot(self) -> P123Snapshot:
        with self._rgb_lock:
            rgb = self._rgb
        with self._state_lock:
            depth, geometry, hands, fast_geometry = self._depth, self._geometry, self._hands, self._fast_geometry
        canonical_geometry = fast_geometry or geometry
        xyz = tuple(self._xyz)
        contract = None
        if rgb is not None and canonical_geometry is not None:
            contract = P4InputState(
                rgb_frame=rgb[0], rgb_capture_id=rgb[1], rgb_timestamp=rgb[2],
                camera_model=self.camera, geometry_state=canonical_geometry, hand_states=xyz,
                data_age_metrics={
                    "depth_age_ms": max(0.0, (time.monotonic() - canonical_geometry.timestamp) * 1000.0),
                    "hand_age_ms": None if hands is None else max(0.0, (time.monotonic() - hands.timestamp) * 1000.0),
                },
                geometry_target_capture_id=canonical_geometry.source_frame_id,
                source_depth_capture_id=canonical_geometry.source_frame_id,
                confidence_metadata={"depth_reliability": None if depth is None else float(np.nanmean(depth.confidence)) if depth.confidence is not None else None},
            )
        now = time.monotonic()
        metrics = P123Metrics(
            self.camera_worker.actual_fps if self.camera_worker.captured_frames > 1 else _rate(self._capture_times), _rate(self._depth_times), _rate(self._geometry_times), _rate(self._fast_temporal_times), _rate(self._hand_times), _rate(self._xyz_times),
            _percentile(self._depth_ages, 50), _percentile(self._depth_ages, 95), _percentile(self._geometry_ages, 95), _percentile(self._hand_ages, 95), _percentile(self._xyz_ages, 95),
            self.camera_worker.captured_frames, self.camera_worker.overwritten_before_consumption, self._depth_errors,
            _rate(self._normal_times), _percentile(self._normal_ages, 95),
        )
        return P123Snapshot(None if rgb is None else rgb[0], None if rgb is None else rgb[1], None if rgb is None else rgb[2], depth, canonical_geometry, hands, xyz, contract, metrics, fast_geometry)

    def stop(self) -> None:
        self._running = False
        for thread in self._threads:
            thread.join(timeout=1.5)
        self._threads.clear()
        self.camera_worker.stop()
        self.hand_engine.close()
