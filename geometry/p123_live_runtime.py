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


def _rate(timestamps: list[float]) -> float | None:
    if len(timestamps) < 2:
        return None
    elapsed = timestamps[-1] - timestamps[0]
    return float((len(timestamps) - 1) / elapsed) if elapsed > 0 else None


def _percentile(values: list[float], p: float) -> float | None:
    return float(np.percentile(values, p)) if values else None


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
        max_state_age_ms: float = 120.0,
    ) -> None:
        self.camera_worker = CameraCaptureWorker(camera_device, width, height, fps)
        self.camera = calibration or CameraModel(width, height, width * 0.82, width * 0.82, width / 2.0, height / 2.0)
        if depth_provider is not None:
            self.depth_provider = depth_provider
        elif depth_backend == "colleague":
            from .colleague_depth import ColleagueDepthProvider
            if depth_size is None:
                colleague_size = max(height, width)
            elif isinstance(depth_size, tuple):
                colleague_size = max(int(v) for v in depth_size)
            else:
                colleague_size = int(depth_size)
            self.depth_provider = ColleagueDepthProvider(device="auto", input_size=colleague_size, fp16=use_fp16)
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
        self._hands: GestureState | None = None
        self._running = False
        self._threads: list[threading.Thread] = []
        self._last_dispatched_id: int | None = None
        self._last_geometry_depth_id: int | str | None = None
        self._processing_id = 0
        self._depth_errors = 0
        self._capture_times: list[float] = []
        self._depth_times: list[float] = []
        self._geometry_times: list[float] = []
        self._hand_times: list[float] = []
        self._xyz_times: list[float] = []
        self._depth_ages: list[float] = []
        self._geometry_ages: list[float] = []
        self._hand_ages: list[float] = []
        self._xyz_ages: list[float] = []
        self._xyz: tuple[HandXYZ, ...] = ()

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
        self._running = True
        targets = [self._dispatch_loop, self._depth_loop, self._geometry_loop, self._hand_loop, self._xyz_loop]
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
                while len(self._rgb_history) > 128:
                    self._rgb_history.pop(self._rgb_order.popleft(), None)
            self._depth_frames.put(frame, capture_id, timestamp)
            self._hand_frames.put(frame, capture_id, timestamp)

    def _depth_loop(self) -> None:
        while self._running:
            packet = self._depth_frames.get()
            if packet is None:
                time.sleep(0.001)
                continue
            try:
                state = self.depth_provider.compute(packet.frame, packet.frame_id, packet.timestamp)
                self._depth_states.put(state)
                completed = time.monotonic()
                with self._state_lock:
                    self._depth = state
                self._depth_times.append(completed)
                self._depth_ages.append(max(0.0, (completed - packet.timestamp) * 1000.0))
            except Exception:
                self._depth_errors += 1

    def _geometry_loop(self) -> None:
        while self._running:
            state = self._depth_states.get()
            if state is None or state.source_frame_id == self._last_geometry_depth_id:
                time.sleep(0.002)
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
            with self._state_lock:
                self._geometry = geometry
            self._geometry_times.append(completed)
            self._geometry_ages.append(max(0.0, (completed - state.timestamp) * 1000.0))

    def _hand_loop(self) -> None:
        while self._running:
            packet = self._hand_frames.get()
            if packet is None:
                time.sleep(0.001)
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
                geometry, hands = self._geometry, self._hands
            if geometry is None or hands is None:
                time.sleep(0.005)
                continue
            key = (geometry.source_frame_id, hands.source_frame_id)
            if key == last_key:
                time.sleep(0.005)
                continue
            last_key = key
            age_ms = max(0.0, (time.monotonic() - geometry.timestamp) * 1000.0)
            if age_ms > self.max_state_age_ms:
                self._xyz = ()
                continue
            values: list[HandXYZ] = []
            for hand in hands.hands:
                z, reliability = sample_depth(geometry.depth, geometry.valid_mask, *hand.palm_uv)
                xyz = None if z <= 0 else tuple(float(v) for v in geometry.camera.unproject(hand.palm_uv[0], hand.palm_uv[1], z))
                values.append(HandXYZ(hand.hand_id, hand.palm_uv, xyz, float(hand.confidence * reliability), hand.timestamp, hands.source_frame_id, age_ms, hand.handedness))
            self._xyz = tuple(values)
            now = time.monotonic()
            self._xyz_times.append(now)
            self._xyz_ages.append(max(0.0, (now - hands.timestamp) * 1000.0))

    def snapshot(self) -> P123Snapshot:
        with self._rgb_lock:
            rgb = self._rgb
        with self._state_lock:
            depth, geometry, hands = self._depth, self._geometry, self._hands
        xyz = tuple(self._xyz)
        contract = None
        if rgb is not None and geometry is not None:
            contract = P4InputState(
                rgb_frame=rgb[0], rgb_capture_id=rgb[1], rgb_timestamp=rgb[2],
                camera_model=self.camera, geometry_state=geometry, hand_states=xyz,
                data_age_metrics={
                    "depth_age_ms": max(0.0, (time.monotonic() - geometry.timestamp) * 1000.0),
                    "hand_age_ms": None if hands is None else max(0.0, (time.monotonic() - hands.timestamp) * 1000.0),
                },
                geometry_target_capture_id=geometry.source_frame_id,
                source_depth_capture_id=geometry.source_frame_id,
                confidence_metadata={"depth_reliability": None if depth is None else float(np.nanmean(depth.confidence)) if depth.confidence is not None else None},
            )
        now = time.monotonic()
        metrics = P123Metrics(
            self.camera_worker.actual_fps if self.camera_worker.captured_frames > 1 else _rate(self._capture_times), _rate(self._depth_times), _rate(self._geometry_times), _rate(self._geometry_times), _rate(self._hand_times), _rate(self._xyz_times),
            _percentile(self._depth_ages, 50), _percentile(self._depth_ages, 95), _percentile(self._geometry_ages, 95), _percentile(self._hand_ages, 95), _percentile(self._xyz_ages, 95),
            self.camera_worker.captured_frames, self.camera_worker.overwritten_before_consumption, self._depth_errors,
        )
        return P123Snapshot(None if rgb is None else rgb[0], None if rgb is None else rgb[1], None if rgb is None else rgb[2], depth, geometry, hands, xyz, contract, metrics)

    def stop(self) -> None:
        self._running = False
        for thread in self._threads:
            thread.join(timeout=1.5)
        self._threads.clear()
        self.camera_worker.stop()
        self.hand_engine.close()
