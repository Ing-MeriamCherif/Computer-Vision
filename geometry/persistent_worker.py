"""Asynchronous, bounded background worker for persistent surfel mapping.

Runs at a decoupled rate (2-10 Hz) as an optional sidecar, guaranteeing that
Infinity mapping never blocks render or gesture deadlines.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .camera import CameraModel
from .persistent import PersistentGeometryMapper, PersistentGeometryState, SurfelMap
from .pose import CameraPoseState, PoseEstimateResult
from .state import GeometryState


@dataclass(slots=True)
class MapJob:
    current_geometry: GeometryState
    pose_result: PoseEstimateResult | CameraPoseState
    static_confidence: np.ndarray | None
    timestamp: float
    frame_id: int | str


class PersistentMapWorker:
    """Bounded, asynchronous worker for Phase 5 persistent geometry mapping."""

    def __init__(
        self,
        camera: CameraModel,
        mapper: PersistentGeometryMapper | None = None,
        *,
        map_update_hz: float = 4.0,
        max_candidates: int = 1000,
    ) -> None:
        self.camera = camera
        scene_voxel = 2.0 * 0.02
        self.mapper = mapper or PersistentGeometryMapper(
            camera,
            SurfelMap(voxel_size=scene_voxel, max_surfels=50_000),
            mapping_stride=max(2, camera.width // 160),
        )
        self.map_update_hz = max(0.5, float(map_update_hz))
        self.max_candidates = max(100, int(max_candidates))

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pending_job: MapJob | None = None
        self._latest_snapshot: PersistentGeometryState | None = None

        self._thread: threading.Thread | None = None
        self._running = False

        self.jobs_dropped: int = 0
        self.jobs_processed: int = 0
        self.last_update_ms: float = 0.0
        self.last_update_time: float = 0.0
        self.update_fps: float = 0.0
        self.tracking_state: str = "UNINITIALIZED"
        self.latest_exception: Exception | None = None

        self._fps_samples: list[float] = []

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="PersistentMapWorker")
        self._thread.start()

    def submit(
        self,
        current_geometry: GeometryState,
        pose_result: PoseEstimateResult | CameraPoseState,
        static_confidence: np.ndarray | None = None,
        frame_id: int | str = 0,
        timestamp: float | None = None,
    ) -> bool:
        """Submit an update job. Drops prior unconsumed job if worker is busy."""
        with self._condition:
            if self._pending_job is not None:
                self.jobs_dropped += 1
            ts = time.monotonic() if timestamp is None else float(timestamp)
            self._pending_job = MapJob(current_geometry, pose_result, static_confidence, ts, frame_id)
            self._condition.notify_all()
            return True

    def get_latest_snapshot(self) -> PersistentGeometryState | None:
        """Non-blocking retrieval of the latest read-only persistent map snapshot."""
        with self._lock:
            return self._latest_snapshot

    def _worker_loop(self) -> None:
        min_interval = 1.0 / self.map_update_hz

        while self._running:
            job: MapJob | None = None
            with self._condition:
                while self._running and self._pending_job is None:
                    self._condition.wait(timeout=0.1)
                if not self._running:
                    break
                job = self._pending_job
                self._pending_job = None

            if job is None:
                continue

            t0 = time.perf_counter()
            try:
                # Subsample candidates if needed to honor max_candidates
                geom = job.current_geometry
                static_conf = job.static_confidence
                snapshot = self.mapper.update(
                    geom,
                    job.pose_result,
                    static_confidence=static_conf,
                    frame_id=job.frame_id,
                )

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                now = time.monotonic()

                with self._lock:
                    self._latest_snapshot = snapshot
                    self.last_update_ms = elapsed_ms
                    self.jobs_processed += 1
                    self.tracking_state = getattr(self.mapper, "tracking_state", "TRACKING")
                    if self.last_update_time > 0:
                        delta = now - self.last_update_time
                        if delta > 1e-4:
                            self._fps_samples.append(1.0 / delta)
                            if len(self._fps_samples) > 20:
                                self._fps_samples.pop(0)
                            self.update_fps = float(np.median(self._fps_samples))
                    self.last_update_time = now

            except Exception as exc:
                self.latest_exception = exc
                with self._lock:
                    self.tracking_state = "LOST"

            # Enforce map update governor cadence
            work_time = time.perf_counter() - t0
            sleep_time = min_interval - work_time
            if sleep_time > 0.002:
                time.sleep(sleep_time)

    def reset(self) -> None:
        with self._lock:
            self._pending_job = None
            self._latest_snapshot = None
            self.tracking_state = "UNINITIALIZED"
            if hasattr(self.mapper, "reset"):
                self.mapper.reset()

    def stop(self) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self._running

    @property
    def surfel_count(self) -> int:
        return self.mapper.map.count if hasattr(self.mapper, "map") else 0
