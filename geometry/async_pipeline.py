"""Bounded asynchronous inference primitives imported from the depth branch.

The original branch kept only a raw frame and consequently reprocessed the
same image repeatedly.  These buffers carry frame IDs and the worker drops
duplicates/stale work so a live renderer can keep consuming the newest state.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable

import numpy as np

from .state import DepthState


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame: np.ndarray
    frame_id: int | str
    timestamp: float


class LatestFrameBuffer:
    def __init__(self) -> None:
        self._packet: FramePacket | None = None
        self._lock = threading.Lock()
        self._put_count = 0

    def put(self, frame: np.ndarray, frame_id: int | str = 0, timestamp: float | None = None) -> None:
        packet = FramePacket(np.asarray(frame), frame_id, time.time() if timestamp is None else float(timestamp))
        with self._lock:
            self._packet = packet
            self._put_count += 1

    def get(self) -> FramePacket | None:
        with self._lock:
            return self._packet

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._put_count


class LatestDepthBuffer:
    def __init__(self) -> None:
        self._state: DepthState | None = None
        self._lock = threading.Lock()
        self._update_count = 0

    def put(self, state: DepthState) -> None:
        with self._lock:
            self._state = state
            self._update_count += 1

    def get(self) -> DepthState | None:
        with self._lock:
            return self._state

    @property
    def update_count(self) -> int:
        with self._lock:
            return self._update_count


class DepthWorker:
    """Run a ``DepthAnythingProvider.compute``-compatible callable once/frame."""

    def __init__(self, infer: Callable[[np.ndarray, int | str, float], DepthState], frame_buffer: LatestFrameBuffer | None = None, depth_buffer: LatestDepthBuffer | None = None) -> None:
        self.frame_buffer = frame_buffer or LatestFrameBuffer()
        self.depth_buffer = depth_buffer or LatestDepthBuffer()
        self._infer = infer
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_processed: int | str | None = None
        self._last_error: str | None = None
        self._last_inference_ms = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="depth-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while self._running:
            packet = self.frame_buffer.get()
            if packet is None or packet.frame_id == self._last_processed:
                time.sleep(0.001)
                continue
            started = time.perf_counter()
            try:
                state = self._infer(packet.frame, packet.frame_id, packet.timestamp)
                self.depth_buffer.put(state)
                self._last_processed = packet.frame_id
                self._last_error = None
                self._last_inference_ms = (time.perf_counter() - started) * 1000.0
            except Exception as exc:  # keep live transport alive; expose error to diagnostics
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._last_processed = packet.frame_id

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms
