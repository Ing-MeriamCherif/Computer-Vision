"""Asynchronous depth inference pipeline.

Architecture:

    Camera thread
         |
    LatestFrameBuffer (always newest frame, copied for safety)
         |
    DepthWorker thread (runs model.infer in background)
         |
    LatestDepthBuffer (always newest DepthState)
         |
    Renderer / consumer

The depth worker always processes the most recent available frame
and discards stale ones, so the renderer never blocks on inference.

Valid-mask convention:
    The current valid_mask is a structural validity mask — all finite
    normalized depth pixels are marked valid. It is NOT a learned
    confidence map. Depth Anything produces dense predictions for
    every pixel; the mask exists for backends that may return
    unreliable regions in the future.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from depth.model import DepthModel, DepthState


class LatestFrameBuffer:
    """Thread-safe buffer holding only the most recent camera frame.

    Frames are copied on write to prevent the camera from overwriting
    a frame while the worker is reading it.  OpenCV's cap.read()
    typically returns a fresh array each call, so the copy is a
    safety net rather than a necessity — measure before removing.
    """

    def __init__(self) -> None:
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._frame_count: int = 0

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame.copy()
            self._frame_count += 1

    def get(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count


class LatestDepthBuffer:
    """Thread-safe buffer holding only the most recent DepthState."""

    def __init__(self) -> None:
        self._state: DepthState | None = None
        self._lock = threading.Lock()
        self._update_count: int = 0

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
    """Background thread that runs depth inference on the newest frame.

    Usage:
        worker = DepthWorker(model, frame_buffer, depth_buffer)
        worker.start()

        # In camera thread:
        frame_buffer.put(camera.read())

        # In renderer thread:
        state = depth_buffer.get()
        if state is not None:
            age_ms = (time.time() - state.timestamp) * 1000
            if age_ms < 200:
                render(state.depth_map)

        worker.stop()
    """

    def __init__(
        self,
        model: DepthModel,
        frame_buffer: LatestFrameBuffer,
        depth_buffer: LatestDepthBuffer,
    ) -> None:
        self._model = model
        self._frame_buffer = frame_buffer
        self._depth_buffer = depth_buffer
        self._running = False
        self._thread: threading.Thread | None = None

        # Statistics
        self._processed_count: int = 0
        self._skipped_count: int = 0
        self._last_inference_ms: float = 0.0
        self._last_state_timestamp: float = 0.0

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while self._running:
            frame = self._frame_buffer.get()
            if frame is None:
                time.sleep(0.001)
                continue

            state = self._model.infer(frame)
            self._last_inference_ms = state.inference_ms
            self._last_state_timestamp = state.timestamp
            self._processed_count += 1
            self._depth_buffer.put(state)

    # --- Statistics ---

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    @property
    def processed_count(self) -> int:
        return self._processed_count

    @property
    def skipped_count(self) -> int:
        return self._skipped_count

    @property
    def last_state_timestamp(self) -> float:
        return self._last_state_timestamp

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stats_dict(self) -> dict:
        return {
            "alive": self.is_alive,
            "processed": self._processed_count,
            "skipped": self._skipped_count,
            "last_inference_ms": self._last_inference_ms,
        }


def validate_depth_state(state: DepthState) -> dict:
    """Check depth properties and return a diagnostics dict.

    The current valid_mask is a structural validity mask: all finite
    normalized depth pixels are considered valid. It is NOT a learned
    confidence map.

    Useful for testing that the model output is well-formed:
      - shape is (H, W)
      - values are finite (no NaN / Inf)
      - metric mode: values in meters; relative mode: [0, 1]
    """
    depth = state.depth_map
    result = {
        "shape": depth.shape,
        "dtype": str(depth.dtype),
        "min": float(depth.min()),
        "max": float(depth.max()),
        "mean": float(depth.mean()),
        "has_nan": bool(np.isnan(depth).any()),
        "has_inf": bool(np.isinf(depth).any()),
        "scale_mode": state.scale_mode,
        "backend": state.backend_name,
        "inference_ms": state.inference_ms,
    }

    if state.valid_mask is not None:
        mask = state.valid_mask
        result["valid_mask_shape"] = mask.shape
        result["valid_ratio"] = float(mask.mean())
        result["valid_mask_dtype"] = str(mask.dtype)
        result["valid_mask_note"] = "structural (all-finite), not learned confidence"
    else:
        result["valid_mask_shape"] = None
        result["valid_ratio"] = None

    return result
