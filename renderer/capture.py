"""Threaded, latest-frame webcam capture without a growing frame queue."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable

import numpy as np


@dataclass(frozen=True, slots=True)
class CapturedRGBFrame:
    """One immutable-by-convention RGB frame published by the capture worker."""

    sequence: int
    rgb: np.ndarray
    captured_at_s: float


@dataclass(frozen=True, slots=True)
class LatestFrameStats:
    frames_captured: int
    frames_replaced: int
    latest_sequence: int
    latest_capture_time_s: float | None


class LatestFrameBuffer:
    """Thread-safe single-slot mailbox. Publishers must not mutate published arrays."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: CapturedRGBFrame | None = None
        self._frames_captured = 0
        self._frames_replaced = 0
        self._last_consumed_sequence = 0

    def publish(self, rgb: np.ndarray, captured_at_s: float | None = None) -> CapturedRGBFrame:
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
            raise TypeError("captured RGB frame must be a uint8 numpy array")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2], default=0) <= 0:
            raise ValueError("captured RGB frame must have shape HxWx3")
        timestamp = time.perf_counter() if captured_at_s is None else float(captured_at_s)
        if not np.isfinite(timestamp):
            raise ValueError("captured_at_s must be finite")

        with self._lock:
            self._frames_captured += 1
            sequence = self._frames_captured
            if self._latest is not None and self._latest.sequence > self._last_consumed_sequence:
                self._frames_replaced += 1
            self._latest = CapturedRGBFrame(sequence, rgb, timestamp)
            return self._latest

    def get_latest(self) -> CapturedRGBFrame | None:
        """Return the newest frame reference and mark its sequence consumed."""
        with self._lock:
            frame = self._latest
            if frame is not None:
                self._last_consumed_sequence = max(self._last_consumed_sequence, frame.sequence)
            return frame

    def stats(self) -> LatestFrameStats:
        with self._lock:
            return LatestFrameStats(
                frames_captured=self._frames_captured,
                frames_replaced=self._frames_replaced,
                latest_sequence=self._latest.sequence if self._latest else 0,
                latest_capture_time_s=self._latest.captured_at_s if self._latest else None,
            )


class WebcamCaptureWorker:
    """Own a VideoCapture on a dedicated thread and publish only its newest RGB frame."""

    def __init__(
        self,
        camera_index: int = 0,
        *,
        capture_factory: Callable[[int], object] | None = None,
        cv2_module=None,
    ) -> None:
        if isinstance(camera_index, bool) or not isinstance(camera_index, int) or camera_index < 0:
            raise ValueError("camera_index must be a non-negative integer")
        self.camera_index = camera_index
        self.frames = LatestFrameBuffer()
        self._capture_factory = capture_factory
        self._cv2 = cv2_module
        self._stop_event = threading.Event()
        self._first_frame_event = threading.Event()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._capture = None
        self._error: BaseException | None = None
        self._released = False
        self.reported_width = 0
        self.reported_height = 0
        self.reported_fps = 0.0
        self.received_width = 0
        self.received_height = 0

    @property
    def cv2(self):
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:  # pragma: no cover - environment-dependent message
                raise RuntimeError("OpenCV is required for webcam input; install requirements.txt") from exc
            self._cv2 = cv2
        return self._cv2

    def start(self, timeout_s: float = 10.0) -> "WebcamCaptureWorker":
        if self._thread is not None:
            raise RuntimeError("webcam capture worker can only be started once")
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"webcam-capture-{self.camera_index}",
            daemon=False,
        )
        self._thread.start()
        if not self._first_frame_event.wait(timeout_s):
            self.stop()
            raise TimeoutError(f"timed out waiting for webcam index {self.camera_index} to produce a frame")
        self.raise_if_failed()
        return self

    def _capture_loop(self) -> None:
        capture = None
        try:
            cv2 = self.cv2
            capture = self._capture_factory(self.camera_index) if self._capture_factory else cv2.VideoCapture(self.camera_index)
            self._capture = capture
            if not capture.isOpened():
                raise RuntimeError(f"could not open webcam at camera index {self.camera_index}")
            self.reported_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            self.reported_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.reported_fps = float(capture.get(cv2.CAP_PROP_FPS))

            while not self._stop_event.is_set():
                ok, frame_bgr = capture.read()
                if self._stop_event.is_set():
                    break
                if not ok or frame_bgr is None:
                    raise RuntimeError(f"webcam index {self.camera_index} stopped returning frames")
                if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
                    raise RuntimeError(f"webcam returned an unsupported frame: {frame_bgr.shape}, {frame_bgr.dtype}")

                # This single color conversion creates the RGB buffer owned by the mailbox.
                # It is not copied again when published; consumers only hold a reference.
                frame_rgb = self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)
                self.received_height, self.received_width = frame_rgb.shape[:2]
                frame_rgb.setflags(write=False)
                self.frames.publish(frame_rgb, time.perf_counter())
                self._first_frame_event.set()
        except BaseException as exc:
            with self._state_lock:
                self._error = exc
            self._first_frame_event.set()
        finally:
            if capture is not None:
                try:
                    capture.release()
                finally:
                    with self._state_lock:
                        self._released = True

    def raise_if_failed(self) -> None:
        with self._state_lock:
            error = self._error
        if error is not None and not self._stop_event.is_set():
            raise RuntimeError(f"webcam capture worker failed: {type(error).__name__}: {error}") from error

    @property
    def released(self) -> bool:
        with self._state_lock:
            return self._released

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_s)
            if thread.is_alive():
                raise TimeoutError(f"webcam capture thread did not stop within {timeout_s:.1f}s")

