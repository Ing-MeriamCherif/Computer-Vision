"""Dedicated camera capture worker and LatestFrameSlot.

Maintains continuous physical camera capture on a dedicated thread, decoupled
from downstream processing and rendering deadlines.
"""

from __future__ import annotations

import threading
import time
import subprocess
from typing import Any

import numpy as np


class LatestFrameSlot:
    """Thread-safe single-frame slot with effective capacity 1.

    When downstream consumers are busy, newly arriving physical camera frames
    overwrite the slot and increment the dropped frame counter.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._frame: np.ndarray | None = None
        self._capture_sequence_id: int = -1
        self._capture_timestamp: float = 0.0
        self._has_new: bool = False
        self._dropped_count: int = 0
        self._total_arrived: int = 0
        self._consumed_count: int = 0
        self._renderer_reuses: int = 0
        self._last_consumed_sequence_id: int | None = None

    def put(
        self,
        frame: np.ndarray,
        capture_sequence_id: int,
        capture_timestamp: float,
    ) -> None:
        """Store a newly arrived physical camera frame."""
        with self._condition:
            if self._has_new:
                self._dropped_count += 1
            self._frame = frame
            self._capture_sequence_id = int(capture_sequence_id)
            self._capture_timestamp = float(capture_timestamp)
            self._has_new = True
            self._total_arrived += 1
            self._condition.notify_all()

    def get(self, timeout: float | None = 0.5) -> tuple[np.ndarray, int, float] | None:
        """Block until a new unconsumed frame arrives, or until timeout."""
        with self._condition:
            if not self._has_new:
                signaled = self._condition.wait(timeout=timeout)
                if not signaled or not self._has_new:
                    return None
            self._has_new = False
            if self._last_consumed_sequence_id == self._capture_sequence_id:
                self._renderer_reuses += 1
            else:
                self._consumed_count += 1
                self._last_consumed_sequence_id = self._capture_sequence_id
            assert self._frame is not None
            return self._frame, self._capture_sequence_id, self._capture_timestamp

    def get_latest(self) -> tuple[np.ndarray, int, float] | None:
        """Non-blocking retrieval of the current frame in the slot."""
        with self._lock:
            if self._frame is None:
                return None
            self._has_new = False
            if self._last_consumed_sequence_id == self._capture_sequence_id:
                self._renderer_reuses += 1
            else:
                self._consumed_count += 1
                self._last_consumed_sequence_id = self._capture_sequence_id
            return self._frame, self._capture_sequence_id, self._capture_timestamp

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    @property
    def total_arrived(self) -> int:
        with self._lock:
            return self._total_arrived


class CameraCaptureWorker:
    """Owns a single physical VideoCapture handle on a dedicated capture thread."""

    def __init__(
        self,
        device: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        backend: int | None = None,
        fourcc: str = "auto",
    ) -> None:
        self.device = device
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.backend = backend
        self.requested_fourcc = str(fourcc).upper()

        self.slot = LatestFrameSlot()
        self._thread: threading.Thread | None = None
        self._running = False
        self._cap: Any = None

        self.actual_width: int = width
        self.actual_height: int = height
        self.actual_fps: float = float(fps)
        self.actual_fourcc: int = 0
        self.capture_sequence_id: int = 0
        self.latest_exception: Exception | None = None
        self.last_capture_time: float = 0.0

        self._fps_samples: list[float] = []
        self._last_frame_ts: float | None = None

    def start(self) -> None:
        """Open camera and start the continuous capture thread."""
        import cv2

        if str(self.device).isdigit():
            dev_idx = int(self.device)
            cap_backend = self.backend or (cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0)
            self._cap = cv2.VideoCapture(dev_idx, cap_backend)
        elif str(self.device).startswith("/dev/"):
            cap_backend = self.backend or (cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0)
            self._cap = cv2.VideoCapture(str(self.device), cap_backend)
        else:
            self._cap = cv2.VideoCapture(self.device)

        if not self._cap.isOpened():
            raise RuntimeError(f"Unable to open physical camera: {self.device}")

        # Prefer compressed MJPG only when the V4L2 device advertises it. This
        # avoids forcing an unsupported format while keeping USB capture at
        # the requested 30 FPS on cameras that expose both MJPG and YUYV.
        if self.requested_fourcc in {"AUTO", "MJPG", "YUYV"}:
            chosen = self.requested_fourcc
            if chosen == "AUTO" and str(self.device).startswith("/dev/"):
                try:
                    probe = subprocess.run(
                        ["v4l2-ctl", "--device", str(self.device), "--list-formats-ext"],
                        capture_output=True, text=True, timeout=2.0, check=False,
                    ).stdout
                    chosen = "MJPG" if "'MJPG'" in probe else "YUYV" if "'YUYV'" in probe else ""
                except (OSError, subprocess.SubprocessError):
                    chosen = ""
            if chosen:
                code = cv2.VideoWriter_fourcc(*chosen)
                self._cap.set(cv2.CAP_PROP_FOURCC, code)

        # Set requested capture properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Query actual negotiated properties
        act_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        act_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        act_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if act_w > 0:
            self.actual_width = act_w
        if act_h > 0:
            self.actual_height = act_h
        if act_fps > 0:
            self.actual_fps = act_fps
        self.actual_fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC) or 0)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraCaptureWorker")
        self._thread.start()

    def _capture_loop(self) -> None:
        import cv2

        while self._running:
            try:
                ok, frame_bgr = self._cap.read()
                if not ok or frame_bgr is None:
                    time.sleep(0.005)
                    continue

                ts = time.monotonic()
                if self._last_frame_ts is not None:
                    delta = ts - self._last_frame_ts
                    if delta > 1e-4:
                        inst_fps = 1.0 / delta
                        self._fps_samples.append(inst_fps)
                        if len(self._fps_samples) > 30:
                            self._fps_samples.pop(0)
                        self.actual_fps = float(np.median(self._fps_samples))
                self._last_frame_ts = ts
                self.last_capture_time = ts

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                self.slot.put(frame_rgb, self.capture_sequence_id, ts)
                self.capture_sequence_id += 1
            except Exception as exc:
                self.latest_exception = exc
                time.sleep(0.01)

    def stop(self) -> None:
        """Deterministically stop capture and release the hardware handle."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self._running

    @property
    def dropped_frames(self) -> int:
        return self.slot.dropped_count

    @property
    def captured_frames(self) -> int:
        """Physical frames accepted from the camera device."""
        return self.slot.total_arrived

    @property
    def consumed_frames(self) -> int:
        with self.slot._lock:
            return self.slot._consumed_count

    @property
    def renderer_reuses(self) -> int:
        with self.slot._lock:
            return self.slot._renderer_reuses

    @property
    def overwritten_before_consumption(self) -> int:
        """Frames replaced in the capacity-one slot before a consumer read."""
        return self.slot.dropped_count
