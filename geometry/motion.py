"""Renderer-independent motion state and optional OpenCV flow provider."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .warp import warp_field_backward


@dataclass(slots=True)
class MotionState:
    """Dense flow correspondence between two image frames.

    ``forward_flow`` maps previous pixels to current pixels. ``backward_flow``
    maps current pixels to previous pixels. Both are pixel-unit vectors with
    shape ``(H, W, 2)``.
    """

    source_frame_id: int | str
    target_frame_id: int | str
    timestamp: float
    forward_flow: np.ndarray
    backward_flow: np.ndarray
    valid_mask: np.ndarray | None = None
    forward_backward_error: np.ndarray | None = None
    flow_confidence: np.ndarray | None = None
    photometric_error: np.ndarray | None = None
    occlusion_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.forward_flow = np.asarray(self.forward_flow, dtype=np.float32)
        self.backward_flow = np.asarray(self.backward_flow, dtype=np.float32)
        if self.forward_flow.ndim != 3 or self.forward_flow.shape[-1] != 2:
            raise ValueError("forward_flow must have shape (H, W, 2)")
        if self.backward_flow.shape != self.forward_flow.shape:
            raise ValueError("backward_flow must match forward_flow shape")
        h, w = self.forward_flow.shape[:2]
        if self.valid_mask is None:
            self.valid_mask = np.isfinite(self.forward_flow).all(axis=-1) & np.isfinite(self.backward_flow).all(axis=-1)
        else:
            self.valid_mask = np.asarray(self.valid_mask, dtype=bool)
            if self.valid_mask.shape != (h, w):
                raise ValueError("valid_mask must have shape (H, W)")
        if self.forward_backward_error is None or self.flow_confidence is None:
            error, confidence, consistent = flow_consistency(self.forward_flow, self.backward_flow, self.valid_mask)
            if self.forward_backward_error is None:
                self.forward_backward_error = error
            if self.flow_confidence is None:
                self.flow_confidence = confidence
            self.valid_mask &= consistent
        else:
            self.forward_backward_error = np.asarray(self.forward_backward_error, dtype=np.float32)
            self.flow_confidence = np.asarray(self.flow_confidence, dtype=np.float32)
        for name in ("forward_backward_error", "flow_confidence", "photometric_error"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, dtype=np.float32)
                if value.shape != (h, w):
                    raise ValueError(f"{name} must have shape (H, W)")
                setattr(self, name, value)
        if self.occlusion_mask is not None:
            self.occlusion_mask = np.asarray(self.occlusion_mask, dtype=bool)
            if self.occlusion_mask.shape != (h, w):
                raise ValueError("occlusion_mask must have shape (H, W)")


def flow_consistency(
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    valid_mask: np.ndarray | None = None,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``F_prev_to_cur + F_cur_to_prev(at destination)`` confidence."""

    forward = np.asarray(forward_flow, dtype=np.float32)
    backward = np.asarray(backward_flow, dtype=np.float32)
    if forward.shape != backward.shape or forward.ndim != 3 or forward.shape[-1] != 2:
        raise ValueError("forward_flow and backward_flow must both have shape (H, W, 2)")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    # Sample backward flow at p + F_forward using a displacement field.
    backward_sample, sample_valid = warp_field_backward(backward, forward, interpolation="bilinear", fill_value=np.nan)
    error_vector = forward + backward_sample
    error = np.linalg.norm(np.nan_to_num(error_vector), axis=-1)
    valid = sample_valid & np.isfinite(forward).all(axis=-1) & np.isfinite(backward).all(axis=-1)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    confidence = np.exp(-((error / sigma) ** 2)).astype(np.float32)
    confidence = np.where(valid, confidence, 0.0)
    return error.astype(np.float32), confidence, valid


class OpticalFlowProvider:
    """Small provider boundary for CPU, NVIDIA, CUDA, or neural flow later."""

    def compute(
        self,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        source_frame_id: int | str,
        target_frame_id: int | str,
        timestamp: float,
    ) -> MotionState:
        raise NotImplementedError


class OpenCVFlowProvider(OpticalFlowProvider):
    """OpenCV DIS baseline with Farneback fallback."""

    def __init__(self, method: str = "dis", fb_sigma: float = 1.5) -> None:
        if method not in {"dis", "farneback"}:
            raise ValueError("method must be 'dis' or 'farneback'")
        self.method = method
        self.fb_sigma = fb_sigma

    @staticmethod
    def _cv2():
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("OpenCV is required for OpenCVFlowProvider") from exc
        return cv2

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        cv2 = OpenCVFlowProvider._cv2()
        array = np.asarray(frame)
        if array.ndim == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        if array.ndim != 2:
            raise ValueError("frames must have shape (H, W) or (H, W, C)")
        return array.astype(np.uint8, copy=False)

    def _compute_one(self, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        cv2 = self._cv2()
        if self.method == "dis" and hasattr(cv2, "DISOpticalFlow_create"):
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            return dis.calc(previous, current, None).astype(np.float32)
        return cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0).astype(np.float32)

    def compute(self, previous_frame: np.ndarray, current_frame: np.ndarray, source_frame_id: int | str, target_frame_id: int | str, timestamp: float) -> MotionState:
        previous = self._gray(previous_frame)
        current = self._gray(current_frame)
        if previous.shape != current.shape:
            raise ValueError("previous and current frames must have the same resolution")
        forward = self._compute_one(previous, current)
        backward = self._compute_one(current, previous)
        return MotionState(source_frame_id, target_frame_id, timestamp, forward, backward)
