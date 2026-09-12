"""Renderer-independent motion state and optional OpenCV flow provider."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .warp import warp_field_backward


@dataclass(slots=True)
class MotionState:
    """Dense flow correspondence between two image frames.

    ``forward_flow`` maps previous pixels to current pixels:
    ``p_cur = p_prev + forward_flow(p_prev)``.

    ``backward_flow`` maps current pixels to previous pixels:
    ``p_prev = p_cur + backward_flow(p_cur)``.

    Both are pixel-unit vectors with shape ``(H, W, 2)``.

    Diagnostics:
    - ``forward_backward_error``: consistency error in pixels, naturally evaluated
      on previous-frame coordinates: ``||forward_flow + backward_flow(p_prev + forward_flow)||``.
    - ``flow_confidence``: canonical single-source confidence in [0, 1] derived from
      consistency error using ``fb_sigma``.
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
    fb_sigma: float = 1.5

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if not np.isfinite(self.fb_sigma) or self.fb_sigma <= 0:
            raise ValueError("fb_sigma must be positive")
        self.forward_flow = np.asarray(self.forward_flow, dtype=np.float32)
        self.backward_flow = np.asarray(self.backward_flow, dtype=np.float32)
        if self.forward_flow.ndim != 3 or self.forward_flow.shape[-1] != 2:
            raise ValueError("forward_flow must have shape (H, W, 2)")
        if self.backward_flow.shape != self.forward_flow.shape:
            raise ValueError("backward_flow must match forward_flow shape")
        h, w = self.forward_flow.shape[:2]

        finite_flow = np.isfinite(self.forward_flow).all(axis=-1) & np.isfinite(self.backward_flow).all(axis=-1)
        if self.valid_mask is None:
            self.valid_mask = finite_flow
        else:
            self.valid_mask = np.array(self.valid_mask, dtype=bool, copy=True)
            if self.valid_mask.shape != (h, w):
                raise ValueError("valid_mask must have shape (H, W)")
            self.valid_mask &= finite_flow

        if self.forward_backward_error is None or self.flow_confidence is None:
            error, confidence, consistent = flow_consistency(
                self.forward_flow, self.backward_flow, self.valid_mask, sigma=self.fb_sigma
            )
            if self.forward_backward_error is None:
                self.forward_backward_error = error
            if self.flow_confidence is None:
                self.flow_confidence = confidence
            self.valid_mask &= consistent

        self.forward_backward_error = np.asarray(self.forward_backward_error, dtype=np.float32)
        if self.forward_backward_error.shape != (h, w):
            raise ValueError("forward_backward_error must have shape (H, W)")
        finite_error = np.isfinite(self.forward_backward_error) & (self.forward_backward_error >= 0.0)
        self.valid_mask &= finite_error
        self.forward_backward_error = np.where(
            self.valid_mask,
            np.nan_to_num(self.forward_backward_error, nan=np.inf, posinf=np.inf),
            np.inf,
        ).astype(np.float32)

        self.flow_confidence = np.asarray(self.flow_confidence, dtype=np.float32)
        if self.flow_confidence.shape != (h, w):
            raise ValueError("flow_confidence must have shape (H, W)")
        self.flow_confidence = np.clip(np.nan_to_num(self.flow_confidence, nan=0.0), 0.0, 1.0)
        self.flow_confidence = np.where(self.valid_mask, self.flow_confidence, 0.0).astype(np.float32)

        if self.photometric_error is not None:
            self.photometric_error = np.asarray(self.photometric_error, dtype=np.float32)
            if self.photometric_error.shape != (h, w):
                raise ValueError("photometric_error must have shape (H, W)")
            self.photometric_error = np.where(
                np.isfinite(self.photometric_error) & (self.photometric_error >= 0.0),
                self.photometric_error,
                np.inf,
            ).astype(np.float32)

        if self.occlusion_mask is not None:
            self.occlusion_mask = np.asarray(self.occlusion_mask, dtype=bool)
            if self.occlusion_mask.shape != (h, w):
                raise ValueError("occlusion_mask must have shape (H, W)")


def flow_consistency(
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    valid_mask: np.ndarray | None = None,
    sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``F_prev_to_cur + F_cur_to_prev(at destination)`` confidence."""

    forward = np.asarray(forward_flow, dtype=np.float32)
    backward = np.asarray(backward_flow, dtype=np.float32)
    if forward.shape != backward.shape or forward.ndim != 3 or forward.shape[-1] != 2:
        raise ValueError("forward_flow and backward_flow must both have shape (H, W, 2)")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    h, w = forward.shape[:2]
    # Sample backward flow at p + F_forward using a displacement field.
    backward_sample, sample_valid = warp_field_backward(backward, forward, interpolation="bilinear", fill_value=np.nan)
    error_vector = forward + backward_sample
    error = np.linalg.norm(np.nan_to_num(error_vector), axis=-1)
    valid = sample_valid & np.isfinite(forward).all(axis=-1) & np.isfinite(backward).all(axis=-1)
    if valid_mask is not None:
        vmask = np.asarray(valid_mask, dtype=bool)
        if vmask.shape != (h, w):
            raise ValueError("valid_mask must have shape (H, W)")
        valid &= vmask
    confidence = np.exp(-((error / sigma) ** 2)).astype(np.float32)
    confidence = np.where(valid, np.clip(confidence, 0.0, 1.0), 0.0).astype(np.float32)
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

    def __init__(self, method: str = "dis", fb_sigma: float = 1.5, flow_scale: float = 1.0) -> None:
        if method not in {"dis", "farneback"}:
            raise ValueError("method must be 'dis' or 'farneback'")
        if not np.isfinite(fb_sigma) or fb_sigma <= 0:
            raise ValueError("fb_sigma must be positive")
        if not np.isfinite(flow_scale) or not 0 < flow_scale <= 1:
            raise ValueError("flow_scale must be in (0, 1]")
        self.method = method
        self.fb_sigma = fb_sigma
        self.flow_scale = float(flow_scale)

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
        if array.ndim not in (2, 3):
            raise ValueError("frames must have shape (H, W) or (H, W, C)")
        if array.ndim == 3:
            if array.shape[-1] not in (1, 3, 4):
                raise ValueError(f"unsupported channel count {array.shape[-1]}")
            if array.shape[-1] == 1:
                array = array[..., 0]
            elif array.shape[-1] == 3:
                array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
            elif array.shape[-1] == 4:
                array = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)

        # Handle float [0, 1] vs float [0, 255] vs uint8 correctly
        if np.issubdtype(array.dtype, np.floating):
            array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
            max_val = float(np.max(array)) if array.size else 0.0
            if max_val <= 1.05:
                array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    def _compute_one(self, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        cv2 = self._cv2()
        if self.method == "dis" and hasattr(cv2, "DISOpticalFlow_create"):
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            return dis.calc(previous, current, None).astype(np.float32)
        return cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0).astype(np.float32)

    def compute(
        self,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        source_frame_id: int | str,
        target_frame_id: int | str,
        timestamp: float,
    ) -> MotionState:
        previous = self._gray(previous_frame)
        current = self._gray(current_frame)
        if previous.shape != current.shape:
            raise ValueError("previous and current frames must have the same resolution")
        if self.flow_scale < 1.0:
            cv2 = self._cv2()
            height, width = previous.shape
            scaled_width = max(2, int(round(width * self.flow_scale)))
            scaled_height = max(2, int(round(height * self.flow_scale)))
            previous_small = cv2.resize(previous, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
            current_small = cv2.resize(current, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
            scale_x, scale_y = width / scaled_width, height / scaled_height

            def restore(flow_small: np.ndarray) -> np.ndarray:
                restored = cv2.resize(flow_small, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                restored[..., 0] *= scale_x
                restored[..., 1] *= scale_y
                return restored

            forward = restore(self._compute_one(previous_small, current_small))
            backward = restore(self._compute_one(current_small, previous_small))
        else:
            forward = self._compute_one(previous, current)
            backward = self._compute_one(current, previous)
        return MotionState(
            source_frame_id,
            target_frame_id,
            timestamp,
            forward,
            backward,
            fb_sigma=self.fb_sigma,
        )
