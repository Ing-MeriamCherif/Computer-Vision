"""Adaptive, tracking-only enhancement for dim camera frames."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class TrackingFrame:
    rgb: np.ndarray
    luminance: float
    gamma: float
    clahe_active: bool
    elapsed_ms: float


class AdaptiveLowLightPreprocessor:
    """Lift dim tracking inputs without touching the user-visible camera frame."""

    def __init__(self, *, threshold: float | None = None, darkest_luma: float = 34.0, max_gamma: float = 1.65) -> None:
        configured = os.getenv("HAND_LOW_LIGHT_THRESHOLD", "88")
        self.threshold = float(configured) if threshold is None else float(threshold)
        self.darkest_luma = max(1.0, min(float(darkest_luma), self.threshold - 1.0))
        self.max_gamma = max(1.0, float(max_gamma))
        self._clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self._lut_cache: dict[float, np.ndarray] = {}

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    def _gamma_lut(self, gamma: float) -> np.ndarray:
        quantized = round(float(gamma) * 20.0) / 20.0
        lut = self._lut_cache.get(quantized)
        if lut is None:
            values = np.arange(256, dtype=np.float32) / 255.0
            lut = np.clip(np.rint(np.power(values, 1.0 / quantized) * 255.0), 0, 255).astype(np.uint8)
            self._lut_cache[quantized] = lut
        return lut

    def process(self, rgb: np.ndarray) -> TrackingFrame:
        started = time.perf_counter()
        source = np.asarray(rgb, dtype=np.uint8)
        if source.ndim != 3 or source.shape[2] < 3:
            raise ValueError("tracking frame must have shape (height, width, 3+)")
        source = source[..., :3]
        luma = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        luminance = float(np.mean(luma, dtype=np.float64))
        if luminance >= self.threshold:
            return TrackingFrame(source, luminance, 1.0, False, (time.perf_counter() - started) * 1000.0)

        strength = self._smoothstep((self.threshold - luminance) / (self.threshold - self.darkest_luma))
        gamma = 1.0 + (self.max_gamma - 1.0) * strength
        lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
        lifted = cv2.LUT(lab[..., 0], self._gamma_lut(gamma))
        local = self._clahe.apply(lifted)
        l_mix = 0.32 * strength
        lab[..., 0] = cv2.addWeighted(lifted, 1.0 - l_mix, local, l_mix, 0.0)
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return TrackingFrame(
            enhanced,
            luminance,
            gamma,
            True,
            (time.perf_counter() - started) * 1000.0,
        )


__all__ = ["AdaptiveLowLightPreprocessor", "TrackingFrame"]
