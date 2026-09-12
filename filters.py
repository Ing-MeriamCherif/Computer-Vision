"""Temporal filters. Start with EMA only (contexte.md rule: no Kalman yet)."""
from __future__ import annotations

import numpy as np


class EMAFilter:
    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha = float(alpha)
        self.value: np.ndarray | None = None

    def update(self, new_value: np.ndarray) -> np.ndarray:
        new_value = np.asarray(new_value, dtype=np.float64)
        if self.value is None:
            self.value = new_value.copy()
        else:
            self.value = self.alpha * new_value + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


# Stub for later (Kalman per-landmark). Kept here so imports don't break.
class KalmanStub:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Kalman is deferred to Phase-2 (EMA first).")
