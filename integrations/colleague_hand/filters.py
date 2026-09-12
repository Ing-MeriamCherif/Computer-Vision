"""Temporal filters. Start with EMA only (contexte.md rule: no Kalman yet)."""
from __future__ import annotations

import numpy as np


class EMAFilter:
    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha = float(alpha)
        self.value: np.ndarray | None = None

    def update(self, new_value: np.ndarray) -> np.ndarray:
        return self.update_dynamic(new_value, self.alpha)

    def update_dynamic(self, new_value: np.ndarray, alpha: float) -> np.ndarray:
        """Same EMA with per-call alpha. Low alpha = heavy smoothing.

        Used for edge-aware smoothing: near image borders / low detector
        confidence -> small alpha so the torch doesn't jump.
        """
        new_value = np.asarray(new_value, dtype=np.float64)
        a = float(min(max(alpha, 0.01), 1.0))
        if self.value is None:
            self.value = new_value.copy()
        else:
            self.value = a * new_value + (1.0 - a) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


def edge_factor(u: float, v: float, w: int, h: int, margin: float = 0.12) -> float:
    """0 at image center -> 1 at the border. Scales smoothing near edges."""
    nx = abs(u / w - 0.5) * 2.0
    ny = abs(v / h - 0.5) * 2.0
    m = max(nx, ny)
    if m <= (1.0 - margin):
        return 0.0
    return float((m - (1.0 - margin)) / margin)


def adaptive_alpha(base: float, confidence: float, edge: float) -> float:
    """base alpha reduced by low confidence and edge proximity."""
    return float(base * (0.35 + 0.65 * confidence) * (1.0 - 0.7 * edge))


class _LowPass:
    def __init__(self) -> None:
        self.y: float | None = None

    def __call__(self, x: float, alpha: float) -> float:
        if self.y is None:
            self.y = x
        else:
            self.y = alpha * x + (1.0 - alpha) * self.y
        return self.y


def _alpha(cutoff: float, dt: float) -> float:
    te = 2.0 * 3.141592653589793 * cutoff * max(dt, 1e-6)
    return te / (te + 1.0)


class OneEuroFilter:
    """Casiez et al. 1-euro filter: smooth when slow, responsive when fast.

    Better torch feel than fixed EMA: eliminates edge jitter without the lag
    a small fixed alpha would add on fast moves. One instance per dimension.
    mincutoff: steadiness at rest (lower = smoother). beta: responsiveness
    to speed (higher = less lag on fast moves). dcutoff: speed-estimate smoothing.
    """

    def __init__(self, mincutoff: float = 1.0, beta: float = 0.15,
                 dcutoff: float = 1.0) -> None:
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_f = _LowPass()
        self.dx_f = _LowPass()
        self.t_last: float | None = None

    def __call__(self, x: float, t: float) -> float:
        if self.t_last is None:
            self.t_last = t
            return self.x_f(x, 1.0)
        dt = max(t - self.t_last, 1e-3)
        self.t_last = t
        dx = (x - (self.x_f.y if self.x_f.y is not None else x)) / dt
        edx = self.dx_f(dx, _alpha(self.dcutoff, dt))
        cutoff = self.mincutoff + self.beta * abs(edx)
        return self.x_f(x, _alpha(cutoff, dt))

    def reset(self) -> None:
        self.x_f = _LowPass()
        self.dx_f = _LowPass()
        self.t_last = None


class OneEuroVec3:
    """3x OneEuro, one per axis, for the 3D light position."""

    def __init__(self, mincutoff: float = 1.0, beta: float = 0.15,
                 dcutoff: float = 1.0) -> None:
        self.fs = [OneEuroFilter(mincutoff, beta, dcutoff) for _ in range(3)]

    def __call__(self, v, t: float):
        import numpy as _np
        return _np.array([f(float(x), t) for f, x in zip(self.fs, v)],
                         dtype=_np.float64)

    def reset(self) -> None:
        for f in self.fs:
            f.reset()


def jump_rejected(new_uv, last_uv, max_jump_px: float) -> bool:
    """Outlier gate: single-frame teleport = misdetection, not motion."""
    if new_uv is None or last_uv is None:
        return False
    import math
    d = math.hypot(new_uv[0] - last_uv[0], new_uv[1] - last_uv[1])
    return d > max_jump_px


# Stub for later (Kalman per-landmark). Kept here so imports don't break.
class KalmanStub:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Kalman is deferred to Phase-2 (EMA first).")
