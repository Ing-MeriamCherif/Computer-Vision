"""Small utilities: intrinsics, sampling, profiling, FPS."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


def bilinear_sample(img: np.ndarray, u: float, v: float) -> float:
    """Bilinear sample of single-channel float image. Clamped to borders."""
    h, w = img.shape[:2]
    u = min(max(u, 0.0), w - 1.001)
    v = min(max(v, 0.0), h - 1.001)
    x0, y0 = int(u), int(v)
    dx, dy = u - x0, v - y0
    a = float(img[y0, x0])
    b = float(img[y0, x0 + 1])
    c = float(img[y0 + 1, x0])
    d = float(img[y0 + 1, x0 + 1])
    return a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * (1 - dx) * dy + d * dx * dy


class StageProfiler:
    """Per-stage latency recorder -> avg/min/max/p95 + JSON dump."""

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = {}

    def add(self, stage: str, ms: float) -> None:
        self.samples.setdefault(stage, []).append(ms)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for k, vals in self.samples.items():
            a = np.asarray(vals, dtype=np.float64)
            out[k] = {
                "avg_ms": float(a.mean()),
                "min_ms": float(a.min()),
                "max_ms": float(a.max()),
                "p95_ms": float(np.percentile(a, 95)),
                "n": int(len(a)),
            }
        return out


class FPSMeter:
    def __init__(self, window: int = 30) -> None:
        self.times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        now = time.perf_counter()
        self.times.append(now)
        if len(self.times) < 2:
            return 0.0
        dt = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / dt if dt > 0 else 0.0


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
