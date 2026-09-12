"""Small dependency-free timing and memory diagnostics for geometry integration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import time

import numpy as np

from .state import GeometryState
from .motion import MotionState


def summarize_timings(samples_ms: list[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=np.float64)
    if values.size == 0:
        return {key: float("nan") for key in ("mean", "p50", "p95", "p99", "max")}
    return {"mean": float(values.mean()), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)), "max": float(values.max())}


@dataclass(slots=True)
class StageTimer:
    samples: dict[str, list[float]] = field(default_factory=dict)

    def measure(self, stage: str):
        return _TimerContext(self, stage)

    def report(self) -> dict[str, dict[str, float]]:
        return {stage: summarize_timings(values) for stage, values in self.samples.items()}


class _TimerContext:
    def __init__(self, owner: StageTimer, stage: str) -> None:
        self.owner, self.stage = owner, stage
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.owner.samples.setdefault(self.stage, []).append((time.perf_counter() - self.start) * 1000.0)
        return False


@dataclass(frozen=True, slots=True)
class GeometryDiagnostics:
    total_ms: float | None = None
    flow_ms: float | None = None
    warp_ms: float | None = None
    alignment_ms: float | None = None
    fusion_ms: float | None = None
    backprojection_ms: float | None = None
    normals_ms: float | None = None
    valid_geometry_percent: float | None = None
    valid_normal_percent: float | None = None
    history_acceptance_percent: float | None = None
    history_rejection_percent: float | None = None
    occlusion_percent: float | None = None
    disocclusion_percent: float | None = None
    mean_temporal_age: float | None = None
    mean_confidence: float | None = None
    alignment_success: bool | None = None
    alignment_model: str | None = None
    alignment_residual: float | None = None
    reset_reason: str | None = None
    depth_source_frame_delta: int | None = None


def geometry_state_nbytes(state: GeometryState) -> int:
    """Count retained NumPy buffers, avoiding double-counting aliases."""
    seen: set[int] = set()
    total = 0
    for item in fields(state):
        value = getattr(state, item.name)
        if isinstance(value, np.ndarray) and id(value) not in seen:
            seen.add(id(value))
            total += value.nbytes
    return total


def motion_state_nbytes(state: MotionState) -> int:
    """Count retained NumPy buffers in a motion state."""
    seen: set[int] = set()
    total = 0
    for item in fields(state):
        value = getattr(state, item.name)
        if isinstance(value, np.ndarray) and id(value) not in seen:
            seen.add(id(value))
            total += value.nbytes
    return total
