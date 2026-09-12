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
    total_ms: float = 0.0
    flow_ms: float = 0.0
    warp_ms: float = 0.0
    alignment_ms: float = 0.0
    fusion_ms: float = 0.0
    backprojection_ms: float = 0.0
    normals_ms: float = 0.0
    valid_geometry_percent: float = 0.0
    valid_normal_percent: float = 0.0
    history_acceptance_percent: float = 0.0
    history_rejection_percent: float = 0.0
    occlusion_percent: float = 0.0
    disocclusion_percent: float = 0.0
    mean_temporal_age: float = 0.0
    mean_confidence: float = 0.0
    alignment_success: bool = False
    alignment_model: str = "none"
    alignment_residual: float = float("inf")
    reset_reason: str = "none"
    depth_source_frame_delta: int = 0


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
