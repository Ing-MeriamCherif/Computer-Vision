"""Frozen baseline configuration for the depth pipeline.

This is the single source of truth for the production depth setup.
Change values here, then update all consumers.

Selected baseline (2026-09-12 benchmark):
    Model:     Depth Anything V2 Small
    Input:     420x420
    Device:    CUDA
    Precision: FP16
    Latency:   32.2 ms median / 42.4 ms p95
    FPS:       31.1
    VRAM:      143 MB
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DepthConfig:
    backend: str = "depth_anything_v2_small"
    device: str = "cuda"
    fp16: bool = True
    input_size: int = 420
    target_fps: int = 30
    metric: bool = False  # True = metric models (meters), False = relative (0-1)


# Importable singleton — use `from depth.config import DEPTH_CONFIG`.
DEPTH_CONFIG = DepthConfig()
