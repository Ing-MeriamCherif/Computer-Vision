"""UNIT: Metric calculation and view composition unit tests for legacy UI session.

Operates purely on synthetic in-memory arrays; requires no physical camera hardware.
"""

from __future__ import annotations

import pytest
import numpy as np

from tools.webcam_geometry_app import WebcamGeometrySession, _compose_live_view, _format_live_metrics


def test_live_metrics_report_callback_fps_and_latency() -> None:
    session = object.__new__(WebcamGeometrySession)
    session._metric_last_start = 10.0
    session._metric_processed_fps = None

    metrics = session._update_live_metrics(10.2, 250.0)

    assert metrics["processed_fps"] == pytest.approx(5.0)
    assert metrics["latency_ms"] == 250.0


def test_live_metrics_ignore_model_warmup_gap() -> None:
    session = object.__new__(WebcamGeometrySession)
    session._metric_last_start = 10.0
    session._metric_processed_fps = 30.0

    metrics = session._update_live_metrics(12.0, 2000.0)

    assert metrics["processed_fps"] is None


def test_live_metrics_formatter_contains_current_values() -> None:
    text = _format_live_metrics(
        {
            "frame_id": 7,
            "mode": "Phase 5 persistent",
            "latency_ms": 312.5,
            "metrics": {"processed_fps": 3.2},
            "depth": {"inference_ms": 101.0},
            "cuda_geometry": {"backprojection_ms": 1.5, "normals_ms": 2.5},
            "persistent": {"surfel_count": 1234},
        }
    )

    assert "3.20 FPS" in text
    assert "312.5 ms" in text
    assert "1,234 surfels" in text


def test_live_resolution_bounds_processing_without_changing_preview_contract() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    bounded = WebcamGeometrySession._live_resolution(frame)

    assert bounded.shape == (192, 256, 3)


def test_live_view_is_single_four_panel_video_frame() -> None:
    source = np.zeros((192, 256, 3), dtype=np.uint8)
    rendered = np.full((192, 256, 3), 64, dtype=np.uint8)
    stats = {
        "mode": "CUDA current geometry",
        "latency_ms": 32.0,
        "metrics": {"processed_fps": 30.0},
    }

    frame = _compose_live_view(source, (rendered, rendered, rendered, rendered, stats))

    assert frame.shape == (192, 256, 3)
    assert frame.dtype == np.uint8
