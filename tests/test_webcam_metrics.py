from __future__ import annotations

import pytest
import numpy as np

from tools.webcam_geometry_app import WebcamGeometrySession, _format_live_metrics


def test_live_metrics_report_callback_fps_and_latency() -> None:
    session = object.__new__(WebcamGeometrySession)
    session._metric_last_start = 10.0
    session._metric_processed_fps = None

    metrics = session._update_live_metrics(10.2, 250.0)

    assert metrics["processed_fps"] == pytest.approx(5.0)
    assert metrics["latency_ms"] == 250.0


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
