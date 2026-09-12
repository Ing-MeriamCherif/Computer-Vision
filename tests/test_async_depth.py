"""Tests for the async depth pipeline.

Run:  python -m pytest tests/test_async_depth.py -v

No GPU / camera / model weights needed — uses synthetic frames and a
fake backend so every test runs anywhere in < 2 seconds.
"""

from __future__ import annotations

import time
import threading
import numpy as np

from depth.async_depth import (
    DepthWorker,
    LatestDepthBuffer,
    LatestFrameBuffer,
    validate_depth_state,
)
from depth.config import DepthConfig, DEPTH_CONFIG
from depth.model import DepthModel, DepthState


# ---------------------------------------------------------------------------
# Fake backend for unit tests — no torch, no HF pipeline
# ---------------------------------------------------------------------------

class _FakeBackend:
    """Deterministic depth backend that returns a gradient depth map."""

    scale_mode = "relative"

    def __init__(self, device: str = "cpu", input_size: int = 420, fp16: bool = False):
        self.input_size = input_size

    def warmup(self, iterations: int = 3) -> None:
        pass

    def predict(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        gradient = np.linspace(0.0, 1.0, h * w, dtype=np.float32).reshape(h, w)
        valid_mask = np.ones((h, w), dtype=bool)
        return gradient, valid_mask


def _make_model() -> DepthModel:
    """Build a DepthModel backed by _FakeBackend (no downloads)."""
    model = DepthModel.__new__(DepthModel)
    model._backend = _FakeBackend()
    model._backend_name = "fake"
    return model


def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests: LatestFrameBuffer
# ---------------------------------------------------------------------------

class TestLatestFrameBuffer:
    def test_put_get(self):
        buf = LatestFrameBuffer()
        assert buf.get() is None
        frame = _make_frame()
        buf.put(frame)
        result = buf.get()
        assert np.array_equal(result, frame)

    def test_put_copies_frame(self):
        """put() copies the frame so camera can't overwrite it."""
        buf = LatestFrameBuffer()
        frame = _make_frame()
        buf.put(frame)
        # Mutate original — buffer should be unaffected
        frame[:] = 0
        result = buf.get()
        assert result is not None
        assert result.mean() > 0  # still has the original data

    def test_overwrites_old(self):
        buf = LatestFrameBuffer()
        f1 = _make_frame()
        f2 = _make_frame()
        buf.put(f1)
        buf.put(f2)
        assert np.array_equal(buf.get(), f2)

    def test_frame_count(self):
        buf = LatestFrameBuffer()
        assert buf.frame_count == 0
        buf.put(_make_frame())
        buf.put(_make_frame())
        assert buf.frame_count == 2


# ---------------------------------------------------------------------------
# Tests: LatestDepthBuffer
# ---------------------------------------------------------------------------

class TestLatestDepthBuffer:
    def test_put_get(self):
        buf = LatestDepthBuffer()
        assert buf.get() is None
        state = DepthState(
            depth_map=np.zeros((10, 10), dtype=np.float32),
            timestamp=time.time(),
        )
        buf.put(state)
        assert buf.get() is state

    def test_update_count(self):
        buf = LatestDepthBuffer()
        assert buf.update_count == 0
        for _ in range(5):
            buf.put(DepthState(depth_map=np.zeros((10, 10)), timestamp=time.time()))
        assert buf.update_count == 5


# ---------------------------------------------------------------------------
# Tests: DepthWorker
# ---------------------------------------------------------------------------

class TestDepthWorker:
    def test_produces_depth(self):
        model = _make_model()
        fb = LatestFrameBuffer()
        db = LatestDepthBuffer()
        worker = DepthWorker(model, fb, db)

        worker.start()
        fb.put(_make_frame(480, 640))
        time.sleep(0.1)

        state = db.get()
        worker.stop()

        assert state is not None
        assert isinstance(state, DepthState)
        assert state.depth_map.shape == (480, 640)
        assert state.depth_map.dtype == np.float32

    def test_discards_stale_frames(self):
        """Worker should always process the newest frame, not queue old ones."""
        model = _make_model()
        fb = LatestFrameBuffer()
        db = LatestDepthBuffer()
        worker = DepthWorker(model, fb, db)

        worker.start()

        # Put 50 different-sized frames — only the last one matters
        for i in range(50):
            sz = 100 + i  # 100, 101, ..., 149
            fb.put(_make_frame(sz, sz))
        time.sleep(0.1)

        state = db.get()
        worker.stop()

        assert state is not None
        # Depth map should match the LAST frame's size, not an earlier one
        assert state.depth_map.shape == (149, 149)

    def test_stop_halts_worker(self):
        model = _make_model()
        fb = LatestFrameBuffer()
        db = LatestDepthBuffer()
        worker = DepthWorker(model, fb, db)

        worker.start()
        assert worker.is_alive

        worker.stop()
        time.sleep(0.05)
        assert not worker.is_alive


# ---------------------------------------------------------------------------
# Tests: validate_depth_state
# ---------------------------------------------------------------------------

class TestValidateDepthState:
    def test_valid_relative(self):
        depth = np.random.rand(480, 640).astype(np.float32)
        mask = np.ones((480, 640), dtype=bool)
        state = DepthState(
            depth_map=depth,
            timestamp=time.time(),
            scale_mode="relative",
            valid_mask=mask,
            backend_name="test",
            inference_ms=12.3,
        )
        diag = validate_depth_state(state)

        assert diag["shape"] == (480, 640)
        assert diag["scale_mode"] == "relative"
        assert 0.0 <= diag["min"] <= diag["max"] <= 1.0
        assert diag["valid_ratio"] == 1.0
        assert not diag["has_nan"]
        assert not diag["has_inf"]
        assert diag["inference_ms"] == 12.3

    def test_catches_nan(self):
        depth = np.zeros((10, 10), dtype=np.float32)
        depth[5, 5] = float("nan")
        state = DepthState(depth_map=depth, timestamp=time.time())
        diag = validate_depth_state(state)
        assert diag["has_nan"]

    def test_no_valid_mask(self):
        state = DepthState(
            depth_map=np.zeros((10, 10), dtype=np.float32),
            timestamp=time.time(),
            valid_mask=None,
        )
        diag = validate_depth_state(state)
        assert diag["valid_ratio"] is None


# ---------------------------------------------------------------------------
# Tests: DepthConfig
# ---------------------------------------------------------------------------

class TestDepthConfig:
    def test_defaults(self):
        cfg = DepthConfig()
        assert cfg.backend == "depth_anything_v2_small"
        assert cfg.device == "cuda"
        assert cfg.fp16 is True
        assert cfg.input_size == 420
        assert cfg.target_fps == 30

    def test_frozen(self):
        cfg = DepthConfig()
        try:
            cfg.input_size = 518  # type: ignore[misc]
            assert False, "should be frozen"
        except AttributeError:
            pass

    def test_singleton(self):
        assert DEPTH_CONFIG.input_size == 420


# ---------------------------------------------------------------------------
# Integration: end-to-end async with fake model
# ---------------------------------------------------------------------------

class TestAsyncIntegration:
    def test_full_pipeline(self):
        """Simulate: camera -> frame_buffer -> worker -> depth_buffer -> renderer."""
        model = _make_model()
        fb = LatestFrameBuffer()
        db = LatestDepthBuffer()
        worker = DepthWorker(model, fb, db)

        worker.start()

        # Simulate camera producing 30 frames
        for i in range(30):
            fb.put(_make_frame(240, 320))
            time.sleep(0.005)

        time.sleep(0.1)
        state = db.get()
        worker.stop()

        assert state is not None
        diag = validate_depth_state(state)
        assert diag["shape"] == (240, 320)
        assert diag["valid_ratio"] == 1.0
        assert not diag["has_nan"]
        assert db.update_count > 0
