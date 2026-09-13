from __future__ import annotations

import numpy as np

from geometry import CameraModel, DepthState, DepthWorker, HandControlEngine, LatestDepthBuffer, LatestFrameBuffer, TorchGeometryBackend, create_hand_tracker, light_from_palm, sample_depth, shade_geometry
import time


def _geometry():
    camera = CameraModel(32, 24, 28.0, 28.0, 15.5, 11.5)
    depth = np.full((24, 32), 2.0, dtype=np.float32)
    state = TorchGeometryBackend("cpu").process_depth(depth, camera, frame_id=0, timestamp=0.0)
    return state


def test_hand_engine_unavailable_backend_is_safe():
    engine = HandControlEngine(backend="missing", max_hands=2)
    state = engine.update(np.zeros((24, 32, 3), dtype=np.uint8), 0.0, 0)
    assert not state.active
    assert state.backend == "unavailable"
    engine.close()


def test_colleague_tracker_is_real_upstream_backend():
    """The merged adapter must execute the preserved colleague tracker."""
    tracker = create_hand_tracker(model_path="models/hand_landmarker.task", backend="colleague", max_hands=2)
    try:
        assert tracker.name.startswith("colleague-mediapipe-")
    finally:
        tracker.close()


def test_colleague_adapter_preserves_mediapipe_handedness(monkeypatch):
    from integrations.colleague_hand import hand_tracker as upstream
    from geometry.hand_control import _ColleagueBackend

    class StubTracker:
        backend_name = "mediapipe-tasks"

        def process(self, _rgb):
            return upstream.HandResult(
                True,
                (10.0, 12.0),
                np.zeros((21, 2), dtype=np.float32),
                0.95,
                self.backend_name,
                18.0,
                "Left",
            )

        def close(self):
            return None

    monkeypatch.setattr(upstream, "create_tracker", lambda **_kwargs: StubTracker())
    adapter = _ColleagueBackend(None, 1)
    try:
        hands = adapter.process(np.zeros((24, 32, 3), dtype=np.uint8))
        assert len(hands) == 1
        assert hands[0].handedness == "Left"
    finally:
        adapter.close()


def test_sample_depth_uses_median_fallback_for_invalid_palm():
    depth = np.full((8, 8), 2.0, dtype=np.float32)
    valid = np.ones_like(depth, dtype=bool)
    depth[3, 3] = np.nan
    valid[3, 3] = False
    value, confidence = sample_depth(depth, valid, 3.0, 3.0)
    assert value == 2.0
    assert 0.0 < confidence < 1.0


def test_real_depth_palm_backprojects_and_shades():
    geometry = _geometry()
    light = light_from_palm(geometry, (15.5, 11.5), 0.9)
    assert light is not None
    assert light.position_camera_m[2] == 2.0
    image = np.full((24, 32, 3), 80, dtype=np.uint8)
    relit, stats = shade_geometry(image, geometry, [light])
    assert relit.shape == image.shape
    assert relit.dtype == np.uint8
    assert stats["lights"] == 1.0
    assert np.isfinite(relit).all()


def test_async_depth_worker_does_not_reprocess_same_frame():
    calls = []

    def infer(frame, frame_id, timestamp):
        calls.append(frame_id)
        return DepthState(np.ones((2, 2), dtype=np.float32), timestamp, frame_id, "relative")

    frames, depths = LatestFrameBuffer(), LatestDepthBuffer()
    worker = DepthWorker(infer, frames, depths)
    worker.start()
    frames.put(np.zeros((2, 2, 3), dtype=np.uint8), 4, 1.0)
    time.sleep(0.03)
    frames.put(np.zeros((2, 2, 3), dtype=np.uint8), 5, 1.1)
    time.sleep(0.03)
    worker.stop()
    assert calls == [4, 5]
    assert depths.get() is not None and depths.get().source_frame_id == 5
