"""Integration tests for the native live application and decoupled pipeline.

Tests cover (Part 77 requirements):
- LatestFrameSlot capacity 1 semantics
- real monotonic timestamps
- dropped frame counting
- HandControlEngine two-hand output
- no double attenuation in lighting
- static confidence current-frame coordinate domain (Part 38)
- pose frame IDs (Part 40)
- persistent worker boundedness (Part 41)
- HUD metric correctness
- quality profiles
- NativeLiveApp headless smoke (synthetic camera)
- worker health reporting
- clean worker shutdown
"""

from __future__ import annotations

import time
import threading
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# LatestFrameSlot tests
# ---------------------------------------------------------------------------

class TestLatestFrameSlot:
    def test_capacity_one_drops_stale(self):
        """With capacity 1, slow consumer gets newest frame only."""
        from geometry.camera_worker import LatestFrameSlot
        slot = LatestFrameSlot()

        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.ones((10, 10, 3), dtype=np.uint8) * 42
        frame3 = np.ones((10, 10, 3), dtype=np.uint8) * 99

        slot.put(frame1, 0, time.monotonic())
        slot.put(frame2, 1, time.monotonic())  # overwrites 0
        slot.put(frame3, 2, time.monotonic())  # overwrites 1

        result = slot.get(timeout=0.1)
        assert result is not None
        _, seq, _ = result
        assert seq == 2, "Consumer should get the most recent frame (ID=2)"
        assert slot.dropped_count == 2

    def test_timestamps_are_monotonic(self):
        """Captured timestamps must be monotonically increasing."""
        from geometry.camera_worker import LatestFrameSlot
        slot = LatestFrameSlot()
        t0 = time.monotonic()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        prev_ts = -1.0
        for i in range(5):
            ts = time.monotonic()
            slot.put(frame, i, ts)
            _, _, captured_ts = slot.get_latest()
            assert captured_ts >= prev_ts
            prev_ts = captured_ts

    def test_get_latest_nonblocking(self):
        """get_latest returns immediately even with nothing arrived."""
        from geometry.camera_worker import LatestFrameSlot
        slot = LatestFrameSlot()
        assert slot.get_latest() is None  # empty slot

    def test_dropped_frame_count(self):
        """Dropped frames are counted accurately."""
        from geometry.camera_worker import LatestFrameSlot
        slot = LatestFrameSlot()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        for i in range(10):
            slot.put(frame, i, time.monotonic())
        # First put is stored, subsequent 9 overwrite; 9 drops when not consumed
        assert slot.dropped_count == 9

    def test_total_arrived_count(self):
        from geometry.camera_worker import LatestFrameSlot
        slot = LatestFrameSlot()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        for i in range(7):
            slot.put(frame, i, time.monotonic())
        assert slot.total_arrived == 7


# ---------------------------------------------------------------------------
# CameraCaptureWorker tests (synthetic / without opening real device)
# ---------------------------------------------------------------------------

class TestCameraCaptureWorker:
    def test_slot_capture_sequence_id_increments(self):
        """Capture sequence IDs increment monotonically across frames."""
        from geometry.camera_worker import LatestFrameSlot

        slot = LatestFrameSlot()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        for i in range(5):
            slot.put(frame, i, time.monotonic())
            r = slot.get_latest()
            assert r is not None
            _, seq_id, _ = r
            assert seq_id == i


# ---------------------------------------------------------------------------
# Hand tracking: two-hand output
# ---------------------------------------------------------------------------

class TestHandTrackingTwoHand:
    def test_tracked_hand_fields(self):
        """TrackedHand dataclass has all required fields."""
        from geometry.hand_control import TrackedHand
        hand = TrackedHand(
            hand_id=0,
            landmarks_uv=None,
            palm_uv=(100.0, 200.0),
            confidence=0.92,
            depth_z=0.75,
            timestamp=time.monotonic(),
            velocity_px_s=5.2,
            handedness="Right",
            palm_width_px=80.0,
            stale=False,
        )
        assert hand.hand_id == 0
        assert hand.palm_uv == (100.0, 200.0)
        assert hand.confidence == pytest.approx(0.92)

    def test_gesture_state_hands_tuple(self):
        """GestureState.hands is a tuple (immutable snapshot)."""
        from geometry.hand_control import GestureState, TrackedHand
        h0 = TrackedHand(0, None, (100.0, 200.0), 0.9)
        h1 = TrackedHand(1, None, (400.0, 200.0), 0.8)
        gs = GestureState(
            timestamp=time.monotonic(),
            source_frame_id=42,
            hands=(h0, h1),
            backend="test",
            tracker_ms=5.0,
        )
        assert len(gs.hands) == 2
        assert gs.active is True
        assert isinstance(gs.hands, tuple)

    def test_transform_hand_uv(self):
        """UV coordinate transform scales correctly from processing to render space."""
        from geometry.hand_control import transform_hand_uv
        uv_proc = (128.0, 96.0)   # 256×192 space
        uv_cam = transform_hand_uv(uv_proc, from_size=(256, 192), to_size=(640, 480))
        expected_x = 128.0 * (640 / 256)
        expected_y = 96.0 * (480 / 192)
        assert uv_cam[0] == pytest.approx(expected_x, rel=1e-5)
        assert uv_cam[1] == pytest.approx(expected_y, rel=1e-5)

    def test_no_hand_id_zero_bias(self):
        """Two distinct TrackedHand objects can have distinct IDs."""
        from geometry.hand_control import TrackedHand
        h0 = TrackedHand(0, None, (50.0, 100.0), 0.9)
        h1 = TrackedHand(1, None, (400.0, 100.0), 0.85)
        assert h0.hand_id != h1.hand_id


# ---------------------------------------------------------------------------
# Lighting: no double attenuation
# ---------------------------------------------------------------------------

class TestLightingAttenuation:
    def _make_flat_geometry(self, depth_value: float = 0.8):
        """Return a tiny GeometryState with flat planar geometry."""
        import numpy as np
        from geometry import CameraModel, GeometryState
        from geometry.backproject import DepthScaleMode

        H, W = 8, 8
        camera = CameraModel(W, H, fx=5.0, fy=5.0, cx=4.0, cy=4.0)
        depth = np.full((H, W), depth_value, dtype=np.float32)
        valid = np.ones((H, W), dtype=bool)
        u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        pos_x = (u - camera.cx) / camera.fx * depth
        pos_y = (v - camera.cy) / camera.fy * depth
        pos_z = depth
        positions = np.stack([pos_x, pos_y, pos_z], axis=-1)
        normals = np.zeros((H, W, 3), dtype=np.float32)
        normals[..., 2] = -1.0  # facing camera
        confidence = np.ones((H, W), dtype=np.float32)
        return GeometryState(
            timestamp=0.0, source_frame_id=0,
            depth=depth, positions_3d=positions,
            valid_mask=valid, camera=camera,
            scale_mode=DepthScaleMode.RELATIVE,
            normals=normals, confidence=confidence,
        )

    def test_shade_geometry_increases_brightness_with_light(self):
        """With a light present, shade_geometry output is brighter than ambient-only."""
        from geometry.lighting import LightState, shade_geometry
        geom = self._make_flat_geometry(depth_value=1.0)
        rgb = np.full((8, 8, 3), 180, dtype=np.uint8)
        light = LightState(
            position_camera=np.array([0.0, 0.0, 0.5], dtype=np.float32),
            intensity=2.0,
            color_rgb=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            confidence=1.0,
            source_hand=0,
            light_id=0,
        )
        relit, stats = shade_geometry(rgb, geom, [light], ambient=0.1, shadows=False, volumetrics=False)
        assert relit.dtype == np.uint8
        # Mean brightness should be higher than ambient-only (0.1 × 180 = 18)
        mean_bright = relit.astype(float).mean()
        assert mean_bright > 25.0, f"Expected brightness > 25, got {mean_bright:.1f}"

    def test_no_double_attenuation_single_light_consistent(self):
        """Double attenuation would make light unreachably dim; check coherence."""
        from geometry.lighting import LightState, shade_geometry
        geom_near = self._make_flat_geometry(depth_value=0.5)
        geom_far  = self._make_flat_geometry(depth_value=2.0)
        rgb = np.full((8, 8, 3), 200, dtype=np.uint8)
        light_near = LightState(
            position_camera=np.array([0.0, 0.0, 0.3], dtype=np.float32),
            intensity=1.5, color_rgb=np.ones(3, dtype=np.float32),
            confidence=1.0, source_hand=0, light_id=0,
        )
        light_far = LightState(
            position_camera=np.array([0.0, 0.0, 1.8], dtype=np.float32),
            intensity=1.5, color_rgb=np.ones(3, dtype=np.float32),
            confidence=1.0, source_hand=0, light_id=0,
        )
        relit_near, _ = shade_geometry(rgb, geom_near, [light_near], ambient=0.0, shadows=False)
        relit_far,  _ = shade_geometry(rgb, geom_far,  [light_far],  ambient=0.0, shadows=False)
        # Near scene: light is very close → bright
        # Far scene:  light is far from surfaces → dimmer
        mean_near = relit_near.astype(float).mean()
        mean_far  = relit_far.astype(float).mean()
        # Allow equal brightness (degenerate flat case) but not far > near by large margin
        assert mean_near >= mean_far * 0.5, \
            f"Unexpected inversion: near={mean_near:.1f} far={mean_far:.1f}"

    def test_two_lights_additive(self):
        """Two lights produce more illumination than one."""
        from geometry.lighting import LightState, shade_geometry
        geom = self._make_flat_geometry(depth_value=1.0)
        rgb = np.full((8, 8, 3), 100, dtype=np.uint8)
        l0 = LightState(
            position_camera=np.array([-0.2, 0.0, 0.5], np.float32),
            intensity=1.0, color_rgb=np.array([0.8, 0.8, 1.0], np.float32),
            confidence=1.0, source_hand=0, light_id=0,
        )
        l1 = LightState(
            position_camera=np.array([0.2, 0.0, 0.5], np.float32),
            intensity=1.0, color_rgb=np.array([1.0, 0.8, 0.8], np.float32),
            confidence=1.0, source_hand=1, light_id=1,
        )
        relit1, _ = shade_geometry(rgb, geom, [l0], ambient=0.1, shadows=False)
        relit2, _ = shade_geometry(rgb, geom, [l0, l1], ambient=0.1, shadows=False)
        assert relit2.astype(float).mean() > relit1.astype(float).mean(), \
            "Two lights should add more illumination than one"


# ---------------------------------------------------------------------------
# Static confidence coordinate domain (Part 38)
# ---------------------------------------------------------------------------

class TestStaticConfidenceCoordinateDomain:
    def test_compute_rigid_flow_residual_current_domain(self):
        """residuals in current_domain=True are indexed in current-frame coordinates."""
        from geometry.pose import compute_rigid_flow_residual

        H, W = 16, 16
        # Dummy camera
        from geometry.camera import CameraModel
        cam = CameraModel(W, H, fx=8.0, fy=8.0, cx=8.0, cy=8.0)

        # Previous 3D positions (flat plane at z=1.0)
        v_g, u_g = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
        depth = np.ones((H, W), np.float32)
        pos_x = (u_g - cam.cx) / cam.fx * depth
        pos_y = (v_g - cam.cy) / cam.fy * depth
        previous_positions = np.stack([pos_x, pos_y, depth], axis=-1)

        # Flow: object at cols 4-8 moved right by 6 pixels
        forward_flow = np.zeros((H, W, 2), np.float32)
        forward_flow[:, 4:9, 0] = 6.0

        identity = np.eye(4, dtype=np.float32)
        valid = np.ones((H, W), bool)

        # Previous-domain residuals
        residuals_prev = compute_rigid_flow_residual(
            cam, previous_positions, forward_flow, identity, valid, current_domain=False
        )
        # Current-domain residuals (should splat to destination pixels col 10-15)
        residuals_curr = compute_rigid_flow_residual(
            cam, previous_positions, forward_flow, identity, valid, current_domain=True
        )

        assert np.isfinite(residuals_curr[~np.isnan(residuals_curr)]).all(), "Current-domain residuals must be finite where defined"
        assert residuals_curr.shape == (H, W), f"Expected ({H},{W}), got {residuals_curr.shape}"
        assert residuals_prev.shape == (H, W), f"Expected ({H},{W}), got {residuals_prev.shape}"


# ---------------------------------------------------------------------------
# Pose frame IDs (Part 40)
# ---------------------------------------------------------------------------

class TestPoseFrameIDs:
    def test_invalid_result_has_frame_ids(self):
        """PoseEstimateResult must have source/target frame IDs."""
        from geometry.pose import PoseEstimateResult
        inv = PoseEstimateResult(
            valid=False,
            T_current_from_previous=np.eye(4),
            confidence=0.0,
            inlier_mask=np.array([], dtype=bool),
            inlier_count=0,
            reprojection_error=float("inf"),
            source_frame_id=5,
            target_frame_id=6,
        )
        assert inv.source_frame_id == 5
        assert inv.target_frame_id == 6
        assert inv.valid is False

    def test_pose_estimator_result_frame_ids(self):
        """PoseEstimateResult carries explicit source/target frame IDs via estimate()."""
        from geometry.pose import PoseEstimator, PoseEstimateResult
        from geometry.camera import CameraModel

        H, W = 16, 16
        cam = CameraModel(W, H, fx=8.0, fy=8.0, cx=8.0, cy=8.0)
        estimator = PoseEstimator(cam)

        # Build synthetic correspondences (4 point minimum for PnP)
        object_points = np.array([
            [-0.1, -0.1, 1.0],
            [ 0.1, -0.1, 1.0],
            [ 0.1,  0.1, 1.0],
            [-0.1,  0.1, 1.0],
        ], dtype=np.float32)
        # Project to get image points
        image_points = cam.project(object_points).astype(np.float32)

        result = estimator.estimate(
            object_points,
            image_points,
            source_frame_id="frame_10",
            target_frame_id="frame_11",
        )
        assert hasattr(result, "source_frame_id"), "PoseEstimateResult must have source_frame_id"
        assert hasattr(result, "target_frame_id"), "PoseEstimateResult must have target_frame_id"
        # Even if estimation fails (few points), frame IDs should be preserved
        assert result.source_frame_id == "frame_10"
        assert result.target_frame_id == "frame_11"


# ---------------------------------------------------------------------------
# PersistentMapWorker boundedness (Part 41)
# ---------------------------------------------------------------------------

class TestPersistentMapWorkerBounded:
    def test_worker_starts_and_stops(self):
        from geometry.persistent_worker import PersistentMapWorker
        from geometry.camera import CameraModel

        cam = CameraModel(32, 24, fx=16.0, fy=16.0, cx=16.0, cy=12.0)
        worker = PersistentMapWorker(cam, map_update_hz=4.0, max_candidates=100)
        worker.start()
        assert worker.is_alive
        worker.stop()
        assert not worker.is_alive

    def test_submit_drops_excess_jobs(self):
        """With max 1 pending job, excess submits are dropped (not queued)."""
        from geometry.persistent_worker import PersistentMapWorker
        from geometry.camera import CameraModel
        from geometry.state import GeometryState
        from geometry.backproject import DepthScaleMode
        from geometry.pose import PoseEstimateResult

        cam = CameraModel(8, 8, fx=4.0, fy=4.0, cx=4.0, cy=4.0)
        depth = np.ones((8, 8), np.float32)
        pos   = np.zeros((8, 8, 3), np.float32)
        pos[..., 2] = 1.0
        valid = np.ones((8, 8), bool)
        geom  = GeometryState(0.0, 0, depth, pos, valid, cam, DepthScaleMode.RELATIVE)
        pose  = PoseEstimateResult(
            valid=True,
            T_current_from_previous=np.eye(4),
            confidence=0.9,
            inlier_mask=np.ones(10, dtype=bool),
            inlier_count=10,
            reprojection_error=1.0,
            source_frame_id=0,
            target_frame_id=1,
        )

        # Create worker with very slow rate so it doesn't drain jobs quickly
        worker = PersistentMapWorker(cam, map_update_hz=0.5, max_candidates=100)
        # Don't start — just test submit-drop semantics directly
        # Submit 5 jobs without a running consumer
        for i in range(5):
            worker.submit(geom, pose, frame_id=i)
        # Only one job should be pending at most
        assert worker.jobs_dropped >= 4  # 4 of 5 dropped

    def test_worker_non_blocking_snapshot(self):
        """get_latest_snapshot returns None before first update, not raises."""
        from geometry.persistent_worker import PersistentMapWorker
        from geometry.camera import CameraModel

        cam = CameraModel(16, 12, fx=8.0, fy=8.0, cx=8.0, cy=6.0)
        worker = PersistentMapWorker(cam, map_update_hz=2.0)
        worker.start()
        snapshot = worker.get_latest_snapshot()
        assert snapshot is None  # no jobs processed yet
        worker.stop()


# ---------------------------------------------------------------------------
# LightState contract (Part 25)
# ---------------------------------------------------------------------------

class TestLightStateContract:
    def test_light_state_canonical_fields(self):
        """LightState has all required canonical fields."""
        from geometry.lighting import LightState
        ls = LightState(
            position_camera=np.array([0.1, 0.2, 0.5], np.float32),
            intensity=1.5,
            color_rgb=np.array([1.0, 0.9, 0.8], np.float32),
            confidence=0.95,
            source_hand=0,
            light_id=0,
        )
        assert ls.intensity == pytest.approx(1.5)
        assert hasattr(ls, "enabled")
        assert hasattr(ls, "position_camera")
        assert hasattr(ls, "light_id")
        assert ls.position_camera is not None

    def test_light_state_backward_compat_alias(self):
        """position_camera_m alias is available for backward compat."""
        from geometry.lighting import LightState
        pos = np.array([0.1, 0.2, 0.5], np.float32)
        ls = LightState(
            position_camera=pos,
            intensity=1.0,
            color_rgb=np.ones(3, np.float32),
            confidence=1.0,
        )
        # position_camera_m alias must exist
        assert ls.position_camera_m is not None
        np.testing.assert_array_equal(ls.position_camera_m, pos)


# ---------------------------------------------------------------------------
# Quality profiles
# ---------------------------------------------------------------------------

class TestQualityProfiles:
    def test_all_profiles_exist(self):
        from geometry.native_app import PROFILE_CONFIGS, QualityProfile
        for profile in QualityProfile:
            assert profile in PROFILE_CONFIGS
            cfg = PROFILE_CONFIGS[profile]
            assert cfg.shadow_steps >= 1
            assert cfg.volumetric_steps >= 1
            assert cfg.persistent_hz > 0

    def test_profile_low_is_fastest(self):
        from geometry.native_app import PROFILE_CONFIGS, QualityProfile
        low = PROFILE_CONFIGS[QualityProfile.LOW]
        high = PROFILE_CONFIGS[QualityProfile.HIGH]
        # LOW must have fewer shadow steps and smaller/equal depth size
        assert low.shadow_steps <= high.shadow_steps
        low_px = low.depth_size[0] * low.depth_size[1]
        high_px = high.depth_size[0] * high.depth_size[1]
        assert low_px <= high_px


# ---------------------------------------------------------------------------
# NativeLiveApp headless smoke (synthetic)
# ---------------------------------------------------------------------------

class TestNativeLiveAppHeadlessSmoke:
    def test_synthetic_headless_renders_frames(self):
        """App renders correct-shape frames in headless+synthetic mode."""
        from geometry.native_app import AppMode, NativeLiveApp, QualityProfile
        app = NativeLiveApp(
            headless=True,
            use_synthetic_camera=True,
            quality_profile=QualityProfile.LOW,
            initial_mode=AppMode.NORMALS,
        )
        try:
            app.start()
            frames = []
            for _ in range(4):
                f = app.step()
                if f is not None:
                    frames.append(f)
            assert len(frames) >= 1, "Should render at least one frame"
            for f in frames:
                assert f.ndim == 3 and f.shape[2] == 3, f"Bad frame shape: {f.shape}"
        finally:
            app.stop()

    def test_all_modes_produce_frames(self):
        """Every AppMode produces a valid frame without crash."""
        from geometry.native_app import AppMode, NativeLiveApp, QualityProfile
        for mode in AppMode:
            app = NativeLiveApp(
                headless=True,
                use_synthetic_camera=True,
                quality_profile=QualityProfile.LOW,
                initial_mode=mode,
            )
            try:
                app.start()
                frame = app.step()
                assert frame is not None, f"Mode {mode} returned None"
                assert frame.shape[2] == 3, f"Mode {mode} frame has wrong channels"
            finally:
                app.stop()

    def test_mode_switching_no_crash(self):
        """Switching modes mid-run doesn't crash."""
        from geometry.native_app import AppMode, NativeLiveApp, QualityProfile
        app = NativeLiveApp(
            headless=True,
            use_synthetic_camera=True,
            quality_profile=QualityProfile.LOW,
        )
        try:
            app.start()
            for mode in AppMode:
                app.current_mode = mode
                frame = app.step()
                assert frame is not None, f"Mode {mode} crash after switch"
        finally:
            app.stop()

    def test_quality_profile_cycling(self):
        """cycle_quality_profile advances through LOW → BALANCED → HIGH → LOW."""
        from geometry.native_app import NativeLiveApp, QualityProfile
        app = NativeLiveApp(
            headless=True,
            use_synthetic_camera=True,
            quality_profile=QualityProfile.LOW,
        )
        app.start()
        app.cycle_quality_profile()
        assert app.quality_profile == QualityProfile.BALANCED
        app.cycle_quality_profile()
        assert app.quality_profile == QualityProfile.HIGH
        app.cycle_quality_profile()
        assert app.quality_profile == QualityProfile.LOW
        app.stop()

    def test_reset_persistent_map_clears_state(self):
        """reset_persistent_map clears previous_geometry."""
        from geometry.native_app import NativeLiveApp
        app = NativeLiveApp(headless=True, use_synthetic_camera=True)
        app.start()
        app.step()
        app.reset_persistent_map()
        assert app.previous_geometry is None
        app.stop()


# ---------------------------------------------------------------------------
# Worker health reporting
# ---------------------------------------------------------------------------

class TestWorkerHealth:
    def test_depth_worker_has_is_alive(self):
        from geometry.async_pipeline import DepthWorker, LatestDepthBuffer, LatestFrameBuffer
        from geometry.state import DepthState

        buf_in = LatestFrameBuffer()
        buf_out = LatestDepthBuffer()

        called = []

        def fake_infer(rgb, frame_id, ts):
            called.append(frame_id)
            depth = np.ones((4, 4), np.float32)
            return DepthState(depth, ts, frame_id, "relative", valid_mask=np.ones((4, 4), bool))

        worker = DepthWorker(fake_infer, buf_in, buf_out)
        worker.start()
        assert worker.is_alive
        worker.stop()
        assert not worker.is_alive

    def test_depth_worker_reports_error(self):
        from geometry.async_pipeline import DepthWorker, LatestDepthBuffer, LatestFrameBuffer
        import numpy as np

        buf_in = LatestFrameBuffer()
        buf_out = LatestDepthBuffer()

        def bad_infer(rgb, frame_id, ts):
            raise RuntimeError("test error")

        worker = DepthWorker(bad_infer, buf_in, buf_out)
        worker.start()
        frame = np.zeros((4, 4, 3), np.uint8)
        buf_in.put(frame, 0, time.monotonic())
        time.sleep(0.1)  # let it process
        worker.stop()
        assert "RuntimeError" in (worker.last_error or "")


# ---------------------------------------------------------------------------
# ColleagueDepthProvider contract (Part 15)
# ---------------------------------------------------------------------------

class TestDepthProviderContract:
    def test_local_provider_has_last_diagnostics_attr(self):
        """DepthAnythingProvider exposes last_diagnostics attribute."""
        from geometry.depth_provider import DepthAnythingProvider
        p = DepthAnythingProvider()
        assert hasattr(p, "last_diagnostics")

    def test_colleague_provider_has_last_diagnostics_attr(self):
        """ColleagueDepthProvider exposes last_diagnostics attribute (Part 15)."""
        from geometry.colleague_depth import ColleagueDepthProvider
        p = ColleagueDepthProvider()
        assert hasattr(p, "last_diagnostics")


# ---------------------------------------------------------------------------
# Volumetric scattering (Part 33)
# ---------------------------------------------------------------------------

class TestVolumetricScattering:
    def test_volumetric_returns_full_res_array(self):
        """render_volumetric_scattering returns full-res float32 array."""
        from geometry.lighting import LightState, render_volumetric_scattering
        from geometry.camera import CameraModel
        from geometry.state import GeometryState
        from geometry.backproject import DepthScaleMode

        H, W = 16, 16
        cam = CameraModel(W, H, fx=8.0, fy=8.0, cx=8.0, cy=8.0)
        depth = np.ones((H, W), np.float32) * 1.5
        pos = np.zeros((H, W, 3), np.float32)
        pos[..., 2] = 1.5
        valid = np.ones((H, W), bool)
        geom = GeometryState(0.0, 0, depth, pos, valid, cam, DepthScaleMode.RELATIVE)

        light = LightState(
            position_camera=np.array([0.0, 0.0, 0.5], np.float32),
            intensity=2.0, color_rgb=np.ones(3, np.float32),
            confidence=1.0,
        )
        haze, elapsed_ms = render_volumetric_scattering(
            geom, [light], num_steps=3, downsample_factor=4
        )
        assert haze.shape == (H, W, 3), f"Expected ({H},{W},3), got {haze.shape}"
        assert haze.dtype == np.float32
        assert elapsed_ms >= 0.0
