"""Comprehensive unit tests for the P123 Google Material 3 UI."""

from __future__ import annotations

import numpy as np

from geometry.hand_control import GestureState, TrackedHand
from geometry.p123_contract import HandXYZ
from geometry.p123_live_runtime import P123Metrics, P123Snapshot
from p123 import views
from p123.views import common
from tools.p123_live_app import _parse_display_size


def make_test_snapshot(
    has_frame: bool = True,
    has_hands: bool = True,
    xyz_fresh: bool = True,
    depth_state: bool = False,
) -> P123Snapshot:
    """Helper to synthesize structured in-memory snapshots."""
    metrics = P123Metrics(
        capture_hz=30.0,
        depth_hz=14.0,
        geometry_hz=30.0,
        temporal_hz=30.0,
        hand_hz=30.0,
        xyz_hz=30.0,
        depth_age_p50_ms=45.0,
        depth_age_p95_ms=75.0,
        geometry_age_p95_ms=10.0,
        hand_age_p95_ms=15.0,
        xyz_age_p95_ms=18.0 if xyz_fresh else 280.0,
        captured=100,
        overwritten_before_consumption=0,
        depth_errors=0,
        normal_hz=30.0,
        normal_age_p95_ms=12.0,
    )

    rgb = np.zeros((480, 640, 3), dtype=np.uint8) if has_frame else None
    if rgb is not None:
        rgb[:, :] = (30, 35, 45)

    if has_hands:
        landmarks = np.array([[100 + i * 5, 120 + i * 3] for i in range(21)], dtype=np.float32)
        hand = TrackedHand(
            hand_id=0,
            landmarks_uv=landmarks,
            palm_uv=(150.0, 150.0),
            confidence=0.98,
            stale=False,
            depth_z=0.65,
            depth_confidence=0.95,
        )
        hand_state = GestureState(timestamp=10.0, source_frame_id=100, hands=(hand,), backend="test", tracker_ms=2.0)
        xyz_item = HandXYZ(
            hand_id=0,
            palm_uv=(150.0, 150.0),
            xyz_camera=(0.12, -0.05, 0.65),
            confidence=0.98,
            timestamp=10.0,
            source_frame_id=100,
            age_ms=18.0 if xyz_fresh else 280.0,
        )
        xyz_tuple = (xyz_item,)
    else:
        hand_state = None
        xyz_tuple = ()

    return P123Snapshot(
        rgb_frame=rgb,
        rgb_capture_id=100 if has_frame else None,
        rgb_timestamp=10.0 if has_frame else None,
        depth_state=None,
        geometry_state=None,
        hand_state=hand_state,
        xyz=xyz_tuple,
        contract=None,
        metrics=metrics,
        fast_geometry_state=None,
    )


def test_layout_computation_across_resolutions():
    for w, h in [(640, 480), (800, 600), (960, 600), (1024, 640), (1280, 720), (1920, 1080)]:
        layout = common.compute_layout((w, h))
        assert layout["w"] == w
        assert layout["h"] == h
        if layout.get("is_fhd"):
            assert 42 <= layout["header_h"] <= 68
            assert 160 <= layout["sidebar_w"] <= 290
        else:
            assert 35 <= layout["header_h"] <= 55
            assert 140 <= layout["sidebar_w"] <= 220
        assert layout["vw"] > 0
        assert layout["vh"] > 0
        assert len(layout["buttons_rects"]) == 7


def test_sidebar_hit_testing():
    size = (960, 600)
    layout = common.compute_layout(size)
    for mode_id, bx, by, bw, bh, _, _ in layout["buttons_rects"]:
        hit = common.hit_test_navigation(bx + 10, by + 10, size)
        assert hit == mode_id

    # Clicks outside sidebar
    assert common.hit_test_navigation(layout["sidebar_w"] + 50, 100, size) is None
    # Clicks in header
    assert common.hit_test_navigation(50, 10, size) is None


def test_all_seven_views_render_valid_rgb_canvas():
    snap = make_test_snapshot(has_frame=True, has_hands=True)
    for mode in range(1, 8):
        canvas = views.render(snap, mode, (960, 600), display_fps=30.0)
        assert canvas is not None
        assert canvas.shape == (600, 960, 3)
        assert canvas.dtype == np.uint8
        # Ensure canvas is populated (not pure black)
        assert np.any(canvas > 0)


def test_empty_and_waiting_states_render_gracefully():
    # Camera unavailable
    snap_no_cam = make_test_snapshot(has_frame=False, has_hands=False)
    canvas = views.render(snap_no_cam, 1, (960, 600))
    assert canvas.shape == (600, 960, 3)
    assert np.any(canvas > 0)

    # Depth pending
    canvas_depth = views.render(snap_no_cam, 2, (960, 600))
    assert canvas_depth.shape == (600, 960, 3)

    # Normals pending
    canvas_normals = views.render(snap_no_cam, 3, (960, 600))
    assert canvas_normals.shape == (600, 960, 3)

    # Temporal stabilizing
    canvas_temp = views.render(snap_no_cam, 4, (960, 600))
    assert canvas_temp.shape == (600, 960, 3)


def test_hand_and_xyz_views_handle_missing_hands():
    snap_no_hands = make_test_snapshot(has_frame=True, has_hands=False)
    c5 = views.render(snap_no_hands, 5, (960, 600))
    assert c5.shape == (600, 960, 3)

    c6 = views.render(snap_no_hands, 6, (960, 600))
    assert c6.shape == (600, 960, 3)


def test_xyz_freshness_status_distinction():
    snap_fresh = make_test_snapshot(has_frame=True, has_hands=True, xyz_fresh=True)
    c_fresh = views.render(snap_fresh, 6, (960, 600))
    assert c_fresh.shape == (600, 960, 3)

    snap_stale = make_test_snapshot(has_frame=True, has_hands=True, xyz_fresh=False)
    c_stale = views.render(snap_stale, 6, (960, 600))
    assert c_stale.shape == (600, 960, 3)


def test_debug_overlay_hud():
    snap = make_test_snapshot(has_frame=True)
    canvas = views.render(snap, 1, (960, 600), show_debug=True, display_fps=30.0)
    assert canvas.shape == (600, 960, 3)


def test_display_size_parser():
    assert _parse_display_size("native", (640, 480)) == (640, 480)
    assert _parse_display_size("auto", (640, 480)) == (1920, 1080)
    assert _parse_display_size("fhd", (640, 480)) == (1920, 1080)
    assert _parse_display_size("1080p", (640, 480)) == (1920, 1080)
    assert _parse_display_size("fullscreen", (640, 480)) == (1920, 1080)
    assert _parse_display_size("1280x720", (640, 480)) == (1280, 720)
    assert _parse_display_size("800", (640, 480)) == (800, 800)
