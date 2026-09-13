import numpy as np
import p123.views.relight as relight_view

from geometry.backproject import DepthScaleMode
from geometry.camera import CameraModel
from geometry.hand_control import GestureState, TrackedHand
from geometry.p123_contract import HandXYZ
from geometry.state import GeometryState
from geometry.p123_live_runtime import P123Metrics, P123Snapshot
from p123.views.relight import RelightRenderer


def _snapshot(hand_count: int = 1) -> P123Snapshot:
    width, height = 160, 120
    camera = CameraModel(width, height, 120.0, 120.0, 80.0, 60.0)
    depth = np.full((height, width), 1.2, dtype=np.float32)
    valid = np.ones((height, width), dtype=bool)
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    points = camera.unproject(uu, vv, depth).astype(np.float32)
    normals = np.zeros((height, width, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    geometry = GeometryState(
        timestamp=1.0,
        source_frame_id=1,
        depth=depth,
        positions_3d=points,
        valid_mask=valid,
        camera=camera,
        scale_mode=DepthScaleMode.RELATIVE,
        normals=normals,
        confidence=np.ones((height, width), dtype=np.float32),
    )
    hands = tuple(
        TrackedHand(
            hand_id=index,
            landmarks_uv=None,
            palm_uv=(58.0 + index * 44.0, 60.0),
            confidence=0.95,
            stale=False,
            depth_z=0.8,
        )
        for index in range(hand_count)
    )
    hand_state = GestureState(timestamp=1.0, source_frame_id=1, hands=hands, backend="test", tracker_ms=0.1)
    metrics = P123Metrics(30.0, 20.0, 30.0, 30.0, 30.0, 30.0, 20.0, 30.0, 20.0, 20.0, 20.0, 1, 0, 0)
    frame = np.full((height, width, 3), 90, dtype=np.uint8)
    xyz = tuple(HandXYZ(hand.hand_id, hand.palm_uv, (0.0, 0.0, 1.2), 0.95, 1.0, 1, 10.0) for hand in hands)
    return P123Snapshot(frame, 1, 1.0, None, geometry, hand_state, xyz, None, metrics)


def test_p123_relight_projects_light_and_returns_cached_material_view():
    snapshot = _snapshot()
    renderer = RelightRenderer(max_width=80, max_height=60)
    image, title, waiting = renderer.render(snapshot)
    again, _, _ = renderer.render(snapshot)

    assert title == "MODE 7 - HAND-HELD RELIGHT"
    assert waiting is None
    assert image.shape == snapshot.rgb_frame.shape
    assert renderer.last_light_count == 1
    assert renderer.last_render_ms > 0
    assert again is image
    assert image[60, 58].mean() > snapshot.rgb_frame[60, 58].mean()


def test_p123_relight_preserves_full_resolution_camera_detail():
    snapshot = _snapshot()
    checker = (np.indices(snapshot.rgb_frame.shape[:2]).sum(axis=0) % 2) * 120 + 50
    snapshot.rgb_frame[:] = checker[..., None]

    image, _, _ = RelightRenderer(max_width=80, max_height=60).render(snapshot)

    # This distant patch should retain the camera's fine checker texture rather
    # than inherit the blur from enlarging the low-resolution lighting render.
    assert image[94:112, 136:154].var() > 2500.0


def test_p123_relight_keeps_two_hand_sources_separate():
    renderer = RelightRenderer(max_width=80, max_height=60)
    image, _, waiting = renderer.render(_snapshot(hand_count=2))
    assert waiting is None
    assert renderer.last_light_count == 2
    assert image[60, 58].max() > 100
    assert image[60, 102].max() > 100


def test_p123_relight_enables_shadow_rays_and_volumetrics(monkeypatch):
    captured = {}
    original = relight_view.shade_geometry

    def capture_settings(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(relight_view, "shade_geometry", capture_settings)
    renderer = RelightRenderer(max_width=80, max_height=60)
    renderer.render(_snapshot(hand_count=2))

    assert captured["shadows"] is True
    assert captured["volumetrics"] is True
    assert renderer.last_lighting_stats["lights"] == 2
    assert renderer.last_lighting_stats["volumetrics_ms"] > 0


def test_p123_relight_handles_geometry_pending():
    snapshot = _snapshot()
    snapshot = P123Snapshot(
        snapshot.rgb_frame,
        snapshot.rgb_capture_id,
        snapshot.rgb_timestamp,
        snapshot.depth_state,
        None,
        snapshot.hand_state,
        snapshot.xyz,
        None,
        snapshot.metrics,
    )
    image, _, waiting = RelightRenderer().render(snapshot)
    assert image.shape == snapshot.rgb_frame.shape
    assert waiting == "waiting for surface geometry"
