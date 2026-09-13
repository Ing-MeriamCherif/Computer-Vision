import numpy as np
from dataclasses import replace
import time
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
    captured_at = time.monotonic()
    camera = CameraModel(width, height, 120.0, 120.0, 80.0, 60.0)
    depth = np.full((height, width), 1.2, dtype=np.float32)
    valid = np.ones((height, width), dtype=bool)
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    points = camera.unproject(uu, vv, depth).astype(np.float32)
    normals = np.zeros((height, width, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    geometry = GeometryState(
        timestamp=captured_at,
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
    hand_state = GestureState(timestamp=captured_at, source_frame_id=1, hands=hands, backend="test", tracker_ms=0.1)
    metrics = P123Metrics(30.0, 20.0, 30.0, 30.0, 30.0, 30.0, 20.0, 30.0, 20.0, 20.0, 20.0, 1, 0, 0)
    frame = np.full((height, width, 3), 90, dtype=np.uint8)
    xyz = tuple(HandXYZ(hand.hand_id, hand.palm_uv, (0.0, 0.0, 1.2), 0.95, captured_at, 1, 10.0) for hand in hands)
    return P123Snapshot(frame, 1, captured_at, None, geometry, hand_state, xyz, None, metrics)


def test_p123_relight_projects_light_and_returns_cached_material_view(monkeypatch):
    snapshot = _snapshot()
    stable_clock = snapshot.xyz[0].timestamp + 0.03
    monkeypatch.setattr(relight_view.time, "monotonic", lambda: stable_clock)
    renderer = RelightRenderer(max_width=80, max_height=60, use_gpu=False)
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

    image, _, _ = RelightRenderer(max_width=80, max_height=60, use_gpu=False).render(snapshot)

    # This distant patch should retain the camera's fine checker texture rather
    # than inherit the blur from enlarging the low-resolution lighting render.
    assert image[94:112, 136:154].var() > 2500.0


def test_p123_relight_keeps_two_hand_sources_separate():
    renderer = RelightRenderer(max_width=80, max_height=60, use_gpu=False)
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
    renderer = RelightRenderer(max_width=80, max_height=60, use_gpu=False)
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


def test_zero_hands_keeps_camera_view_clear_without_a_modal_card():
    snapshot = _snapshot(hand_count=0)
    image, _, waiting = RelightRenderer(use_gpu=False).render(snapshot)

    assert waiting is None
    np.testing.assert_array_equal(image, snapshot.rgb_frame)


def test_handxyz_positions_drive_lights_and_keep_ids_colors_and_range():
    snapshot = _snapshot(hand_count=2)
    xyz = (
        replace(snapshot.xyz[0], xyz_camera=(0.21, -0.08, 0.62), source_age_ms=25.0),
        replace(snapshot.xyz[1], xyz_camera=(-0.18, 0.04, 0.71), source_age_ms=35.0),
    )
    lights, age = relight_view.lights_from_snapshot(replace(snapshot, xyz=xyz))

    assert [light.light_id for light in lights] == [0, 1]
    np.testing.assert_allclose(lights[0].position_camera, (0.21, -0.08, 0.62))
    np.testing.assert_allclose(lights[1].position_camera, (-0.18, 0.04, 0.71))
    assert not np.array_equal(lights[0].color_rgb, lights[1].color_rgb)
    assert all(light.range_m == 0.30 for light in lights)
    assert all(light.source_radius_m == 0.018 for light in lights)
    assert all(light.visual_radius_m == 0.035 for light in lights)
    assert age >= 35.0 and age < 250.0


def test_production_light_position_does_not_sample_palm_depth(monkeypatch):
    import geometry.lighting as lighting

    snapshot = _snapshot()
    expected = np.asarray(snapshot.xyz[0].xyz_camera)

    def unexpected_depth_sample(*_args, **_kwargs):
        raise AssertionError("palm depth sampled")

    monkeypatch.setattr(lighting, "sample_depth", unexpected_depth_sample)
    lights, _ = relight_view.lights_from_snapshot(snapshot)
    np.testing.assert_allclose(lights[0].position_camera, expected)


def test_missing_invalid_or_stale_handxyz_never_fabricates_a_light():
    snapshot = _snapshot()
    missing = replace(snapshot.xyz[0], xyz_camera=None)
    stale = replace(snapshot.xyz[0], source_age_ms=400.0)
    unknown_age = replace(snapshot.xyz[0], timestamp=float("nan"), source_age_ms=float("nan"))
    for xyz in (missing, stale, unknown_age):
        lights, _ = relight_view.lights_from_snapshot(replace(snapshot, xyz=(xyz,)))
        assert lights == []


def test_fresh_handxyz_survives_between_detector_optical_flow_frames():
    snapshot = _snapshot()
    tracked = replace(snapshot.hand_state.hands[0], stale=True, confidence=0.82)
    hand_state = replace(snapshot.hand_state, hands=(tracked,), stale=True)
    xyz = replace(snapshot.xyz[0], source_age_ms=35.0, confidence=0.9)

    lights, age = relight_view.lights_from_snapshot(replace(snapshot, hand_state=hand_state, xyz=(xyz,)))

    assert len(lights) == 1
    assert lights[0].source_hand == tracked.hand_id
    assert lights[0].confidence > 0.7
    assert age >= 35.0 and age < 250.0


def test_frozen_capture_timestamp_expires_even_when_worker_age_is_unchanged():
    snapshot = _snapshot()
    old_xyz = replace(snapshot.xyz[0], timestamp=time.monotonic() - 0.5, source_age_ms=10.0)
    lights, age = relight_view.lights_from_snapshot(replace(snapshot, xyz=(old_xyz,)))
    assert lights == []
    assert age >= 500.0


def test_source_change_clears_cached_frame_latches_and_renderer_history():
    class RendererStub:
        def __init__(self):
            self.reset = False

        def reset_history(self):
            self.reset = True

    renderer = RelightRenderer(use_gpu=False)
    renderer._cache_key = (1,)
    renderer._cache_image = np.ones((2, 2, 3), dtype=np.uint8)
    renderer._gesture_enabled[3] = True
    renderer._gesture_last_seen[3] = time.monotonic()
    renderer._recent_lights[3] = relight_view.LightState(source_hand=3)
    renderer._gpu_renderer = RendererStub()
    renderer.reset_for_source_change()

    assert renderer._cache_key is None
    assert renderer._cache_image is None
    assert renderer._gesture_enabled == {}
    assert renderer._gesture_last_seen == {}
    assert renderer._recent_lights == {}
    assert renderer._gpu_renderer.reset


def test_open_palm_enables_light_and_closed_palm_disables_it():
    snapshot = _snapshot()
    open_points = np.zeros((21, 2), dtype=np.float32)
    open_points[0] = (80, 100)
    for tip, pip, x in ((8, 6, 55), (12, 10, 72), (16, 14, 89), (20, 18, 106)):
        open_points[pip] = (x, 72)
        open_points[tip] = (x, 40)
    closed_points = open_points.copy()
    for tip, pip, _ in ((8, 6, 55), (12, 10, 72), (16, 14, 89), (20, 18, 106)):
        closed_points[tip] = (open_points[pip] + open_points[0]) * 0.5

    renderer = RelightRenderer(use_gpu=False)
    open_hand = replace(snapshot.hand_state.hands[0], landmarks_uv=open_points, stale=False)
    opened = replace(snapshot, hand_state=replace(snapshot.hand_state, hands=(open_hand,)))
    renderer.render(opened)
    assert renderer.last_light_count == 1

    closed_hand = replace(open_hand, landmarks_uv=closed_points)
    closed = replace(opened, rgb_capture_id=2, hand_state=replace(opened.hand_state, hands=(closed_hand,)))
    image, _, _ = renderer.render(closed)
    assert renderer.last_light_count == 0
    assert renderer._recent_lights == {}
    np.testing.assert_array_equal(image, closed.rgb_frame)


def test_brief_two_hand_dropout_keeps_recent_light_then_expires(monkeypatch):
    snapshot = _snapshot(hand_count=2)
    captured_at = snapshot.xyz[0].timestamp
    clock = [captured_at + 0.03]
    monkeypatch.setattr(relight_view.time, "monotonic", lambda: clock[0])

    open_points = np.zeros((21, 2), dtype=np.float32)
    open_points[0] = (80, 100)
    for tip, pip, x in ((8, 6, 55), (12, 10, 72), (16, 14, 89), (20, 18, 106)):
        open_points[pip] = (x, 72)
        open_points[tip] = (x, 40)
    open_hands = tuple(
        replace(hand, landmarks_uv=open_points, stale=False)
        for hand in snapshot.hand_state.hands
    )
    opened = replace(snapshot, hand_state=replace(snapshot.hand_state, hands=open_hands))
    renderer = RelightRenderer(use_gpu=False)
    renderer.render(opened)
    assert renderer.last_light_count == 2

    clock[0] = captured_at + 0.10
    surviving_hand = replace(open_hands[0], timestamp=clock[0])
    surviving_xyz = replace(snapshot.xyz[0], timestamp=clock[0], source_age_ms=0.0, age_ms=0.0)
    one_hand = replace(
        opened,
        rgb_capture_id=2,
        rgb_timestamp=clock[0],
        hand_state=replace(opened.hand_state, timestamp=clock[0], hands=(surviving_hand,)),
        xyz=(surviving_xyz,),
    )
    renderer.render(one_hand)
    assert renderer.last_light_count == 2

    clock[0] = captured_at + 0.22
    surviving_hand = replace(surviving_hand, timestamp=clock[0])
    surviving_xyz = replace(surviving_xyz, timestamp=clock[0], source_age_ms=0.0, age_ms=0.0)
    one_hand = replace(
        one_hand,
        rgb_capture_id=3,
        rgb_timestamp=clock[0],
        hand_state=replace(one_hand.hand_state, timestamp=clock[0], hands=(surviving_hand,)),
        xyz=(surviving_xyz,),
    )
    renderer.render(one_hand)
    assert renderer.last_light_count == 1


def test_fast_geometry_is_preferred_and_slow_geometry_is_fallback(monkeypatch):
    snapshot = _snapshot()
    fast = replace(snapshot.geometry_state, source_frame_id=2)
    renderer = RelightRenderer(max_width=80, max_height=60, use_gpu=False)
    used_sources = []
    low_geometry = renderer._low_geometry

    def capture(geometry):
        used_sources.append(geometry.source_frame_id)
        return low_geometry(geometry)

    monkeypatch.setattr(renderer, "_low_geometry", capture)
    renderer.render(replace(snapshot, fast_geometry_state=fast))
    assert used_sources[-1] == 2
    renderer._cache_key = None
    renderer.render(replace(snapshot, fast_geometry_state=None))
    assert used_sources[-1] == snapshot.geometry_state.source_frame_id


def test_gpu_shader_warmup_does_not_block_the_live_view():
    snapshot = _snapshot()
    renderer = RelightRenderer(max_width=80, max_height=60, use_gpu=True)
    try:
        started = time.perf_counter()
        image, _, _ = renderer.render(snapshot)
        elapsed = time.perf_counter() - started
        assert elapsed < 1.0
        assert image.shape == snapshot.rgb_frame.shape
        if renderer._gpu_warm_thread is not None:
            renderer._gpu_warm_thread.join(timeout=10.0)
        if renderer._gpu_error is None:
            assert renderer._gpu_warmed
            image, _, _ = renderer.render(_snapshot())
            assert image.shape == snapshot.rgb_frame.shape
            assert renderer.last_lighting_stats["renderer"] == "GPU"
    finally:
        renderer.close()
