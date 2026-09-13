from __future__ import annotations

from types import SimpleNamespace
import time

import numpy as np

from geometry.backproject import DepthScaleMode
from geometry.camera import CameraModel
from geometry.lighting import LightState, _emission_lobe, _shadow_factor, render_light_orbs, shade_geometry
from geometry.palm_light import PALM_LIGHT_OFFSET_M, PalmLightController
from geometry.state import GeometryState


def _geometry(width: int = 160, height: int = 120, depth_value: float = 1.2) -> GeometryState:
    camera = CameraModel(width, height, 120.0, 120.0, width / 2.0, height / 2.0)
    depth = np.full((height, width), depth_value, dtype=np.float32)
    valid = np.ones((height, width), dtype=bool)
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    points = camera.unproject(uu, vv, depth).astype(np.float32)
    normals = np.zeros((height, width, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    return GeometryState(1.0, 1, depth, points, valid, camera, DepthScaleMode.RELATIVE, normals=normals)


def _hand(hand_id: int = 0, handedness: str = "Right", *, back: bool = False) -> SimpleNamespace:
    cx, cy = 80.0 + hand_id * 35.0, 60.0
    right = handedness == "Right"
    points = np.tile(np.array([[cx, cy]], dtype=np.float32), (21, 1))
    points[0] = (cx, cy + 30)
    index_side = 1.0 if right else -1.0
    if back:
        index_side *= -1.0
    points[5] = (cx + 10 * index_side, cy)
    points[9] = (cx, cy - 15)
    points[13] = (cx, cy - 10)
    points[17] = (cx - 10 * index_side, cy)
    for base, pip, dip, tip, offset in (
        (5, 6, 7, 8, -8), (9, 10, 11, 12, -3),
        (13, 14, 15, 16, 3), (17, 18, 19, 20, 8),
    ):
        x = points[base, 0] + offset
        points[pip] = (x, cy - 8)
        points[dip] = (x, cy - 16)
        points[tip] = (x, cy - 25)
    return SimpleNamespace(
        hand_id=hand_id,
        handedness=handedness,
        landmarks_uv=points,
        palm_uv=(cx, cy),
        palm_width_px=20.0,
        confidence=0.95,
        timestamp=1.0,
    )


def test_front_palm_pose_uses_3d_landmarks_and_offsets_light_outward():
    geometry = _geometry()
    controller = PalmLightController()
    light = controller.update(_hand(0, "Right"), geometry, mirrored_input=True, timestamp=1.0)

    assert light.palm_normal_camera is not None
    assert light.palm_normal_camera[2] < -0.98
    assert light.orb_visibility > 0.8
    assert light.effective_intensity > 0.5
    np.testing.assert_allclose(
        np.linalg.norm(light.position_camera - light.palm_center_camera),
        PALM_LIGHT_OFFSET_M,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        (light.position_camera - light.palm_center_camera) / PALM_LIGHT_OFFSET_M,
        light.palm_normal_camera,
        atol=1e-5,
    )


def test_left_and_right_hands_both_point_the_palm_normal_toward_camera():
    geometry = _geometry()
    for hand_id, handedness in ((0, "Right"), (1, "Left")):
        light = PalmLightController().update(
            _hand(hand_id, handedness), geometry, mirrored_input=True, timestamp=1.0
        )
        assert light.palm_normal_camera is not None
        assert light.palm_normal_camera[2] < -0.98
        assert light.orb_visibility > 0.8


def test_non_mirrored_mediapipe_label_is_corrected_for_unmirrored_camera():
    geometry = _geometry()
    # MP assumes selfie mirroring, so an unmirrored viewer-right hand is labeled Left.
    hand = _hand(0, "Right")
    hand.handedness = "Left"
    light = PalmLightController().update(
        hand, geometry, mirrored_input=False, timestamp=1.0
    )
    assert light.palm_normal_camera is not None
    assert light.palm_normal_camera[2] < -0.98


def test_back_of_hand_hides_orb_but_keeps_physical_emitter_active():
    geometry = _geometry()
    hidden = PalmLightController().update(
        _hand(0, "Right", back=True), geometry, mirrored_input=True, timestamp=1.0
    )
    assert hidden.palm_normal_camera is not None
    assert hidden.palm_normal_camera[2] > 0.98
    assert hidden.position_camera[2] > hidden.palm_center_camera[2]
    assert hidden.orb_visibility == 0.0
    assert hidden.enabled
    assert hidden.effective_intensity > 0.5

    controller = PalmLightController()
    hand = _hand(0, "Right")
    controller.update(hand, geometry, mirrored_input=True, timestamp=1.0)
    result = None
    for frame in range(1, 12):
        hand = _hand(0, "Right", back=True)
        result = controller.update(hand, geometry, mirrored_input=True, timestamp=1.0 + frame / 30.0)
    assert result is not None and result.orb_visibility < 0.03
    assert result.enabled and result.effective_intensity > 0.5


def test_sideways_palm_transitions_to_partial_orb_visibility_without_binary_flicker():
    geometry = _geometry()
    controller = PalmLightController()
    hand = _hand()
    controller.update(hand, geometry, mirrored_input=True, timestamp=1.0)
    # Nearly edge-on 3D palm: wrist-to-middle runs mostly sideways and in depth.
    points = hand.landmarks_uv.copy()
    points[0] = (65.0, 60.0)
    points[5] = (90.0, 60.0)
    points[9] = (95.0, 59.0)
    points[17] = (70.0, 60.0)
    hand.landmarks_uv = points
    hand.palm_uv = (80.0, 60.0)
    geometry.depth[60, 65] = 0.95
    geometry.depth[59, 95] = 1.15
    activations = []
    for frame in range(1, 61):
        activations.append(controller.update(
            hand, geometry, mirrored_input=True, timestamp=1.0 + frame / 30.0
        ).orb_visibility)
    assert any(0.02 < activation < 0.95 for activation in activations)
    assert max(activations[-5:]) - min(activations[-5:]) < 0.03


def test_invalid_depth_falls_back_finitely_and_fades_light():
    geometry = _geometry()
    geometry.valid_mask[:] = False
    fallback = np.array([0.1, -0.05, 1.0], dtype=np.float32)
    light = PalmLightController().update(
        _hand(), geometry, fallback_position=fallback, mirrored_input=True, timestamp=1.0
    )
    assert np.isfinite(light.position_camera).all()
    np.testing.assert_allclose(light.position_camera, fallback)
    assert light.orientation_confidence == 0.0
    assert light.orb_visibility == 0.0
    assert light.enabled and light.effective_intensity > 0.5


def test_two_hands_activate_independently():
    geometry = _geometry()
    controller = PalmLightController()
    active = controller.update(_hand(0, "Right"), geometry, mirrored_input=True, timestamp=1.0)
    inactive = controller.update(_hand(1, "Left", back=True), geometry, mirrored_input=True, timestamp=1.0)
    assert active.orb_visibility > 0.8
    assert inactive.orb_visibility == 0.0
    assert inactive.enabled and inactive.effective_intensity > 0.5


def test_occluded_palm_orb_is_fully_hidden_but_virtual_orb_contract_is_preserved():
    geometry = _geometry(width=80, height=60)
    light = LightState(
        position_camera=np.array([0.0, 0.0, 1.24], dtype=np.float32),
        intensity=1.0,
        orb_visibility=1.0,
        is_palm_attached=True,
        palm_center_camera=np.array([0.0, 0.0, 1.2], dtype=np.float32),
    )
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    result = render_light_orbs(image, geometry.camera, [light], depth=geometry.depth, valid=geometry.valid_mask)
    np.testing.assert_array_equal(result, image)


def test_disabled_emitter_skips_shading_and_returns_unchanged_image(monkeypatch):
    import geometry.lighting as lighting

    geometry = _geometry(width=32, height=24)
    image = np.full((24, 32, 3), 80, dtype=np.uint8)
    light = LightState(position_camera=np.array([0.0, 0.0, 0.8]), is_palm_attached=True, enabled=False, orb_visibility=0.0)
    monkeypatch.setattr(lighting, "_shadow_factor", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("shadow pass should be skipped")))
    result, stats = shade_geometry(image, geometry, [light], shadows=True, volumetrics=True)
    np.testing.assert_array_equal(result, image)
    assert stats["lights"] == 0.0
    assert stats["volumetrics_ms"] == 0.0


def test_hidden_source_still_emits_when_camera_depth_hides_its_orb():
    geometry = _geometry(width=80, height=60, depth_value=1.2)
    geometry.depth[28:33, 38:43] = 0.5
    light = LightState(
        position_camera=np.array([0.0, 0.0, 0.8], dtype=np.float32),
        intensity=1.2,
        confidence=1.0,
        orb_visibility=1.0,
        is_palm_attached=True,
        palm_center_camera=np.array([0.0, 0.0, 0.78], dtype=np.float32),
    )
    image = np.full((60, 80, 3), 36, dtype=np.uint8)
    orb = render_light_orbs(image, geometry.camera, [light], depth=geometry.depth, valid=geometry.valid_mask)
    np.testing.assert_array_equal(orb, image)
    relit, stats = shade_geometry(image, geometry, [light], ambient=0.05, shadows=True, volumetrics=True)
    assert stats["lights"] == 1.0
    assert stats["raytrace_ms"] > 0.0
    assert stats["volumetrics_ms"] > 0.0
    assert np.any(relit != image)


def test_orb_visibility_changes_neither_direct_light_nor_shadow_and_volume_passes():
    geometry = _geometry(width=32, height=24)
    image = np.full((24, 32, 3), 42, dtype=np.uint8)
    hidden_orb = LightState(
        position_camera=np.array([0.0, 0.0, 0.8], dtype=np.float32),
        intensity=1.0,
        confidence=1.0,
        orb_visibility=0.0,
        is_palm_attached=True,
        range_m=1.0,
    )
    visible_orb = LightState(
        position_camera=hidden_orb.position_camera.copy(),
        intensity=hidden_orb.intensity,
        confidence=hidden_orb.confidence,
        orb_visibility=1.0,
        is_palm_attached=True,
        range_m=hidden_orb.range_m,
    )
    hidden_result, hidden_stats = shade_geometry(image, geometry, [hidden_orb], shadows=True, volumetrics=True)
    visible_result, visible_stats = shade_geometry(image, geometry, [visible_orb], shadows=True, volumetrics=True)
    np.testing.assert_array_equal(hidden_result, visible_result)
    assert hidden_stats["lights"] == visible_stats["lights"] == 1.0
    assert hidden_stats["raytrace_ms"] > 0.0
    assert hidden_stats["volumetrics_ms"] > 0.0


def test_rotating_palm_steers_light_without_turning_emitter_off():
    light = LightState(
        position_camera=np.array([0.0, 0.0, 0.8], dtype=np.float32),
        intensity=1.0,
        confidence=1.0,
        enabled=True,
        orb_visibility=0.0,
        is_palm_attached=True,
        palm_normal_camera=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    targets = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    right_facing = _emission_lobe(light, targets)
    light.palm_normal_camera = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    left_facing = _emission_lobe(light, targets)

    assert light.enabled and light.effective_intensity == 1.0
    assert right_facing[0] > 0.95 and right_facing[1] >= 0.20
    assert left_facing[1] > 0.95 and left_facing[0] >= 0.20


def test_owner_palm_blocks_its_hidden_source_while_clear_receiver_is_lit():
    width, height = 64, 48
    camera = CameraModel(width, height, 60.0, 60.0, 32.0, 24.0)
    depth = np.full((height, width), 2.0, dtype=np.float32)
    receiver_uv = (12, 24)
    depth[receiver_uv[1], receiver_uv[0]] = 0.5
    depth[23:26, 31:34] = 0.82
    valid = np.ones((height, width), dtype=bool)
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    points = camera.unproject(uu, vv, depth).astype(np.float32)
    normals = np.zeros((height, width, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    geometry = GeometryState(1.0, 1, depth, points, valid, camera, DepthScaleMode.RELATIVE, normals=normals)
    light = LightState(
        position_camera=np.array([0.0, 0.0, 0.85], dtype=np.float32),
        intensity=1.0,
        confidence=1.0,
        orb_visibility=0.0,
        is_palm_attached=True,
        palm_center_camera=np.array([0.0, 0.0, 0.82], dtype=np.float32),
        self_intersection_epsilon_m=0.002,
        source_radius_m=0.001,
        range_m=1.0,
    )
    blocked = _shadow_factor(geometry, light, steps=8)
    assert blocked[receiver_uv[1], receiver_uv[0]] < 0.5
    geometry.depth[23:26, 31:34] = 2.0
    clear = _shadow_factor(geometry, light, steps=8)
    assert clear[receiver_uv[1], receiver_uv[0]] > 0.9


def test_palm_pose_update_is_negligible_for_two_hands():
    geometry = _geometry()
    controller = PalmLightController()
    hands = (_hand(0, "Right"), _hand(1, "Left"))
    start = time.perf_counter()
    for frame in range(80):
        for hand in hands:
            controller.update(hand, geometry, mirrored_input=True, timestamp=1.0 + frame / 30.0)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5


def test_palm_center_filter_reduces_one_pixel_landmark_jitter():
    geometry = _geometry()
    controller = PalmLightController()
    xs = []
    for frame in range(60):
        hand = _hand()
        jitter = -1.0 if frame % 2 else 1.0
        hand.landmarks_uv[:, 0] += jitter
        light = controller.update(hand, geometry, timestamp=1.0 + frame / 30.0)
        xs.append(float(light.palm_center_camera[0]))

    assert np.var(xs[10:]) < 0.35 * (1.2 / 120.0) ** 2


def test_palm_center_filter_tracks_fast_deliberate_motion():
    geometry = _geometry()
    controller = PalmLightController()
    light = None
    for frame in range(12):
        hand = _hand()
        hand.landmarks_uv[:, 0] += frame * 5.0
        light = controller.update(hand, geometry, timestamp=1.0 + frame / 30.0)
    expected_x = ((80.0 + 55.0) - geometry.camera.cx) * 1.2 / geometry.camera.fx

    assert light is not None
    assert abs(float(light.palm_center_camera[0]) - expected_x) < 0.12
