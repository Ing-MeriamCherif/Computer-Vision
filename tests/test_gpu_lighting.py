from __future__ import annotations

import numpy as np
import pytest

from geometry.backproject import DepthScaleMode
from geometry.camera import CameraModel
from geometry.gpu_lighting import (
    GPURelightRenderer,
    QUALITY_PROFILES,
    _COMPOSITE_SHADER,
    _SURFACE_SHADER,
    _VOLUME_SHADER,
)
from geometry.lighting import LightState
from geometry.state import GeometryState


def _geometry(width: int = 32, height: int = 24) -> tuple[np.ndarray, GeometryState]:
    camera = CameraModel(width, height, 28.0, 28.0, (width - 1) / 2, (height - 1) / 2)
    depth = np.full((height, width), 1.2, dtype=np.float32)
    valid = np.ones((height, width), dtype=bool)
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    points = camera.unproject(uu, vv, depth).astype(np.float32)
    normals = np.zeros_like(points)
    normals[..., 2] = -1.0
    confidence = np.ones((height, width), dtype=np.float32)
    geometry = GeometryState(
        timestamp=1.0,
        source_frame_id=1,
        depth=depth,
        positions_3d=points,
        valid_mask=valid,
        camera=camera,
        scale_mode=DepthScaleMode.METRIC,
        normals=normals,
        confidence=confidence,
    )
    rows = np.arange(height, dtype=np.uint8)[:, None]
    cols = np.arange(width, dtype=np.uint8)[None, :]
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = 70 + rows
    rgb[..., 1] = 90 + cols
    rgb[..., 2] = 110
    return rgb, geometry


def _light(camera: CameraModel, hand_id: int, color: tuple[float, float, float]) -> LightState:
    return LightState(
        position_camera=camera.unproject(16.0 + hand_id * 2.0, 12.0, 0.55).astype(np.float32),
        intensity=0.7,
        color_rgb=np.asarray(color, dtype=np.float32),
        confidence=1.0,
        source_hand=hand_id,
        light_id=hand_id,
        range_m=0.30,
        source_radius_m=0.018,
    )


def test_shader_path_reconstructs_position_without_positions_3d_upload():
    shader = _SURFACE_SHADER + _VOLUME_SHADER + _COMPOSITE_SHADER
    assert "uDepth" in shader and "uCamera" in shader
    assert "positions_3d" not in shader
    assert "uCamera.x * s.x / s.z" in shader
    assert "uLightShadowBias" in _SURFACE_SHADER
    assert "ownerSelfHit" not in shader
    assert "uLightVisualMeta" in _COMPOSITE_SHADER
    assert "lessThan(uv, vec2(0.0))" in shader
    assert "uShadowSteps" in _SURFACE_SHADER
    assert "uVolShadowSteps" in _VOLUME_SHADER
    assert set(QUALITY_PROFILES) == {"low", "balanced", "high"}


def test_gpu_shader_keeps_specular_additive_and_normalizes_volume_steps():
    assert "diffuseTerm += base * uLightColor[i]" in _SURFACE_SHADER
    assert "specularTerm += uLightColor[i]" in _SURFACE_SHADER
    assert "specularTerm += base *" not in _SURFACE_SHADER
    assert "uLightCount == 1" in _SURFACE_SHADER
    assert "stepLength" in _VOLUME_SHADER
    assert "* scatter * stepLength" in _VOLUME_SHADER


def test_l2_shader_guards_normals_front_faces_and_runtime_parameters():
    assert "uDiffuseStrength" in _SURFACE_SHADER
    assert "uSpecularStrength" in _SURFACE_SHADER
    assert "uShininess" in _SURFACE_SHADER
    assert "uDirectGain" in _SURFACE_SHADER
    assert "uShadowsEnabled" in _SURFACE_SHADER
    assert "if (!(normalLength > 1e-4))" in _SURFACE_SHADER
    assert "specular = diffuse > 0.0" in _SURFACE_SHADER


def test_cpu_reference_zero_normal_and_low_confidence_preserve_camera_rgb():
    from dataclasses import replace
    from geometry.lighting import shade_geometry

    rgb, geometry = _geometry()
    zero_normal = replace(
        geometry,
        normals=np.zeros_like(geometry.normals),
        normal_confidence=np.zeros_like(geometry.depth),
    )
    light = _light(geometry.camera, 0, (1.0, 1.0, 1.0))
    result, _ = shade_geometry(rgb, zero_normal, [light], shadows=False, volumetrics=False)
    np.testing.assert_allclose(result, rgb, atol=1)
    low_conf = replace(geometry, confidence=np.zeros_like(geometry.depth), normal_confidence=np.zeros_like(geometry.depth))
    result, _ = shade_geometry(rgb, low_conf, [light], shadows=False, volumetrics=False)
    np.testing.assert_allclose(result, rgb, atol=1)


def test_cpu_reference_back_facing_light_has_no_specular_response():
    from geometry.lighting import shade_geometry

    rgb, geometry = _geometry()
    front = _light(geometry.camera, 0, (1.0, 1.0, 1.0))
    back = LightState(
        position_camera=np.array([0.0, 0.0, 2.0], dtype=np.float32),
        intensity=1.0,
        color_rgb=np.ones(3, dtype=np.float32),
        confidence=1.0,
        range_m=2.0,
    )
    front_result, _ = shade_geometry(rgb, geometry, [front], shadows=False, volumetrics=False)
    back_result, _ = shade_geometry(rgb, geometry, [back], shadows=False, volumetrics=False)
    assert front_result[12, 16].mean() > rgb[12, 16].mean()
    np.testing.assert_allclose(back_result, rgb, atol=1)


def test_texture_path_avoids_full_gpu_finish():
    import inspect

    source = inspect.getsource(GPURelightRenderer._render_impl)
    assert "gl.glFinish" not in source
    assert "gl.glFlush()" in source


def test_gpu_renderer_compiles_renders_detail_and_reallocates_on_resize():
    rgb, geometry = _geometry()
    try:
        renderer = GPURelightRenderer("low")
    except Exception as exc:  # OpenGL is optional on headless CI machines.
        pytest.skip(f"OpenGL 3.3 context unavailable: {exc}")
    try:
        no_light, no_light_stats = renderer.render(rgb, geometry, [])
        assert no_light.shape == rgb.shape
        assert no_light.dtype == np.uint8
        assert np.isfinite(no_light).all()
        np.testing.assert_allclose(no_light, rgb, atol=1)
        assert no_light_stats["renderer"] == "GPU"
        assert no_light_stats["gl_version"].startswith(("3.", "4."))

        light = _light(geometry.camera, 0, (0.44, 0.72, 0.82))
        one_light, _ = renderer.render(rgb, geometry, [light])
        assert one_light[12, 16].mean() > no_light[12, 16].mean()
        center_gain = one_light[12, 16].astype(np.float32).mean() - no_light[12, 16].mean()
        edge_gain = one_light[0, 0].astype(np.float32).mean() - no_light[0, 0].mean()
        assert center_gain > edge_gain

        second = _light(geometry.camera, 1, (0.88, 0.63, 0.40))
        two_lights, stats = renderer.render(rgb, geometry, [light, second])
        assert two_lights.shape == rgb.shape
        assert stats["lights"] == 2.0
        assert not np.array_equal(one_light, two_lights)

        texture_id, texture_stats = renderer.render_to_texture(rgb, geometry, [light])
        assert texture_id > 0
        assert texture_stats["renderer"] == "GPU"

        renderer.set_quality("balanced")
        assert (renderer._volume_width, renderer._volume_height) == (8, 6)
        assert renderer._history_key is None

        larger_rgb, larger_geometry = _geometry(40, 28)
        renderer.render(larger_rgb, larger_geometry, [light])
        assert (renderer._width, renderer._height) == (40, 28)
        assert renderer._history_key is None or renderer._history_key[0:2] == (40, 28)
    finally:
        renderer.close()
        renderer.close()


def test_quality_profiles_keep_volume_exposure_consistent():
    rgb, geometry = _geometry()
    try:
        renderer = GPURelightRenderer("low")
    except Exception as exc:
        pytest.skip(f"OpenGL 3.3 context unavailable: {exc}")
    try:
        light = _light(geometry.camera, 0, (0.44, 0.72, 0.82))
        means = []
        target_sizes = []
        for quality in ("low", "balanced", "high"):
            renderer.set_quality(quality)
            image, _ = renderer.render(rgb, geometry, [light])
            means.append(float(image.mean()))
            target_sizes.append((renderer._volume_width, renderer._volume_height))
        assert max(means) - min(means) < 2.0
        assert target_sizes == [(6, 4), (8, 6), (11, 8)]
    finally:
        renderer.close()


def test_gpu_hidden_orb_keeps_emitter_active_and_disabled_light_is_filtered():
    rgb, geometry = _geometry()
    try:
        renderer = GPURelightRenderer("low")
    except Exception as exc:
        pytest.skip(f"OpenGL 3.3 context unavailable: {exc}")
    try:
        light = LightState(
            position_camera=geometry.camera.unproject(16.0, 12.0, 0.55).astype(np.float32),
            intensity=0.7,
            confidence=1.0,
            is_palm_attached=True,
            orb_visibility=0.0,
            palm_center_camera=np.array([0.0, 0.0, 0.56], dtype=np.float32),
            palm_normal_camera=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            self_intersection_epsilon_m=0.002,
        )
        hidden_orb, hidden_stats = renderer.render(rgb, geometry, [light])
        assert hidden_stats["lights"] == 1.0
        assert np.isfinite(hidden_orb).all()
        assert hidden_stats["shadow_submit_ms"] >= 0.0
        assert hidden_stats["volumetric_submit_ms"] >= 0.0

        light.orb_visibility = 1.0
        visible_orb, visible_stats = renderer.render(rgb, geometry, [light])
        assert visible_stats["lights"] == 1.0
        assert not np.array_equal(visible_orb[12, 16], hidden_orb[12, 16])

        light.enabled = False
        inactive, inactive_stats = renderer.render(rgb, geometry, [light])
        assert inactive_stats["lights"] == 0.0
        np.testing.assert_allclose(inactive, rgb, atol=1)
    finally:
        renderer.close()


def test_gpu_hidden_palm_emitter_still_shadows_from_its_owner_hand():
    rgb, geometry = _geometry()
    geometry.depth[:] = 1.2
    geometry.depth[12, 6] = 0.5
    geometry.depth[10:15, 14:19] = 0.82
    geometry.normals[:] = (0.0, 0.0, 1.0)
    try:
        renderer = GPURelightRenderer("low")
    except Exception as exc:
        pytest.skip(f"OpenGL 3.3 context unavailable: {exc}")
    light = LightState(
        position_camera=np.array([0.0, 0.0, 0.85], dtype=np.float32),
        intensity=1.0,
        confidence=1.0,
        is_palm_attached=True,
        orb_visibility=1.0,
        palm_center_camera=np.array([0.0, 0.0, 0.82], dtype=np.float32),
        source_radius_m=0.001,
        range_m=1.0,
    )
    try:
        blocked, blocked_stats = renderer.render(rgb, geometry, [light])
        assert blocked_stats["lights"] == 1.0

        geometry.depth[10:15, 14:19] = 1.2
        geometry.source_frame_id = 2
        clear, _ = renderer.render(rgb, geometry, [light])
        assert clear[12, 6].mean() > blocked[12, 6].mean()
    finally:
        renderer.close()


def test_temporal_history_rejects_light_id_and_geometry_jumps():
    rgb, geometry = _geometry()
    try:
        renderer = GPURelightRenderer("balanced")
    except Exception as exc:
        pytest.skip(f"OpenGL 3.3 context unavailable: {exc}")
    try:
        renderer.resize(rgb.shape[1], rgb.shape[0])
        first = _light(geometry.camera, 0, (0.44, 0.72, 0.82))
        assert not renderer._history_compatible(geometry, [first], 10.0)
        assert renderer._history_compatible(geometry, [first], 10.03)
        other = _light(geometry.camera, 1, (0.88, 0.63, 0.40))
        assert not renderer._history_compatible(geometry, [other], 10.06)
        jumped = GeometryState(
            timestamp=geometry.timestamp,
            source_frame_id=20,
            depth=geometry.depth,
            positions_3d=geometry.positions_3d,
            valid_mask=geometry.valid_mask,
            camera=geometry.camera,
            scale_mode=geometry.scale_mode,
            normals=geometry.normals,
            confidence=geometry.confidence,
        )
        assert not renderer._history_compatible(jumped, [other], 10.09)
    finally:
        renderer.close()
