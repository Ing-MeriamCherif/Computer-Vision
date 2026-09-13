import numpy as np

from geometry.camera import CameraModel
from geometry.lighting import LightState, project_light_orb, render_light_orbs


def _camera() -> CameraModel:
    return CameraModel(width=160, height=120, fx=300.0, fy=300.0, cx=80.0, cy=60.0)


def _light(u: float, v: float, z: float, color=(1.0, 0.7, 0.2), source_hand=0) -> LightState:
    camera = _camera()
    return LightState(
        position_camera=camera.unproject(u, v, z).astype(np.float32),
        intensity=1.5,
        color_rgb=np.asarray(color, dtype=np.float32),
        confidence=1.0,
        source_hand=source_hand,
    )


def test_light_orb_center_is_camera_projection_of_lighting_position():
    camera = _camera()
    light = _light(47.0, 39.0, 0.8)
    projected = project_light_orb(camera, light)
    assert projected is not None
    assert projected[:2] == tuple(camera.project(light.position_camera))

    image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    result = render_light_orbs(image, camera, [light], depth_aware=False)
    assert 100 < result[39, 47].max() < 220
    assert result[39, 47].mean() > result[39, 35].mean()


def test_orb_tracks_lateral_and_depth_motion():
    camera = _camera()
    near = project_light_orb(camera, _light(40.0, 55.0, 0.5))
    far = project_light_orb(camera, _light(40.0, 55.0, 1.0))
    moved = project_light_orb(camera, _light(110.0, 55.0, 0.5))
    assert near is not None and far is not None and moved is not None
    np.testing.assert_allclose(near[:2], (40.0, 55.0), atol=1e-5)
    assert near[2] > far[2]
    np.testing.assert_allclose(moved[:2], (110.0, 55.0), atol=1e-5)


def test_two_lights_render_distinct_color_sources():
    camera = _camera()
    cyan = _light(48.0, 60.0, 0.8, color=(0.1, 0.8, 1.0), source_hand=0)
    amber = _light(112.0, 60.0, 0.8, color=(1.0, 0.45, 0.1), source_hand=1)
    image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    result = render_light_orbs(image, camera, [cyan, amber], depth_aware=False)
    cyan_halo = result[60, 54].astype(float)
    amber_halo = result[60, 106].astype(float)
    assert cyan_halo[2] > cyan_halo[0]
    assert amber_halo[0] > amber_halo[2]
    assert 100 < result[60, 48].max() < 220
    assert 100 < result[60, 112].max() < 220


def test_near_surface_dims_halo_without_hiding_emitter_core():
    camera = _camera()
    light = _light(80.0, 60.0, 0.8)
    image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    occluding_depth = np.full((camera.height, camera.width), 0.3, dtype=np.float32)
    visible = render_light_orbs(image, camera, [light], depth_aware=False)
    depth_aware = render_light_orbs(image, camera, [light], depth=occluding_depth)
    assert depth_aware[60, 80].max() > 100
    assert depth_aware[60, 80].max() == visible[60, 80].max()
    assert depth_aware[60, 86].max() < visible[60, 86].max()


def test_invalid_and_disabled_lights_do_not_draw():
    camera = _camera()
    image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    disabled = _light(80.0, 60.0, 0.8)
    disabled.enabled = False
    behind_camera = LightState(position_camera=np.array([0.0, 0.0, -1.0], dtype=np.float32))
    assert project_light_orb(camera, disabled) is None
    assert project_light_orb(camera, behind_camera) is None
    result = render_light_orbs(image, camera, [disabled, behind_camera])
    assert result is image
