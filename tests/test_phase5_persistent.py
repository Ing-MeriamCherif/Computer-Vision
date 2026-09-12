import numpy as np

from geometry import (
    CameraModel, CameraPoseState, DepthState, GeometryState, PersistentGeometryMapper,
    PoseEstimator, SurfelMap, TemporalGeometryEngine, compose_world_pose,
)


def _camera() -> CameraModel:
    return CameraModel(320, 240, 280.0, 275.0, 157.5, 118.5)


def _transform() -> np.ndarray:
    angle = np.deg2rad(4.0)
    c, s = np.cos(angle), np.sin(angle)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = ((c, 0, s), (0, 1, 0), (-s, 0, c))
    transform[:3, 3] = (0.08, -0.03, 0.04)
    return transform


def test_pose_estimator_recovers_combined_motion() -> None:
    camera = _camera()
    rng = np.random.default_rng(12)
    points = np.column_stack((rng.uniform(-1.2, 1.2, 240), rng.uniform(-0.8, 0.8, 240), rng.uniform(3.0, 6.0, 240))).astype(np.float32)
    transform = _transform()
    current = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    pixels = camera.project(current).astype(np.float32)
    result = PoseEstimator(camera, max_samples=500, min_inliers=20).estimate(points, pixels)
    assert result.valid, result.reason
    assert result.inlier_count >= 200
    np.testing.assert_allclose(result.T_current_from_previous, transform, atol=2e-3)
    assert result.reprojection_error < 1e-3


def test_pose_outliers_are_rejected_and_degenerate_is_safe() -> None:
    camera = _camera()
    rng = np.random.default_rng(8)
    points = np.column_stack((rng.uniform(-1, 1, 180), rng.uniform(-.7, .7, 180), rng.uniform(2, 5, 180))).astype(np.float32)
    pixels = camera.project(points).astype(np.float32)
    pixels[:54] = rng.uniform(0, 320, (54, 2))
    result = PoseEstimator(camera, min_inliers=20).estimate(points, pixels)
    assert result.valid and result.inlier_count >= 100
    narrow = points[:10].copy()
    narrow[:, 1] = 0.0
    narrow_pixels = camera.project(narrow).astype(np.float32)
    assert not PoseEstimator(camera, min_correspondences=6).estimate(narrow, narrow_pixels).valid


def test_world_pose_composition_convention() -> None:
    previous_world = np.eye(4, dtype=np.float32)
    relative = _transform()
    composed = compose_world_pose(previous_world, relative)
    np.testing.assert_allclose(composed, np.linalg.inv(relative), atol=1e-6)


def test_bounded_surfel_fusion_and_normal_layer_separation() -> None:
    surfels = SurfelMap(voxel_size=.1, max_surfels=20, max_age_frames=3)
    points = np.zeros((8, 8, 3), np.float32)
    points[..., 2] = 2.0
    normals = np.zeros_like(points)
    normals[..., 2] = -1.0
    confidence = np.ones((8, 8), np.float32)
    first = surfels.insert(points, normals, confidence, 0, stride=1)
    second = surfels.insert(points + .01, normals, confidence, 1, stride=1)
    assert surfels.count <= 20 and first["new_surfels"] > 0 and second["fused_surfels"] > 0
    opposite = normals.copy(); opposite[..., 2] = 1.0
    surfels.insert(points, opposite, confidence, 2, stride=1)
    assert surfels.count > 0


def test_mapper_reprojects_nearest_surface_and_fails_closed() -> None:
    camera = CameraModel(32, 24, 30, 30, 15.5, 11.5)
    depth = np.full((24, 32), 2.0, np.float32)
    engine = TemporalGeometryEngine(camera)
    geometry = engine.update(None, camera, 0, 0.0, DepthState(depth, 0.0, 0, "relative"))
    mapper = PersistentGeometryMapper(camera, SurfelMap(voxel_size=.02, max_surfels=5000), mapping_stride=2)
    pose = CameraPoseState(0.0, 0, np.eye(4, dtype=np.float32), True, 1.0, 100, 0.0)
    persistent = mapper.update(geometry, pose)
    assert persistent is not None and persistent.projected_valid.any()
    assert mapper.update(geometry, CameraPoseState(0.0, 99, np.eye(4, dtype=np.float32), True, 1.0)) is None
