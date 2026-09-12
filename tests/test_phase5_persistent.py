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


def test_pose_estimator_from_flow() -> None:
    camera = _camera()
    h, w = camera.height, camera.width
    depth = np.full((h, w), 3.0, dtype=np.float32)
    engine = TemporalGeometryEngine(camera)
    geom = engine.update(None, camera, 0, 0.0, DepthState(depth, 0.0, 0, "metric"))

    # Synthetic 2-pixel translation flow (H, W, 2)
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[..., 0] = 2.0
    from geometry import MotionState
    motion = MotionState(
        source_frame_id=0,
        target_frame_id=1,
        timestamp=0.0,
        forward_flow=flow,
        backward_flow=-flow,
        valid_mask=np.ones((h, w), dtype=bool),
    )

    estimator = PoseEstimator(camera, max_samples=500, min_inliers=10)
    result = estimator.estimate_from_flow(geom, motion)
    assert result.valid, result.reason
    assert result.inlier_count >= 50
    assert result.T_current_from_previous[0, 3] != 0.0


def test_rigid_flow_residual_and_static_confidence() -> None:
    from geometry import compute_rigid_flow_residual, compute_static_confidence
    camera = _camera()
    h, w = camera.height, camera.width
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[..., 2] = 2.5
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    points[..., 0] = (u - camera.cx) * points[..., 2] / camera.fx
    points[..., 1] = (v - camera.cy) * points[..., 2] / camera.fy

    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = (0.05, 0.0, 0.0)

    # Compute ground truth rigid flow
    curr_pts = (transform[:3, :3] @ points.reshape(-1, 3).T).T + transform[:3, 3]
    rigid_pixels = camera.project(curr_pts).reshape(h, w, 2)
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[..., 0] = rigid_pixels[..., 0] - u
    flow[..., 1] = rigid_pixels[..., 1] - v

    # Inject independent moving object in center (10x10 patch)
    flow[50:60, 50:60, 0] += 5.0

    residual = compute_rigid_flow_residual(camera, points, flow, transform)
    static_conf = compute_static_confidence(residual, np.ones((h, w), dtype=np.float32), threshold=1.0)

    # Background has near-zero residual and near-1 static confidence
    assert np.nanmean(residual[:40, :40]) < 0.1
    assert np.mean(static_conf[:40, :40]) > 0.95

    # Moving foreground object has high residual and low static confidence
    assert np.nanmean(residual[52:58, 52:58]) > 4.0
    assert np.mean(static_conf[52:58, 52:58]) < 0.35


def test_dynamic_contamination_and_suppression() -> None:
    from geometry import compute_dynamic_contamination, PersistentGeometryConfig
    camera = CameraModel(10, 10, 10.0, 10.0, 5.0, 5.0)
    config = PersistentGeometryConfig(voxel_size=0.05, min_static_confidence=0.6)
    mapper = PersistentGeometryMapper(camera, config=config, mapping_stride=1)

    points = np.zeros((10, 10, 3), dtype=np.float32)
    points[..., 0] = np.linspace(1.0, 2.0, 10)[:, None]
    points[..., 1] = np.linspace(1.0, 2.0, 10)[None, :]
    points[..., 2] = 3.0
    depth = points[..., 2].copy()

    normals = np.zeros_like(points)
    normals[..., 2] = -1.0
    confidence = np.ones((10, 10), dtype=np.float32)

    # Mark points inside [1.0, 1.5] as dynamic (static_confidence = 0.1)
    static_conf = np.ones((10, 10), dtype=np.float32)
    static_conf[:5, :5] = 0.1

    from geometry.state import GeometryState
    geom = GeometryState(
        0.0,
        0,
        depth,
        points,
        np.ones((10, 10), bool),
        camera,
        "metric",
        normals=normals,
        normal_valid_mask=np.ones((10, 10), bool),
        confidence=confidence,
    )
    pose = CameraPoseState(0.0, 0, np.eye(4, dtype=np.float32), True, 1.0, 100, 0.0)

    mapper.update(geom, pose, static_confidence=static_conf)

    # Surfels should only contain static region (points > 1.5), not dynamic region
    arrays = mapper.map.arrays()
    contamination = compute_dynamic_contamination(
        arrays["positions_world"],
        np.array([0.9, 0.9, 2.8], dtype=np.float32),
        np.array([1.5, 1.5, 3.2], dtype=np.float32),
    )
    assert contamination == 0.0, f"Dynamic contamination: {contamination}%"


def test_map_assisted_hole_filling() -> None:
    from geometry import AdvancedGeometryEngine, PersistentGeometryConfig
    camera = _camera()
    h, w = camera.height, camera.width
    depth = np.full((h, w), 2.5, dtype=np.float32)
    engine = TemporalGeometryEngine(camera)

    config = PersistentGeometryConfig(
        mapping_stride=1,
        persistent_hole_fill_enabled=True,
        min_hole_fill_confidence=0.1,
    )
    adv_engine = AdvancedGeometryEngine(
        engine,
        advanced_geometry_enabled=True,
        config=config,
        persistent_hole_fill_enabled=True,
    )
    pose = CameraPoseState(0.0, 0, np.eye(4, dtype=np.float32), True, 1.0, 100, 0.0)

    # Populate map
    adv_engine.update(None, camera, 0, 0.0, DepthState(depth, 0.0, 0, "metric"), pose=pose)

    # Next frame has a hole (invalid center region) in current geometry
    depth_with_hole = depth.copy()
    depth_with_hole[20:30, 20:30] = np.nan
    pose1 = CameraPoseState(1/30.0, 1, np.eye(4, dtype=np.float32), True, 1.0, 100, 0.0)
    filled_state = adv_engine.update(None, camera, 1, 1/30.0, DepthState(depth_with_hole, 1/30.0, 1, "metric"), pose=pose1)

    # Hole region should be populated from persistent map
    hole_valid = filled_state.valid_mask[20:30, 20:30]
    assert hole_valid.sum() > 0
    assert np.all(filled_state.positions_3d[20:30, 20:30][hole_valid, 2] > 0.0)


def test_moving_object_pose_rejection() -> None:
    camera = _camera()
    rng = np.random.default_rng(21)
    n = 200
    points = np.column_stack((rng.uniform(-1, 1, n), rng.uniform(-0.7, 0.7, n), rng.uniform(2.5, 5.0, n))).astype(np.float32)
    transform = _transform()

    # 80% static background, 20% moving foreground
    curr = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    pixels = camera.project(curr).astype(np.float32)
    # inject independent motion for 20% of points
    pixels[:40] += rng.uniform(5.0, 15.0, (40, 2)).astype(np.float32)

    result = PoseEstimator(camera, max_samples=400, min_inliers=20).estimate(points, pixels)
    assert result.valid, result.reason
    assert result.inlier_count >= 120
    np.testing.assert_allclose(result.T_current_from_previous[:3, 3], transform[:3, 3], atol=1e-2)


def test_reprojection_metrics_calculation() -> None:
    from geometry import compute_reprojection_metrics
    proj = np.array([[2.0, 2.0], [2.0, np.nan]], dtype=np.float32)
    gt = np.array([[2.0, 2.02], [2.0, 2.0]], dtype=np.float32)
    metrics = compute_reprojection_metrics(proj, gt)
    assert metrics["coverage_percent"] == 75.0
    assert metrics["mean_error"] < 0.01


def test_advanced_geometry_engine_isolation_and_fallback() -> None:
    from geometry import AdvancedGeometryEngine
    camera = _camera()
    depth = np.full((camera.height, camera.width), 2.0, dtype=np.float32)
    base_engine = TemporalGeometryEngine(camera)
    adv_engine = AdvancedGeometryEngine(base_engine, advanced_geometry_enabled=False)

    # Disabled advanced mode yields identical result and None persistent state
    state, persistent = adv_engine.update_with_persistent(None, camera, 0, 0.0, DepthState(depth, 0.0, 0, "metric"))
    assert state.valid_mask.any()
    assert persistent is None

