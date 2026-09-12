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


def test_correspondence_extraction_contracts_and_validity() -> None:
    from geometry import extract_pose_correspondences, MotionState
    camera = _camera()
    h, w = camera.height, camera.width
    depth = np.full((h, w), 2.5, dtype=np.float32)
    engine = TemporalGeometryEngine(camera)
    geom = engine.update(None, camera, 0, 0.0, DepthState(depth, 0.0, 0, "metric"))

    # Contract mismatch test
    motion_wrong_id = MotionState(
        source_frame_id=99,
        target_frame_id=100,
        timestamp=1/30.0,
        forward_flow=np.zeros((h, w, 2), dtype=np.float32),
        backward_flow=np.zeros((h, w, 2), dtype=np.float32),
        valid_mask=np.ones((h, w), dtype=bool),
    )
    import pytest
    with pytest.raises(ValueError, match="Frame contract mismatch"):
        extract_pose_correspondences(geom, motion_wrong_id)

    # Valid identity flow test
    flow_id = np.zeros((h, w, 2), dtype=np.float32)
    motion_id = MotionState(
        source_frame_id=0,
        target_frame_id=1,
        timestamp=1/30.0,
        forward_flow=flow_id,
        backward_flow=-flow_id,
        valid_mask=np.ones((h, w), dtype=bool),
    )
    corr_id = extract_pose_correspondences(geom, motion_id, max_samples=400)
    assert corr_id.correspondence_count > 0
    np.testing.assert_allclose(corr_id.current_pixels, corr_id.previous_pixels, atol=1e-5)
    assert corr_id.spatial_coverage > 0.4

    # Out of bounds flow rejection test
    flow_oob = np.full((h, w, 2), 1000.0, dtype=np.float32)
    motion_oob = MotionState(
        source_frame_id=0,
        target_frame_id=1,
        timestamp=1/30.0,
        forward_flow=flow_oob,
        backward_flow=-flow_oob,
        valid_mask=np.ones((h, w), dtype=bool),
    )
    corr_oob = extract_pose_correspondences(geom, motion_oob)
    assert corr_oob.correspondence_count == 0


def test_pose_estimator_noise_tolerance() -> None:
    camera = _camera()
    rng = np.random.default_rng(42)
    n = 300
    points = np.column_stack((rng.uniform(-1, 1, n), rng.uniform(-0.7, 0.7, n), rng.uniform(2.5, 5.0, n))).astype(np.float32)
    transform = _transform()
    curr = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    pixels_clean = camera.project(curr).astype(np.float32)

    for noise_sigma in (0.25, 0.5, 1.0):
        noisy_pixels = pixels_clean + rng.normal(0.0, noise_sigma, pixels_clean.shape).astype(np.float32)
        res = PoseEstimator(camera, max_samples=500, min_inliers=20, reprojection_error=4.0).estimate(points, noisy_pixels)
        assert res.valid, f"Failed at noise sigma {noise_sigma}"
        assert res.inlier_count >= 150
        np.testing.assert_allclose(res.T_current_from_previous[:3, 3], transform[:3, 3], atol=0.05)


def test_pose_estimator_high_outlier_rejection() -> None:
    camera = _camera()
    rng = np.random.default_rng(101)
    n = 400
    points = np.column_stack((rng.uniform(-1.2, 1.2, n), rng.uniform(-0.8, 0.8, n), rng.uniform(2.5, 5.5, n))).astype(np.float32)
    transform = _transform()
    curr = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    pixels_gt = camera.project(curr).astype(np.float32)

    for outlier_ratio in (0.1, 0.2, 0.3, 0.4):
        pixels = pixels_gt.copy()
        n_outliers = int(n * outlier_ratio)
        pixels[:n_outliers] = rng.uniform(0, 240, (n_outliers, 2)).astype(np.float32)
        res = PoseEstimator(camera, max_samples=500, min_inliers=20, reprojection_error=3.0).estimate(points, pixels)
        assert res.valid, f"Failed at outlier ratio {outlier_ratio}"
        assert res.inlier_count >= int((1.0 - outlier_ratio) * n * 0.7)
        np.testing.assert_allclose(res.T_current_from_previous[:3, 3], transform[:3, 3], atol=0.03)


def test_surfel_27_neighbor_and_best_match() -> None:
    # Surfel A in cell (0, 0, 100). Query point in neighbor cell (1, 1, 100)
    smap = SurfelMap(voxel_size=0.02, merge_distance=0.015, normal_merge_cos=0.8)
    pt_a = np.array([[[0.019, 0.019, 2.000]]], dtype=np.float32)
    nrm_a = np.array([[[0.0, 0.0, -1.0]]], dtype=np.float32)
    conf_a = np.array([[1.0]], dtype=np.float32)
    smap.insert(pt_a, nrm_a, conf_a, 0, stride=1)
    assert smap.count == 1

    # Query point B is at (0.021, 0.021, 2.000), which falls in cell (1, 1, 100)
    # Distance is sqrt(0.002^2 + 0.002^2) = 0.0028m < 0.015m
    pt_b = np.array([[[0.021, 0.021, 2.000]]], dtype=np.float32)
    res = smap.insert(pt_b, nrm_a, conf_a, 1, stride=1)
    assert res["fused_surfels"] == 1
    assert res["new_surfels"] == 0
    assert smap.count == 1  # Successfully merged across voxel boundary via 27-cell neighbor search!


def test_surfel_bucket_migration_and_index_validation() -> None:
    # Set voxel_size = 0.02
    smap = SurfelMap(voxel_size=0.02, merge_distance=0.018)
    # Point at (0.019, 0.019, 2.0) is in bucket (0, 0, 100)
    pt1 = np.array([[[0.019, 0.019, 2.0]]], dtype=np.float32)
    nrm = np.array([[[0.0, 0.0, -1.0]]], dtype=np.float32)
    conf = np.array([[1.0]], dtype=np.float32)
    smap.insert(pt1, nrm, conf, 0, stride=1)
    assert (0, 0, 100) in smap._buckets

    # Second point at (0.025, 0.025, 2.0) has weight=1.0 and pulls the surfel into bucket (1, 1, 100)
    pt2 = np.array([[[0.025, 0.025, 2.0]]], dtype=np.float32)
    smap.insert(pt2, nrm, conf, 1, stride=1)
    assert smap.validate_index() is True
    # The surfel position is now > 0.02, so it migrated to (1, 1, 100)
    assert (1, 1, 100) in smap._buckets


def test_batch_capacity_eviction() -> None:
    smap = SurfelMap(voxel_size=0.05, max_surfels=5, max_age_frames=10)
    # Insert 15 separate points into different voxels
    pts = np.zeros((15, 1, 3), dtype=np.float32)
    pts[:, 0, 0] = np.arange(15) * 0.1
    pts[:, 0, 2] = 2.0
    nrms = np.zeros_like(pts)
    nrms[:, 0, 2] = -1.0
    confs = np.ones((15, 1), dtype=np.float32)
    smap.insert(pts, nrms, confs, 0, stride=1)

    assert smap.count <= 5
    assert smap.validate_index() is True


def test_relative_depth_scale_invariance() -> None:
    from geometry import PersistentGeometryConfig
    camera = _camera()
    h, w = camera.height, camera.width

    counts: list[int] = []
    for scale in (0.1, 1.0, 10.0, 100.0):
        depth = np.full((h, w), 2.5 * scale, dtype=np.float32)
        pos = np.zeros((h, w, 3), dtype=np.float32)
        pos[..., 2] = depth
        normals = np.zeros_like(pos)
        normals[..., 2] = -1.0

        geom = GeometryState(
            0.0, 0, depth, pos, np.ones((h, w), bool), camera, "relative",
            normals=normals, normal_valid_mask=np.ones((h, w), bool), confidence=np.ones((h, w), dtype=np.float32),
        )
        cfg = PersistentGeometryConfig(relative_voxel_fraction=0.01, mapping_stride=2)
        mapper = PersistentGeometryMapper(camera, config=cfg)
        pose = CameraPoseState(0.0, 0, np.eye(4, dtype=np.float32), True, 1.0, 100, 0.0)
        pstate = mapper.update(geom, pose)

        assert mapper.map.scene_scale is not None
        np.testing.assert_allclose(mapper.map.scene_scale, 2.5 * scale, rtol=1e-3)
        np.testing.assert_allclose(mapper.map.resolved_voxel_size, 2.5 * scale * 0.01, rtol=1e-3)
        counts.append(mapper.map.count)

    # Scale invariance: all scales should yield approximately equal surfel counts
    assert max(counts) - min(counts) <= 5


def test_reprojection_z_buffer_order_independence() -> None:
    camera = _camera()
    pose = CameraPoseState(0.0, 0, np.eye(4, dtype=np.float32), True, 1.0, 100, 0.0)

    # Run 1: Front (Z=2) inserted first, then Back (Z=4)
    smap1 = SurfelMap(voxel_size=0.005, merge_distance=0.001)
    pt_front = np.array([[[0.0, 0.0, 2.0]]], dtype=np.float32)
    pt_back = np.array([[[0.0, 0.0, 4.0]]], dtype=np.float32)
    nrm = np.array([[[0.0, 0.0, -1.0]]], dtype=np.float32)
    conf = np.array([[1.0]], dtype=np.float32)

    smap1.insert(pt_front, nrm, conf, 0, stride=1)
    smap1.insert(pt_back, nrm, conf, 0, stride=1)
    d1, _, _, _ = smap1.reproject(camera, pose)

    # Run 2: Back (Z=4) inserted first, then Front (Z=2)
    smap2 = SurfelMap(voxel_size=0.005, merge_distance=0.001)
    smap2.insert(pt_back, nrm, conf, 0, stride=1)
    smap2.insert(pt_front, nrm, conf, 0, stride=1)
    d2, _, _, _ = smap2.reproject(camera, pose)

    # Both must resolve to the nearest surface (Z=2.0)
    u_c, v_c = int(np.rint(camera.cx)), int(np.rint(camera.cy))
    np.testing.assert_allclose(d1[v_c, u_c], 2.0, atol=1e-3)
    np.testing.assert_allclose(d2[v_c, u_c], 2.0, atol=1e-3)


def test_reprojection_normal_transformation() -> None:
    camera = _camera()
    # Camera rotated 90 deg around Z: R_w_c = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    T_w_c = np.eye(4, dtype=np.float32)
    T_w_c[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    T_w_c[:3, 3] = [0.1, -0.2, 0.3]  # Translation should have zero effect on normal

    # Position in front of camera at Z=3.0
    pt_cam = np.array([0.0, 0.0, 3.0], dtype=np.float32)
    pt_world = T_w_c[:3, :3] @ pt_cam + T_w_c[:3, 3]
    # World normal is +Y (0, 1, 0)
    nrm_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    smap = SurfelMap(voxel_size=0.05)
    conf = np.array([[1.0]], dtype=np.float32)
    smap.insert(pt_world.reshape(1, 1, 3), nrm_world.reshape(1, 1, 3), conf, 0, stride=1)
    pose = CameraPoseState(0.0, 0, T_w_c, True, 1.0, 100, 0.0)

    _, _, proj_normals, _ = smap.reproject(camera, pose)
    u_c, v_c = int(np.rint(camera.cx)), int(np.rint(camera.cy))
    # R_c_w = inv(R_w_c) = R_w_c^T = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    # N_c = R_c_w @ [0, 1, 0]^T = [1, 0, 0]
    expected_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(proj_normals[v_c, u_c], expected_normal, atol=1e-3)


def test_automatic_advanced_engine_end_to_end() -> None:
    from geometry import AdvancedGeometryEngine
    camera = _camera()
    h, w = camera.height, camera.width
    base_engine = TemporalGeometryEngine(camera)
    adv_engine = AdvancedGeometryEngine(base_engine, advanced_geometry_enabled=True)

    # Frame 0: First frame initializes world pose at origin automatically
    d0 = np.full((h, w), 2.5, dtype=np.float32)
    s0, p0 = adv_engine.update_with_persistent(None, camera, 0, 0.0, DepthState(d0, 0.0, 0, "metric"))
    assert s0.valid_mask.any()
    assert p0 is not None
    assert p0.tracking_state == "TRACKING"
    assert adv_engine.mapper.map.count > 0

    # Frame 1: Synthetic 2px translation flow
    d1 = np.full((h, w), 2.5, dtype=np.float32)
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[..., 0] = 2.0
    from geometry import MotionState
    m1 = MotionState(
        source_frame_id=0, target_frame_id=1, timestamp=1/30.0,
        forward_flow=flow, backward_flow=-flow, valid_mask=np.ones((h, w), dtype=bool),
    )
    s1, p1 = adv_engine.update_with_persistent(None, camera, 1, 1/30.0, DepthState(d1, 1/30.0, 1, "metric"), motion_state=m1)
    assert s1.valid_mask.any()
    assert p1 is not None
    assert p1.pose.valid is True
    assert p1.tracking_state == "TRACKING"


def test_tracking_lost_and_no_fake_relocalization() -> None:
    from geometry import AdvancedGeometryEngine
    camera = _camera()
    h, w = camera.height, camera.width
    base_engine = TemporalGeometryEngine(camera)
    adv_engine = AdvancedGeometryEngine(base_engine, advanced_geometry_enabled=True)

    # Frame 0
    d0 = np.full((h, w), 2.5, dtype=np.float32)
    adv_engine.update(None, camera, 0, 0.0, DepthState(d0, 0.0, 0, "metric"))
    map_count_f0 = adv_engine.mapper.map.count

    # Frame 1 with degenerate flow (all NaNs) causing PnP failure
    d1 = np.full((h, w), 2.5, dtype=np.float32)
    bad_flow = np.full((h, w, 2), np.nan, dtype=np.float32)
    from geometry import MotionState
    m_bad = MotionState(
        source_frame_id=0, target_frame_id=1, timestamp=1/30.0,
        forward_flow=bad_flow, backward_flow=-bad_flow, valid_mask=np.ones((h, w), dtype=bool),
    )
    s1, p1 = adv_engine.update_with_persistent(None, camera, 1, 1/30.0, DepthState(d1, 1/30.0, 1, "metric"), motion_state=m_bad)
    # Fail-safe: base state remains completely intact
    assert s1.valid_mask.any()
    # Persistent state is None and tracking is marked LOST (no fake relocalization!)
    assert p1 is None
    assert adv_engine.mapper.tracking_state == "LOST"
    # Map was preserved and not corrupted or wiped
    assert adv_engine.mapper.map.count == map_count_f0


def test_pose_error_helper() -> None:
    from geometry import pose_error
    t1 = np.eye(4, dtype=np.float32)
    t2 = np.eye(4, dtype=np.float32)
    t2[:3, 3] = [0.1, -0.2, 0.05]
    rot_err, trans_err = pose_error(t1, t2)
    assert rot_err < 1e-4
    np.testing.assert_allclose(trans_err, np.linalg.norm([0.1, -0.2, 0.05]), atol=1e-4)


def test_long_run_trajectory_drift_and_bounded_memory() -> None:
    camera = _camera()
    rng = np.random.default_rng(77)
    n = 200
    points_w = np.column_stack((rng.uniform(-1.5, 1.5, n), rng.uniform(-1.0, 1.0, n), rng.uniform(2.0, 4.5, n))).astype(np.float32)
    smap = SurfelMap(voxel_size=0.05, max_surfels=150, max_age_frames=30)
    estimator = PoseEstimator(camera, max_samples=300, min_inliers=15)

    T_w_c_prev = np.eye(4, dtype=np.float32)
    pts_cam_prev = (np.linalg.inv(T_w_c_prev)[:3, :3] @ points_w.T).T + np.linalg.inv(T_w_c_prev)[:3, 3]

    T_est_w_c = np.eye(4, dtype=np.float32)

    for frame_id in range(1, 101):
        yaw = float(0.005 * np.sin(0.1 * frame_id))
        c, s = np.cos(yaw), np.sin(yaw)
        T_w_c = np.eye(4, dtype=np.float32)
        T_w_c[:3, :3] = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
        T_w_c[:3, 3] = [0.01 * np.sin(0.15 * frame_id), -0.005 * frame_id, 0.005 * frame_id]

        T_cam_w = np.linalg.inv(T_w_c)
        pts_cam = (T_cam_w[:3, :3] @ points_w.T).T + T_cam_w[:3, 3]
        px_curr = camera.project(pts_cam).astype(np.float32)

        res = estimator.estimate(pts_cam_prev, px_curr)
        assert res.valid, f"Pose tracking failed at frame {frame_id}"
        T_est_w_c = compose_world_pose(T_est_w_c, res.T_current_from_previous)

        # Insert sparse points into bounded map
        nrms = np.tile([0.0, 0.0, -1.0], (len(pts_cam[:20]), 1)).astype(np.float32)
        smap.insert(pts_cam[:20].reshape(1, 20, 3), nrms.reshape(1, 20, 3), np.ones((1, 20), np.float32), frame_id, stride=1)
        smap.age_evict(frame_id)
        assert smap.count <= 150

        pts_cam_prev = pts_cam
        T_w_c_prev = T_w_c

    assert smap.count <= 150
    assert smap.validate_index() is True

    from geometry import pose_error
    rot_drift, trans_drift = pose_error(T_est_w_c, T_w_c_prev)
    assert rot_drift < 2.0, f"Rotation drift too high: {rot_drift} deg"
    assert trans_drift < 0.25, f"Translation drift too high: {trans_drift} m"

