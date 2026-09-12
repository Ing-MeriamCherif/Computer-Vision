import json

import numpy as np

from geometry import CameraModel, DepthState, MotionState, NormalConfig, NormalMode, TemporalConfig, TemporalGeometryEngine, align_inverse_depth, estimate_normals, validate_renderer_geometry
from geometry.motion import OpenCVFlowProvider
from geometry.normals import _estimate_at_radius, _select_multiscale
from tools.geometry_benchmark import benchmark
from tools.geometry_stress import run


def test_validator_custom_radii_and_projection_consistency() -> None:
    camera = CameraModel(24, 16, 31.0, 27.0, 7.25, 5.5)
    depth = np.full((16, 24), 2.0, np.float32)
    state = TemporalGeometryEngine(camera, TemporalConfig(normal_mode=NormalMode.MULTI_SCALE), normal_config=NormalConfig(radii=(2, 4, 8))).update(None, camera, 0, 0.0, DepthState(depth, 0.0, 0, "relative"))
    assert validate_renderer_geometry(state, validate_projection=True).valid
    assert validate_renderer_geometry(state, normal_config=NormalConfig(radii=(2, 4, 8))).valid
    state.selected_radius[3, 3] = 1
    assert not validate_renderer_geometry(state, normal_config=NormalConfig(radii=(2, 4, 8))).valid


def test_validator_checks_every_confidence_field() -> None:
    camera = CameraModel(12, 8, 20, 22, 5.5, 3.5)
    depth = np.full((8, 12), 2.0, np.float32)
    state = TemporalGeometryEngine(camera).update(None, camera, 0, 0.0, DepthState(depth, 0.0, 0, "relative"))
    state.spatial_confidence[0, 0] = 1.2
    assert not validate_renderer_geometry(state).valid


def test_alignment_sampling_is_deterministic_and_bounded() -> None:
    y, x = np.indices((40, 50))
    history = (2.0 + 0.01 * x + 0.02 * y).astype(np.float32)
    current = (history * 1.2).astype(np.float32)
    kwargs = dict(min_samples=64, max_samples=200)
    first = align_inverse_depth(current, history, np.ones_like(history, bool), **kwargs)
    second = align_inverse_depth(current, history, np.ones_like(history, bool), **kwargs)
    assert first.sample_count <= 200 and first.input_sample_count == 2000
    assert first == second


def test_streaming_multiscale_matches_reference() -> None:
    camera = CameraModel(20, 16, 24, 25, 9.5, 7.5)
    depth = np.full((16, 20), 2.0, np.float32)
    points, valid = __import__("geometry").backproject_depth(depth, camera)
    config = NormalConfig(radii=(4, 1, 2))
    reference = _select_multiscale([_estimate_at_radius(points, valid, depth, r, True, config) for r in config.radii], config.multi_scale_acceptance)
    streamed = estimate_normals(points, valid, depth, NormalMode.MULTI_SCALE, config)
    np.testing.assert_allclose(streamed.normals, reference.normals, equal_nan=True)
    np.testing.assert_array_equal(streamed.normal_valid_mask, reference.normal_valid_mask)
    np.testing.assert_array_equal(streamed.selected_radius, reference.selected_radius)


def test_flow_scale_restores_vector_magnitude() -> None:
    provider = OpenCVFlowProvider(method="farneback", flow_scale=0.5)
    provider._compute_one = lambda previous, current: np.dstack((np.full(previous.shape, 1.0, np.float32), np.full(previous.shape, 2.0, np.float32)))
    previous = np.zeros((12, 16), np.uint8)
    state = provider.compute(previous, previous, 0, 1, 0.1)
    np.testing.assert_allclose(state.forward_flow[..., 0], 2.0)
    np.testing.assert_allclose(state.forward_flow[..., 1], 4.0)


def test_stress_schedule_and_injection_counters_are_real() -> None:
    report = run(20, 32, 24, 2, 4, True, True, True, [True, False, False, True], motion_dx=0.5)
    assert report["invalid_depth_injections"] > 0
    assert report["bad_flow_injections"] > 0
    assert report["depth_schedule"] == [True, False, False, True]
    assert report["motion_dx"] == 0.5


def test_benchmark_history_starts_at_frame_one_and_remains_valid() -> None:
    result = benchmark(24, 16, warmups=1, iterations=2)
    assert result["stages"]["temporal_update_history_only"]["quality"]["geometry_valid_percent"] > 0
    assert result["stages"]["temporal_update_history_only"]["iterations"] == 2
