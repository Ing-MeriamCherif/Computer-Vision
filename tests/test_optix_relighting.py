from __future__ import annotations

import numpy as np

from geometry.camera import CameraModel
from geometry.optix_relighting import OptixShadowRenderer


def test_depth_mesh_collapses_invalid_cells_and_keeps_fixed_topology():
    renderer = OptixShadowRenderer.__new__(OptixShadowRenderer)
    renderer.width, renderer.height = 4, 3
    renderer.faces = renderer._make_indices(renderer.width, renderer.height)
    camera = CameraModel(8, 6, 8.0, 8.0, 3.5, 2.5)
    depth = np.full((6, 8), 0.8, dtype=np.float32)
    valid = np.zeros_like(depth, dtype=bool)
    normals = np.zeros((6, 8, 3), dtype=np.float32)

    vertices, points, sampled_normals = renderer._mesh(depth, valid, normals, camera)

    assert vertices.shape == (6 * (renderer.width - 1) * (renderer.height - 1), 3)
    assert points.shape == sampled_normals.shape == (renderer.width * renderer.height, 3)
    assert np.isfinite(vertices).all()
    for triangle in vertices.reshape(-1, 3, 3):
        np.testing.assert_array_equal(triangle, np.repeat(triangle[:1], 3, axis=0))


def test_optix_visibility_shader_uses_hardware_trace_and_surface_hook():
    from pathlib import Path

    source = Path(__file__).parents[1] / "geometry" / "optix_shadow.cu"
    shader = source.read_text(encoding="utf-8")
    assert "optixTrace(" in shader
    assert "__raygen__shadow" in shader
    assert "OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT" in shader

    from geometry.gpu_lighting import _SURFACE_SHADER

    assert "uUseRtShadow" in _SURFACE_SHADER
    assert "rtVisibility(vUv, i)" in _SURFACE_SHADER
