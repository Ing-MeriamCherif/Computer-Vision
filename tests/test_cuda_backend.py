import numpy as np
import pytest

from geometry import CameraModel, TorchGeometryBackend, validate_renderer_geometry


torch = pytest.importorskip("torch")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_torch_geometry_backend_preserves_renderer_contract(device: str) -> None:
    camera = CameraModel(32, 24, 30.0, 31.0, 13.25, 10.75)
    depth = np.full((24, 32), 2.0, np.float32)
    depth[2:4, 5:7] = np.nan
    state = TorchGeometryBackend(device).process_depth(depth, camera)
    report = validate_renderer_geometry(state, validate_projection=True)
    assert report.valid, report.errors

