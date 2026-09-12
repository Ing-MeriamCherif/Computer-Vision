#!/usr/bin/env python3
"""Verify CUDA runtime, real tensor execution, and geometry-kernel output."""

from __future__ import annotations

import json
import numpy as np

from geometry import CameraModel, TorchGeometryBackend, torch_cuda_status, validate_renderer_geometry


def main() -> int:
    status = torch_cuda_status()
    if not status.get("available"):
        print(json.dumps(status, indent=2))
        return 2
    camera = CameraModel(320, 180, 288.0, 162.0, 159.5, 89.5)
    depth = np.full((camera.height, camera.width), 2.0, np.float32)
    backend = TorchGeometryBackend("cuda")
    state = backend.process_depth(depth, camera)
    report = validate_renderer_geometry(state, validate_projection=True)
    result = {**status, "diagnostics": backend.last_diagnostics.__dict__ if hasattr(backend.last_diagnostics, "__dict__") else {name: getattr(backend.last_diagnostics, name) for name in backend.last_diagnostics.__slots__}, "geometry_valid": report.valid, "validation_errors": report.errors}
    print(json.dumps(result, indent=2))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
