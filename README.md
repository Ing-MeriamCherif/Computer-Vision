# Computer-Vision-NRW

Phases 1–5 of the NRW AI & Vision Challenge geometry subsystem are
implemented in `geometry/`. It provides a calibrated camera model, explicit
resize/crop/letterbox transforms, vectorized depth-to-camera-space
reconstruction, camera-facing surface normals, discontinuity-aware neighbor
selection, spatial confidence, state contracts, optional OpenCV checkerboard
calibration, deterministic synthetic validation tools, short-term temporal
reconstruction with motion reprojection, history rejection, relative-depth
alignment, confidence-aware fusion, and production integration/diagnostics.

The camera convention is `+X` image-right, `+Y` image-down, and `+Z` forward.
Relative depth is accepted as an arbitrary but consistent Z scale; inverse
depth is rejected until a calibrated conversion is supplied.

```bash
python3 -m pytest -q
python3 main.py --shape tilted
```

See [`docs/geometry-phase1.md`](docs/geometry-phase1.md) for calibration and
demo commands and [`docs/geometry-phase2.md`](docs/geometry-phase2.md) for
normal algorithms, [`docs/geometry-phase3.md`](docs/geometry-phase3.md) for
temporal contracts, and [`docs/geometry-integration.md`](docs/geometry-integration.md)
plus [`docs/geometry-performance.md`](docs/geometry-performance.md) for Phase 4
integration and benchmark guidance. Neural depth, hand tracking, lighting,
shadows, and the final renderer remain outside this geometry core.

Phase 4 tools: `python3 -m tools.geometry_benchmark --json`,
`python3 -m tools.geometry_stress`, and `python3 -m tools.geometry_capabilities`.

Phase 5 is optional persistent world geometry: PnP camera pose estimation and
a bounded voxel-surfel map are available through `geometry.pose` and
`geometry.persistent`, while `TemporalGeometryEngine` remains the safe default.
See [`docs/geometry-phase5.md`](docs/geometry-phase5.md) and run
`python3 -m tools.persistent_geometry_demo` for a deterministic headless demo.
