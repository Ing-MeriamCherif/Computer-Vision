# Computer-Vision-NRW

Phase 1, Phase 2, and Phase 3 of the NRW AI & Vision Challenge geometry subsystem are
implemented in `geometry/`. It provides a calibrated camera model, explicit
resize/crop/letterbox transforms, vectorized depth-to-camera-space
reconstruction, camera-facing surface normals, discontinuity-aware neighbor
selection, spatial confidence, state contracts, optional OpenCV checkerboard
calibration, deterministic synthetic validation tools, and short-term temporal
reconstruction with motion reprojection, history rejection, relative-depth
alignment, and confidence-aware fusion.

The camera convention is `+X` image-right, `+Y` image-down, and `+Z` forward.
Relative depth is accepted as an arbitrary but consistent Z scale; inverse
depth is rejected until a calibrated conversion is supplied.

```bash
python3 -m pytest -q
python3 main.py --shape tilted
```

See [`docs/geometry-phase1.md`](docs/geometry-phase1.md) for calibration and
demo commands and [`docs/geometry-phase2.md`](docs/geometry-phase2.md) for
normal algorithms and [`docs/geometry-phase3.md`](docs/geometry-phase3.md) for
temporal contracts and benchmarks. Neural depth, hand tracking, lighting,
shadows, and the final renderer are intentionally deferred to later phases.
