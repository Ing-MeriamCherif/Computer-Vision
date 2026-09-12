# Computer-Vision-NRW

Phase 1 of the NRW AI & Vision Challenge geometry subsystem is implemented in
`geometry/`. It provides a calibrated camera model, explicit resize/crop/
letterbox transforms, vectorized depth-to-camera-space reconstruction, state
contracts, optional OpenCV checkerboard calibration, and deterministic
synthetic validation tools.

The camera convention is `+X` image-right, `+Y` image-down, and `+Z` forward.
Relative depth is accepted as an arbitrary but consistent Z scale; inverse
depth is rejected until a calibrated conversion is supplied.

```bash
python3 -m pytest -q
python3 main.py --shape tilted
```

See [`docs/geometry-phase1.md`](docs/geometry-phase1.md) for calibration and
demo commands. Neural depth, hand tracking, lighting, shadows, temporal
filtering, and the final renderer are intentionally deferred to later phases.
