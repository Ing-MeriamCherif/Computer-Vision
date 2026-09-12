# Phase 1 geometry foundation

The geometry package uses camera coordinates `+X right`, `+Y down`, and `+Z forward`. Pixel coordinates use a top-left origin. Depth is interpreted as Z-depth; relative depth remains in a consistent arbitrary coordinate scale, while inverse/disparity depth is rejected until an explicit calibrated conversion is supplied.

## Calibration

Capture checkerboard images at the exact RGB resolution used by the demo, then run:

```bash
python3 -m tools.calibrate_camera calibration/*.png \
  --board-width 9 --board-height 6 --square-size 0.024 \
  --output calibration/camera.json
```

Only detected views are used. The command reports mean reprojection error in pixels and writes human-readable JSON. OpenCV is an optional runtime dependency until calibration is needed.

## Tests and demo

```bash
python3 -m pytest -q
python3 main.py --shape tilted
python3 -m tools.geometry_demo --shape step --ply work/step.ply
```

The current phase intentionally does not implement neural depth, hand tracking, lighting, shadows, temporal filtering, or the final renderer. Those consume `DepthState` and later `GeometryState` fields.
