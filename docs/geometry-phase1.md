# Phase 1 geometry foundation

The geometry package uses camera coordinates `+X right`, `+Y down`, and `+Z forward`. Pixel coordinates use a top-left origin. Depth is interpreted as Z-depth; relative depth remains in a consistent arbitrary coordinate scale, while inverse/disparity depth is rejected until an explicit calibrated conversion is supplied.

## Coordinate and Pixel Semantics

- **Origin and Pixel Coordinates**: Continuous coordinates $(u, v)$ start at $(0.0, 0.0)$ at the outer top-left corner of the image plane. Integer indices $(u, v)$ with $u \in [0, W - 1], v \in [0, H - 1]$ define sample rays, whose geometric pixel centers lie at $(u + 0.5, v + 0.5)$.
- **Intrinsics Transformation**: Image transforms apply affine mapping $\begin{bmatrix} u' \\ v' \end{bmatrix} = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}$. Camera intrinsics transform via $f_x' = s_x f_x, f_y' = s_y f_y, c_x' = s_x c_x + t_x, c_y' = s_y c_y + t_y$.
- **Resolution Consistency**: `backproject_depth` strictly validates `depth.shape == (camera.height, camera.width)`. Mismatches are rejected with a descriptive `ValueError`. To back-project at depth resolution, camera models must be mapped using `ImageTransform` beforehand.
- **Framework Integration Caveats**: Frameworks such as OpenCV (`cv2.resize`) and PyTorch (`align_corners=False`) incorporate half-pixel adjustments during spatial resampling. These differences must be explicitly accounted for at pipeline integration boundaries.

## Calibration

Capture checkerboard images at the exact RGB resolution used by the demo, then run:

```bash
python3 -m tools.calibrate_camera calibration/*.png \
  --board-width 9 --board-height 6 --square-size 0.024 \
  --output calibration/camera.json
```

All calibration images within a calibration set must share the exact same resolution; mixed-resolution sets are rejected. Only detected views are used. The command reports mean reprojection error in pixels and writes human-readable JSON. OpenCV is an optional runtime dependency until calibration is needed.

## Synthetic Verification & Exact Ray-Plane Intersection

Synthetic planar scenes utilize exact analytical ray-plane intersection ($X = t \cdot r, n \cdot X + d = 0 \implies t = -d / (n \cdot r)$) in `exact_plane_depth` and `tilted_plane`, producing clean zero-residual verification geometries.

## Tests and demo

```bash
python3 -m pytest -q
python3 main.py --shape tilted
python3 -m tools.geometry_demo --shape step --ply work/step.ply
```

The current phase intentionally does not implement neural depth, hand tracking, lighting, shadows, temporal filtering, or the final renderer. Those consume `DepthState` and later `GeometryState` fields.
