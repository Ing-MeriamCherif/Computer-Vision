# Geometry integration contract

The geometry package is the boundary between the depth producer (Person 2),
temporal geometry (Person 3), and the renderer (Person 4).  It operates in
camera space only; it does not infer camera motion or retain a world model.

## Person 2 -> Person 3

```python
depth = DepthState(
    depth=model_depth,                 # (H, W), preferred float32 Z-depth
    timestamp=frame_timestamp,
    source_frame_id=frame_id,          # must equal the frame being processed
    scale_mode="relative",             # or "metric"
    valid_mask=finite_positive_mask,   # optional; NaN/inf/zero/negative are invalid
    confidence=model_confidence,       # optional float map, interpreted in [0, 1]
)
geometry = engine.update(rgb, camera, frame_id, frame_timestamp, depth)
```

Depth is a Z coordinate in the camera convention (`+X` right, `+Y` down,
`+Z` forward), not Euclidean ray length.  Inputs may be integer or floating
point, but float32 is preferred.  The implementation never mutates caller
arrays.  Resolution must equal `camera.width x camera.height`; a model output
at another resolution requires an explicit resize/crop/letterbox transform and
an explicitly mapped `CameraModel` (`scaled_intrinsics` is available).  No
implicit intrinsic scaling is performed.  Relative depth may have arbitrary
consistent scale; metric depth is never rescaled.  A supplied depth timestamp
must be within the configured tolerance and its frame ID must match exactly.

## Person 3 -> Person 4

```python
geometry = engine.update(...)
P = geometry.positions_3d
N = geometry.normals
valid = geometry.valid_mask & geometry.normal_valid_mask
C = geometry.confidence
```

`depth` is the stabilized Z-depth actually used to construct `positions_3d`
and `normals`; all three describe the same frame and geometry.  `P` is `(H,W,3)`
float32 camera-space XYZ with invalid entries NaN.  `N` is `(H,W,3)` float32,
unit length and camera-facing (`N dot P <= 0`) where `normal_valid_mask` is
true.  `confidence` is the final renderer-facing geometry trust in `[0,1]`;
it already includes the selected spatial/temporal fusion trust but not a
second hidden renderer weighting.  `normal_confidence` is a diagnostic for
normal reliability and may be used as an additional shading gate.

Temporal diagnostics are `(H,W)` maps: `spatial_confidence`,
`history_confidence`, `temporal_confidence`, `temporal_age` (uint16 stale-frame
count), `history_valid`, `history_rejection_mask`, `occlusion_mask`,
`disocclusion_mask`, `selected_radius`, and `depth_alignment_residual`.
`validate_renderer_geometry()` checks these invariants without repairing state.

## Scheduling and ownership

`TemporalGeometryEngine` is stateful and must have one owning execution context
unless the caller supplies synchronization.  A future asynchronous deployment
should use a latest-ready pipeline: capture publishes frames, a depth worker
publishes the newest `DepthState`, a flow/geometry worker publishes the newest
`GeometryState`, and the renderer consumes the newest valid output without
blocking for inference.  Missing depth is handled by bounded image-space
history propagation.  Camera changes, frame discontinuities, timestamp gaps,
and scale-mode changes reset history.

The current path is CPU NumPy/OpenCV with float32 image-sized buffers.  The
future GPU path should keep camera/depth/warp/normals/fusion on device and use
the same flow and state contracts; zero-copy GPU operation is not implemented.
Quality knobs are `NormalConfig`, `TemporalConfig`, and the explicit
`OpenCVFlowProvider` method/sigma.  OpenCV software flow is the GTX 1650 Ti
fallback; no CUDA or NVIDIA Optical Flow dependency is required.
# Live integration boundary

The actual colleague sources are preserved under `integrations/` with their
branch/commit provenance in `integrations/README.md` and each `SOURCE.md`.
`geometry.hand_control.HandControlEngine` now calls their real
`create_tracker()` and `OneEuroFilter` implementations through an adapter;
their `talel-hand` reacquire/hold settings are retained in the imported
`config.py` and `upstream_main.py` reference.

`geometry.lighting.light_from_palm` remains the legacy CPU/native-app adapter
and samples `GeometryState.depth`. P123 Mode 7 does not use it: the async XYZ
worker estimates metric Z from palm width and `CameraModel` intrinsics without
waiting for scene geometry, then adapts fresh `HandXYZ.xyz_camera` records to
`LightState`. The Mode 7 OpenGL 3.3 renderer applies linear-space diffuse,
additive specular, multi-light camera-space area-shadow rays, sample-to-light
volumetric visibility, and projected 3D orbs; `shade_geometry` is its CPU
reference/fallback.

The complete colleague `feature/depth` tree is preserved under
`integrations/colleague_depth/depth`. `geometry.ColleagueDepthProvider` loads
its original `DepthModel` when `NRW_DEPTH_SOURCE=colleague` is set. The normal
live profile uses our frame-ID-corrected `geometry.async_pipeline`, so packets
never reprocess an unchanged frame; diagnostics expose depth and geometry age.

The root command now satisfies the challenge launch contract:

```bash
python3 main.py
```

Use `python3 main.py --smoke --shape tilted` for deterministic geometry
validation.
