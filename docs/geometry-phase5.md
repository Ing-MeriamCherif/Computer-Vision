# Phase 5: optional persistent geometry

Phase 5 is an additive, fail-safe world-memory track. `TemporalGeometryEngine`
continues to produce the authoritative Phase 1–4 `GeometryState`; the optional
`AdvancedGeometryEngine` can pass a valid `PoseEstimateResult` to
`PersistentGeometryMapper` and expose a separate `PersistentGeometryState`.

## Coordinate convention

Transforms are named `T_A_from_B` and map homogeneous points from frame B into
frame A. The first valid camera pose anchors the world (`T_world_from_camera =
I`). PnP estimates `T_current_from_previous`; accumulation uses:

```
T_world_from_current = T_world_from_previous @ inverse(T_current_from_previous)
```

Relative monocular depth is stored in session-relative scene units, not metres.
Metric `DepthState` values retain metric units.

## Bounded surfel map

`SurfelMap` voxel-quantizes candidates, fuses only compatible position/normal
observations, and evicts stale or over-capacity entries. Mapping is single-owner;
callers should pass read-only snapshots to renderers. `reproject()` performs a
vectorized world-to-camera transform and nearest-depth z-buffer, transforming
normals with rotation only.

## Static/dynamic classification and rigid flow residual

Dynamic objects are detected by comparing observed optical flow against the rigid
motion induced by estimated camera motion `T_current_from_previous`:

- `compute_rigid_flow_residual()`: Calculates per-pixel discrepancy in pixels.
- `compute_static_confidence()`: Modulates spatial confidence by $\exp(-\text{residual} / \tau)$.
- `PersistentGeometryConfig.min_static_confidence`: Ensures dynamic objects are excluded from the static surfel map.
- `compute_dynamic_contamination()`: Verifies that no dynamic surfels contaminate the persistent map.

## Optional map-assisted hole filling

When enabled via `persistent_hole_fill_enabled=True` in `PersistentGeometryConfig`:
`persistent_hole_fill()` populates invalid or missing depth patches using high-confidence
persistent surfel reprojections, without overwriting fresh high-confidence current geometry.

## Run tools

Run the deterministic headless demonstration:

```bash
python3 tools/persistent_geometry_demo.py --frames 10 --ply work/surfel_map.ply
```

No renderer, loop closure, global bundle adjustment, TSDF, NeRF, or semantic
tracking is part of this core implementation.

