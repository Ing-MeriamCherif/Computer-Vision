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

Dynamic rejection is represented by an optional `static_confidence` map. Low
static-confidence points are not inserted into the static map. A failed pose
returns no persistent state and leaves the previous map untouched, so the base
renderer can continue with Phase 1–4 geometry.

Run the deterministic headless demonstration:

```bash
python3 -m tools.persistent_geometry_demo --frames 5
```

No renderer, loop closure, global bundle adjustment, TSDF, NeRF, or semantic
tracking is part of this core implementation.
