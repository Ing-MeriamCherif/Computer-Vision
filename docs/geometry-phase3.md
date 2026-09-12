# Phase 3: temporal geometry

Phase 3 adds short-term, image-space temporal reconstruction around the Phase 2
spatial geometry. It does not assume camera pose or persistent world geometry.

## Contracts and algorithms

`MotionState` uses pixel-unit flow with explicit frame IDs:

- `forward_flow = F_prev_to_cur`: `p_cur = p_prev + F_prev_to_cur(p_prev)`.
- `backward_flow = F_cur_to_prev`: `p_prev = p_cur + F_cur_to_prev(p_cur)`.

History is gathered with backward warping (`current + backward_flow`) using
bilinear interpolation for continuous fields and nearest-neighbor for masks and
age. Out-of-frame and invalid source samples are rejected; warped normals are
re-normalized.

`TemporalGeometryEngine` aligns relative-depth history in inverse-depth space:
`q_current = a * q_history + b`, using weighted least squares with iterative
Huber-style reweighting and residual trimming. Metric depth is never
rescaled. Large disagreement, bad flow, photometric mismatch, disocclusion,
and excessive age reduce or reject history. When no new `DepthState` exists,
valid history is motion-reprojected, confidence decays, and age increments
until `max_history_age`.

The returned `GeometryState.depth` is the stabilized depth used to reconstruct
`positions_3d` and recompute Phase 2 normals. `spatial_confidence`,
`history_confidence`, `temporal_confidence`, `history_valid`, and
`occlusion_mask` remain separately available for downstream consumers.

## Deterministic demo and tests

```bash
python3 -m pytest -q
python3 -m tools.temporal_geometry_demo --width 320 --height 180
python3 -m tools.temporal_geometry_demo --width 640 --height 360
python3 -m tools.temporal_geometry_demo --width 1280 --height 720
```

The demo reports raw versus stabilized depth jitter, history acceptance, and
update timing. Temporal tests inject exact synthetic flow; OpenCV flow is only
an optional smoke provider and NVIDIA Optical Flow is not mandatory.
