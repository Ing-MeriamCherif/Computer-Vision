# Phase 2: spatial normals and confidence

Phase 2 extends the Phase 1 camera-space position field with three selectable
normal estimators:

- `baseline`: central finite differences with one-sided fallback, without depth-edge rejection.
- `edge_aware`: rejects neighboring samples whose relative Z jump exceeds `discontinuity_threshold`, then selects same-surface one-sided support where needed.
- `multi_scale`: evaluates configured radii (default `1, 2, 4`) and selects the smallest radius meeting the confidence acceptance score, falling back to the best valid radius.

`NormalConfig.min_tangent_conditioning` is the dimensionless sine-angle
conditioning threshold `||T_y × T_x|| / (||T_y|| ||T_x||)`.  The legacy
`min_cross_norm` keyword remains a deprecated compatibility alias.  Radii are
validated as positive integers, sorted ascending, and duplicate values are
deduplicated before multi-scale selection.

All modes are fully vectorized over pixels. Normals are oriented toward the
camera using `N dot P <= 0`; invalid normals are NaN and have zero confidence.
Confidence is intentionally spatial-only and interpretable: compatible-neighbor
support, tangent quality, cross-product conditioning, and a bounded
discontinuity penalty. A supplied `DepthState.confidence` is clamped and
multiplied in without temporal information.

## Validation and benchmarks

```bash
python3 -m pytest -q
python3 -m tools.geometry_demo --shape tilted --normals --mode baseline
python3 -m tools.geometry_demo --shape tilted --normals --mode edge_aware
python3 -m tools.geometry_demo --shape sphere --normals --mode multi_scale
python3 -m tools.geometry_demo --shape plane --benchmark --width 320 --height 180
python3 -m tools.geometry_demo --shape plane --benchmark --width 640 --height 360
python3 -m tools.geometry_demo --shape plane --benchmark --width 1280 --height 720
```

`geometry_from_depth_state()` constructs a renderer-independent `GeometryState`
with `normals`, `normal_valid_mask`, `confidence`, `normal_confidence`, and
`selected_radius`. No temporal state, optical flow, lighting, shadows, neural
depth, or CUDA kernels are included in this phase.
