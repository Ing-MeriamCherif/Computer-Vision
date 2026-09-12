# Phase 5: Optional Persistent Geometry

Phase 5 is an additive, production-hardened, fail-safe world-memory track. `TemporalGeometryEngine`
continues to produce the authoritative Phase 1–4 `GeometryState`. The optional `AdvancedGeometryEngine`
provides an end-to-end automatic pipeline that estimates camera motion, classifies static scene
geometry, fuses consistent world-space surfels, and exposes a `PersistentGeometryState`.

---

## 1. End-to-End Architecture

```
DepthState + RGB / MotionState
              ↓
  TemporalGeometryEngine (authoritative Phase 1-4)
              ↓
        GeometryState (camera-space positions & normals)
              ↓
  extract_pose_correspondences (stratified grid, confidence gating)
              ↓
  PoseEstimator (OpenCV PnP-RANSAC → T_current_from_previous)
              ↓
  classify_static_geometry (rigid flow residual r = ||p_obs - p_pred||)
              ↓
  PersistentGeometryMapper (relative/metric scale, 27-cell neighbor fusion)
              ↓
          SurfelMap (bounded voxel buckets, bucket migration, batch eviction)
              ↓
        reproject (nearest Z-buffer, normal transformation, view decay)
              ↓
  PersistentGeometryState (projected depth, positions, normals, confidence)
```

---

## 2. Coordinate & Transform Conventions

Transforms follow the explicit `T_A_from_B` convention, mapping homogeneous points from frame B into frame A.

- World anchor: First valid camera frame defines the world origin (`T_world_from_camera = I`).
- Relative pose: `PoseEstimator` computes `T_current_from_previous`.
- World pose accumulation:
  ```
  T_world_from_current = T_world_from_previous @ inv(T_current_from_previous)
  ```
- Normal transformation: Reprojection transforms world normals using rotation only:
  ```
  N_camera = R_camera_from_world @ N_world
  ```
  Normals are strictly invariant to camera translation.

---

## 3. Spatially Stratified Correspondence Extraction

`extract_pose_correspondences()` extracts reliable 3D-2D pairs between consecutive frames:

- **Validity Gating**: Accepts only pixels where previous 3D point is finite with $Z > 0$, previous geometry confidence $\ge \text{min\_geometry\_confidence}$, forward optical flow is finite, and target pixel $p_{\text{curr}} = p_{\text{prev}} + F(p_{\text{prev}})$ lies strictly within camera image bounds.
- **Grid Stratification**: The image is partitioned into an $M \times N$ regular grid (default $16 \times 16$). The highest-quality point (ranked by $C = C_{\text{geom}} \times C_{\text{flow}}$) is selected per cell, ensuring uniform spatial coverage and well-conditioned PnP geometry.
- **Frame Contracts**: Enforces `motion.source_frame_id == previous_geometry.source_frame_id` and raises descriptive errors on mismatch.

---

## 4. Rigid-Motion Residual & Dynamic Rejection

To prevent moving objects from corrupting the persistent static map:

- **Residual Formula**:
  $$r = \|p_{\text{current\_observed}} - \text{project}(T_{\text{current\_from\_previous}} \cdot P_{\text{previous}})\|_2$$
- **Static Confidence**:
  $$C_{\text{static}} = \exp\left(-\frac{\max(0, r - \text{threshold})}{\tau}\right) \times C_{\text{spatial}}$$
- **Dynamic Suppression**: Surfel insertion filters points with $C_{\text{static}} \ge \text{min\_static\_confidence}$ (default 0.5).
- **Validation**: `compute_dynamic_contamination()` quantitatively measures that zero percent of persistent surfels originate from dynamic moving objects.

---

## 5. Relative-Depth Map Scale Normalization

- **Metric Mode**: Voxel size and merge distance operate directly in metric meters.
- **Relative Mode**: On first frame, `scene_scale` is derived once as the median high-confidence stabilized depth. The voxel size is frozen per session:
  $$\text{voxel\_size} = \text{scene\_scale} \times \text{relative\_voxel\_fraction}$$
- **Scale Invariance**: Validated across depth scales 0.1, 1.0, 10.0, 100.0 to produce invariant normalized surfel densities and fusion ratios.
- **Scale Contract**: Refuses mixing metric and relative frames within the same map session.

---

## 6. Surfel Indexing, Neighbor Matching & Bucket Migration

- **27-Cell Neighborhood Search**: Queries the current voxel and all 26 adjacent neighbor cells ($3 \times 3 \times 3$) to prevent boundary seam artifacts.
- **Deterministic Best Match**: Among candidates within `merge_distance` and normal cosine $\ge \text{normal\_merge\_cos}$, selects the candidate minimizing spatial distance, with normal alignment as tie-breaker:
  $$\text{score} = \text{dist} - 10^{-4} \cdot (\mathbf{n}_{\text{cand}} \cdot \mathbf{n}_{\text{obs}})$$
- **Observation-Weighted Fusion**: Position moves by $\alpha = \frac{w}{C \cdot k + w}$, preventing runaway surfel drift across planar surfaces.
- **Bucket Migration**: When a surfel's fused coordinate crosses a voxel boundary, its index is removed from the old bucket and registered into the new bucket.
- **Index Consistency**: `validate_index()` verifies bidirectional consistency between surfels and buckets.

---

## 7. Eviction & Memory Boundedness

- **Batch Capacity Eviction**: When map exceeds `max_surfels`, all surfels are scored via age-decayed confidence $C \cdot \exp(-\Delta t / \tau)$, the lowest-scoring excess surfels are removed in batch, and buckets are rebuilt in a single $O(N)$ pass.
- **Age Eviction**: `age_evict()` purges surfels unseen for longer than `max_age_frames`.
- **Memory Footprint**:
  - Packed NumPy array data: exactly 36 bytes per surfel (positions, normals, confidence, age, count).
  - 10k surfels: 0.36 MB packed / ~2.9 MB Python heap.
  - 50k surfels: 1.80 MB packed / ~15.7 MB Python heap.
  - 100k surfels: 3.60 MB packed / ~31.2 MB Python heap.

---

## 8. Fail-Safe Policy & Pose Chain Contracts

- **Tracking States**: `UNINITIALIZED`, `TRACKING`, `LOST`.
- **Fail-Safe Fallback**: If PnP, correspondence extraction, or mapping encounters failure, the base `GeometryState` is returned untouched, ensuring zero interruption to live downstream components.
- **No Fabricated Recovery**: If tracking is `LOST`, mapping pauses without guessing or inventing unanchored world coordinates.

---

## 9. Tools & Verification

- **Physics-Coherent Demonstration**:
  ```bash
  python3 -m tools.persistent_geometry_demo --frames 20 --ply work/surfel_map.ply
  ```
- **Performance & Memory Benchmark**:
  ```bash
  python3 -m tools.persistent_geometry_benchmark --quick
  ```

---

## 10. Known Limitations

- **Pose Drift**: Frame-to-frame PnP visual odometry accumulates drift over long trajectories without loop closure.
- **No Loop Closure / Global BA**: Map is intentionally a lightweight local visual memory, not a full graph-optimized SLAM system.
- **CPU Mapping Rate**: At 720p resolution, dense mapping stride 2 runs at ~0.06 Hz on CPU; recommended live stride is 4 or 8 (subsampled or keyframe updates).
