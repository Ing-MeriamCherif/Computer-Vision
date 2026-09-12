# Phase 3: Temporal Geometry Hardening

Phase 3 implements short-term, image-space temporal reconstruction around Phase 2 spatial geometry. It provides temporal stabilization for monocular relative and metric depth without requiring known camera poses or persistent world-space geometry.

## 1. Inverse-Depth Alignment with Dimensionless Scale Normalization

When aligning relative-depth history to the current observation frame in inverse-depth space ($q = 1/z$):
$$q_{\text{current}} \approx a \cdot q_{\text{history}} + b$$

### Scale-Normalized Residuals
Direct inverse-depth residuals $r_i = q_{\text{cur}, i} - (a q_{\text{hist}, i} + b)$ depend on the arbitrary depth scale of the scene. To ensure scale-invariance across depths ranging from micrometers to kilometers ($k \in [0.001, 1000]$), residuals are normalized by the median magnitude of the target inverse depth:
$$s_q = \max\left(\text{median}(|q_{\text{current}}|),\, \epsilon\right)$$
$$\tilde{r}_i = \frac{q_{\text{cur}, i} - (a q_{\text{hist}, i} + b)}{s_q}$$

Robust weights in the Iteratively Reweighted Least Squares (IRLS) solver are computed from the dimensionless residual:
$$w_i = \frac{1}{\sqrt{1 + (\tilde{r}_i / \tau)^2}}$$
Initial parameters $(a_0, b_0)$ are computed using robust median statistics ($a_0 = \text{median}(q_{\text{cur}}) / \text{median}(q_{\text{hist}})$, $b_0 = \text{median}(q_{\text{cur}} - a_0 q_{\text{hist}})$), preventing moving-object outliers from distorting initial weights.

## 2. Affine Alignment Conditioning & Fallback

Degenerate scenes with low depth variation (such as frontoparallel flat walls or skyboxes) produce near-singular design matrices ($X^T W X$) where the affine offset $b$ becomes unidentifiable or poorly conditioned.

Alignment condition number $\kappa = \lambda_{\max} / \lambda_{\min}$ and relative variance $\text{Var}(q) / \text{Mean}(q)^2$ are checked:
- If $\kappa > 10^4$, relative variance $< 10^{-4}$, or the determinant is near zero:
- The solver automatically falls back to **robust scale-only alignment**:
  $$q_{\text{current}} \approx a \cdot q_{\text{history}}, \quad a = \frac{\text{median}(q_{\text{current}})}{\text{median}(q_{\text{history}})}$$
  with $b = 0$.
- `DepthAlignmentResult.model_used` records `"affine"`, `"scale_only"`, or `"none"`.

## 3. Optical Flow Confidence Single Source of Truth

Optical flow confidence is parameterized once at the provider/state level via `fb_sigma`:
- In `flow_consistency(forward, backward, sigma=fb_sigma)`:
  $$C_{\text{flow}} = \exp\left(-\left(\frac{\text{error}_{\text{FB}}}{\sigma_{\text{FB}}}\right)^2\right)$$
- Forward-backward error and confidence are stored directly in `MotionState`.
- `OpenCVFlowProvider(fb_sigma=...)` passes this parameter into `MotionState`.
- `TemporalGeometryEngine` reuses the warped flow confidence directly, eliminating redundant double-exponential attenuation.

## 4. Discontinuity-Aware Depth Warping (`warp_depth_backward`)

Standard bilinear interpolation across depth edges creates non-existent intermediate surfaces (phantom surfaces, e.g. blending foreground $z=1.0$ and background $z=3.0$ into fictitious $z=2.0$).

`warp_depth_backward` inspects the 4-tap bilinear footprint:
- When relative depth spread across valid samples exceeds `discontinuity_threshold` ($\Delta z / \min(z) > \tau_{\text{disc}}$, default 0.15):
  1. The output selects the dominant physical depth layer (nearest dominant neighbor sample) rather than blending across the discontinuity.
  2. The boundary pixel is marked with reduced confidence ($C_{\text{warp}} \in [0.35, 0.49]$).
- In continuous interior regions, smooth bilinear interpolation is preserved with full confidence ($C_{\text{warp}} \in [0.7, 1.0]$).

## 5. Distinct Mask Semantics

To prevent conflating distinct physical failure modes into a single ambiguous mask, Phase 3 defines explicit semantic masks:

| Mask | Semantic Definition | Conditions |
|---|---|---|
| `history_valid` | History reprojected and accepted for fusion | Valid flow, confidence $\ge \tau_{\text{min}}$, depth agreement, valid projection |
| `history_rejection_mask` | History attempted but rejected for any reason | `has_reprojected_history & ~history_valid` |
| `occlusion_mask` | Physical occlusion: new foreground object in front | Tested pixels where $z_{\text{cur}} - z_{\text{hist}} < -\tau_{\text{disagree}}$ |
| `disocclusion_mask` | Physical disocclusion: foreground moved away, revealing background | Tested pixels where $z_{\text{cur}} - z_{\text{hist}} > +\tau_{\text{disagree}}$ |

- Flow failures (high FB error) or photometric failures reject history (`history_rejection_mask = True`), but do **not** flag geometric occlusion.
- Geometric step differences unambiguously distinguish occlusion from disocclusion.

## 6. Temporal Age Semantics

- **Accepted Fresh Depth**: When fresh depth observation is valid and fused at pixel $p$, `temporal_age` resets to `0`.
- **History Propagation**: When fresh depth is absent (`depth_state=None`) or invalid, accepted history increments `temporal_age = temporal_age + 1`.
- **Stale Expiration**: Stale history is retired only when `temporal_age > max_history_age` without fresh depth observations. 100 consecutive frames with valid depth never expire history.

## 7. CPU Performance Benchmarks

Measured on host CPU (Intel UHD / x86_64 Linux) using `tools/temporal_geometry_demo.py`:

| Resolution | Frames | Update Mean (ms) | Update P95 (ms) | Jitter Reduction | History Acceptance | Model Used |
|---|---|---|---|---|---|---|
| $320 \times 180$ | 20 | 141.9 ms | 165.1 ms | 50.6% | 100.0% | scale_only |
| $640 \times 360$ | 10 | 559.7 ms | 628.0 ms | 60.5% | 100.0% | scale_only |
| $1280 \times 720$ | 5 | 2641.8 ms | 2857.2 ms | 51.9% | 99.95% | scale_only |

## 8. Explicit Non-Goals & Deferred Items

The following components are strictly out of scope for Phase 3 and deferred to subsequent phases or separate subsystems:
- **Phase 4 Multi-View / Tracking**: Global pose estimation, bundle adjustment, full camera trajectory SLAM.
- **Volumetric Reconstruction**: Persistent world-space TSDF grids, surfel maps, mesh reconstruction.
- **Learned Flow Providers**: Deep optical flow (RAFT, FlowNet), neural feature warping.
- **Renderer Code**: TAA/TSR resolve shaders, spatial-temporal denoisers (NRD), custom GPU rasterization.
