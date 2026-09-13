# Known Issues & Technical Debt

**Repository**: Rami-Troudi/Computer-Vision-NRW  
**Date**: 2026-09-13  
**Pass**: Strict Decluttering and Cleanup Pass  

This ledger records active competition-relevant technical debt preserved for subsequent fix and tuning phases.

---

## 1. Depth Estimation (P2)
- **Palette & Dynamic Range**: Relative depth normalization maps median depth to ~2.0 m, but extreme near-camera objects (<0.3 m) can saturate the upper range of color mapping.
- **Edge Thinning & Depth Bleed**: Monocular depth maps from Depth Anything V2 exhibit minor boundary halo around foreground hand silhouettes against distant backgrounds.
- **Model Benchmark Pending**: Quantitative comparison between local `depth-anything-v2-small` and Meriam's colleague checkpoint on standard competition test images has not been frozen.
- **Temporal Depth Noise**: Independent per-frame depth inferences show low-frequency flicker on untextured surfaces before temporal stabilization.

---

## 2. Surface Normals (P3)
- **Discontinuity Edge Artifacts**: Finite difference normals across large depth discontinuities can create single-pixel spike artifacts when edge filtering thresholds are set too conservatively.
- **Visual Smoothness vs. Detail**: Baseline cross-product normals capture fine creases but can appear grainy under low-light camera sensor noise; multi-scale averaging improves smoothness at a slight compute cost.
- **Temporal Jitter**: Surface normal orientations on planar surfaces fluctuate slightly frame-to-frame if raw un-stabilized depth is used directly.

---

## 3. Temporal Stabilization (P3)
- **Capture Sequence vs. Processing Frame IDs**: When the camera thread runs at 30 FPS and the depth worker runs at ~18 FPS, optical flow must properly pair consecutive processed frames while referencing the exact capture sequence ID of the newest available RGB frame.
- **Fast Motion Disocclusion**: Rapid hand or camera pans cause large disocclusion regions where historical depth reprojection must be rejected and reset quickly to avoid ghosting trails.

---

## 4. Hand Tracking & Detection (P1)
- **Physical Multi-Hand Validation**: While the architecture supports 2 simultaneous hands, physical verification of tracking stability when hands cross or occlude one another is ongoing.
- **Extreme Boundary Tracking**: MediaPipe confidence drops when a hand enters or exits the camera periphery, requiring smooth coasting via the One-Euro filter to prevent light popping.

---

## 5. Hand-Depth Fusion & Hand XYZ (P1/P3)
- **Palm Sampling at Creases**: Sampling depth at the 2D palm center can occasionally hit an invalid depth pixel or self-occluded finger edge; the neighborhood median fallback handles this, but radius tuning affects response time.
- **Physical Scale Metric Accuracy**: Hand $(X, Y, Z)$ positions are reconstructed in camera space using the relative depth scale. Absolute metric accuracy is uncalibrated unless metric camera calibration is supplied.

---

## 6. Lighting, Shading & Shadows Observations (P4)
*(Observations only; P4 implementation is a black box and untouched)*
- **Diffuse Lighting Uniformity**: At oblique angles, Lambertian shading can appear overly dark in areas with high surface normal divergence.
- **Specular Highlight Sharpness**: Specular highlights on skin surfaces appear slightly metallic when shininess exponent is high.
- **Screen-Space Shadow Discontinuities**: Ray-marched screen-space shadows cannot trace occluders that lie off-screen or behind other surfaces, leading to visible shadow cutoff at frame boundaries during rapid hand movement.
- **Volumetric Light Shaft Aliasing**: Low step counts in volumetric ray-marching cause faint banding in the atmospheric haze beam.
