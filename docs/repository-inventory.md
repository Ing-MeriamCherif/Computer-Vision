# Repository Inventory

**Repository**: Rami-Troudi/Computer-Vision-NRW  
**Date**: 2026-09-13  
**Pass**: Strict Decluttering and Cleanup Pass  

---

## Classification Categories

- **A. COMPETITION CRITICAL**: Modules in the mandatory live runtime path (Camera → P2 Depth → P3 Geometry/Normals/Temporal → P1 Hand/XYZ → P123 Contract → P4 Relighting).
- **B. SHARED INFRASTRUCTURE**: Shared contracts, frame slots, workers, and environment setups used by multiple stages.
- **C. DIAGNOSTIC / VALIDATION**: Authoritative physical gates, diagnostic viewers, benchmarks, calibration, and validation tools.
- **D. EXPERIMENTAL / OPTIONAL**: Phase 5 persistent geometry (surfel map, pose estimation, infinity mode) and non-critical benchmarks. Kept out of the default P123 critical path.
- **E. LEGACY**: Superseded viewer or demo scripts preserved for backward compatibility and reference.
- **F. DUPLICATED**: Redundant exports, duplicate helpers, or shadowed imports identified for consolidation.
- **G. UNKNOWN OWNERSHIP — DO NOT TOUCH**: Unattributed code or external upstream artifacts requiring team verification.

---

## Detailed File Inventory

| File | Category | Owner | Purpose | Imported By | Runtime Critical? | Duplicate? | Safe to Remove? | Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `main.py` | A | Shared | Top-level entry point; launches native live competition app | User / CLI | YES | NO | NO | KEEP (authoritative competition entry point) |
| `geometry/__init__.py` | B | Shared | Public package API exports for `geometry` | External callers, tests, tools | YES | YES (duplicate `sample_depth` import/export) | NO | KEEP & CLEAN (consolidate duplicate exports) |
| `geometry/camera.py` | A | P3 | `CameraModel` (intrinsics, projection, unprojection, scaled intrinsics, JSON load/save) | `backproject.py`, `temporal.py`, `p123_live_runtime.py`, `physical_camera_gate.py`, etc. | YES | NO | NO | KEEP |
| `geometry/camera_worker.py` | B | Shared | `CameraCaptureWorker`, `LatestFrameSlot` (capacity=1, monotonic timestamps, dropped frame counting) | `native_app.py`, `p123_live_runtime.py`, `physical_camera_gate.py` | YES | NO | NO | KEEP |
| `geometry/depth_provider.py` | A | P2 | `DepthAnythingProvider` (Depth Anything V2 Small local inference, FP16 CUDA) | `native_app.py`, `p123_live_runtime.py`, `physical_camera_gate.py` | YES | NO | NO | KEEP (normalize interface: `backend_name`, `load`, `compute`, `last_diagnostics`) |
| `geometry/colleague_depth.py` | A | P2 | `ColleagueDepthProvider` (adapter for Meriam's `feature/depth` model under `integrations/`) | `native_app.py`, `physical_camera_gate.py` | YES (when selected) | NO | NO | KEEP (preserves colleague provenance) |
| `geometry/depth_sampling.py` | B | Shared (P1/P3) | Neutral `sample_depth` & `camera_uv_to_depth_uv` (renderer-independent bilinear/neighborhood sampling) | `hand_control.py`, `p123_live_runtime.py`, `__init__.py` | YES | NO | NO | KEEP (isolated neutral utility; frees P1 from P4 dependency) |
| `geometry/backproject.py` | A | P3 | `backproject_depth`, `DepthScaleMode` (depth → camera-space XYZ reconstruction) | `normals.py`, `cuda_backend.py`, `temporal.py` | YES | NO | NO | KEEP |
| `geometry/normals.py` | A | P3 | `estimate_normals`, `normals_to_rgb`, `geometry_from_depth_state` (discontinuity-aware normals) | `cuda_backend.py`, `native_app.py`, `physical_camera_gate.py` | YES | NO | NO | KEEP |
| `geometry/cuda_backend.py` | A | P3 | `TorchGeometryBackend` (CUDA tensor backprojection & normal estimation) | `native_app.py`, `physical_camera_gate.py`, tests | YES | NO | NO | KEEP |
| `geometry/motion.py` | A | P3 | `MotionState`, `OpticalFlowProvider`, `OpenCVFlowProvider` (DIS/Farneback forward/backward flow) | `temporal.py`, `p123_live_runtime.py`, `native_app.py` | YES | NO | NO | KEEP |
| `geometry/alignment.py` | A | P3 | `align_history_depth`, `align_inverse_depth` (scale/shift relative depth alignment) | `temporal.py` | YES | NO | NO | KEEP |
| `geometry/warp.py` | A | P3 | `warp_depth_backward`, `warp_field_backward`, `warp_normals_backward` | `temporal.py`, `motion.py` | YES | NO | NO | KEEP |
| `geometry/temporal.py` | A | P3 | `TemporalGeometryEngine` (history reprojection, photometric verification, confidence fusion) | `p123_live_runtime.py`, `physical_camera_gate.py`, tests | YES | NO | NO | KEEP |
| `geometry/hand_control.py` | A | P1 | `HandControlEngine`, `TrackedHand`, `GestureState` (MediaPipe 21-landmark tracking, One-Euro filter, 2-hand support) | `native_app.py`, `p123_live_runtime.py`, `physical_camera_gate.py` | YES | NO | NO | KEEP |
| `geometry/async_pipeline.py` | B | Shared | `DepthWorker`, `LatestFrameBuffer`, `LatestDepthBuffer` (decoupled inference thread) | `native_app.py`, `p123_live_runtime.py` | YES | NO | NO | KEEP |
| `geometry/p123_contract.py` | B | Shared | Authoritative P123 handoff contract: `HandXYZ`, `P4InputState` | `p123_live_runtime.py`, `physical_camera_gate.py` | YES | NO | NO | KEEP (authoritative contract) |
| `geometry/p123_live_runtime.py` | C | Shared | `P123LiveRuntime`, `P123Snapshot`, `P123Metrics` (asynchronous physical-camera diagnostic runtime) | `tools/p123_live_app.py` | NO (diagnostic only) | NO | NO | KEEP (isolated from P4) |
| `geometry/visualization.py` | C | Shared | `depth_to_rgb`, `normals_to_rgb_diagnostic`, `confidence_to_rgb`, `add_depth_legend` | `tools/p123_live_app.py`, `physical_camera_gate.py` | NO (diagnostic only) | NO | NO | KEEP (pure visualization, no math modification) |
| `geometry/lighting.py` | A | P4 | `LightState`, `shade_geometry`, `render_volumetric_scattering`, `light_from_palm` | `native_app.py` | YES | NO | NO | **KEEP UNTOUCHED** (Absolute P4 rule: black box) |
| `geometry/native_app.py` | A | Shared | `NativeLiveApp`, `AppMode` (1–9), `QualityProfile` (GLFW/OpenGL competition runtime) | `main.py`, `tools/native_live_app.py` | YES | NO | NO | KEEP (isolate Mode 9 persistent worker via flag) |
| `geometry/native_window.py` | B | Shared | `NativeOpenGLWindow` (GLFW + PyOpenGL texture streaming) | `native_app.py` | YES | NO | NO | KEEP |
| `geometry/persistent.py` | D | P3 / Shared | `SurfelMap`, `PersistentGeometryMapper`, `AdvancedGeometryEngine` (Phase 5 persistent mapping) | `persistent_worker.py`, `native_app.py` | NO (experimental) | NO | NO | KEEP (isolate from default P123 path) |
| `geometry/persistent_worker.py` | D | Shared | `PersistentMapWorker` (async sidecar thread for Phase 5 surfel mapping) | `native_app.py` | NO (experimental) | NO | NO | KEEP (isolate from default P123 path) |
| `geometry/pose.py` | D | P3 | `PoseEstimator`, `CameraPoseState`, `PoseEstimateResult` (PnP-RANSAC relative camera pose) | `persistent.py`, `native_app.py` | NO (experimental) | NO | NO | KEEP (isolate from default P123 path) |
| `geometry/calibration.py` | C | P3 | OpenCV checkerboard calibration routines | `tools/calibrate_camera.py`, tests | NO (calibration only) | NO | NO | KEEP |
| `geometry/diagnostics.py` | C | Shared | Timing and array byte-size diagnostics | `temporal.py`, tests | NO (diagnostic only) | NO | NO | KEEP |
| `geometry/debug.py` | C | Shared | Debug utilities and visualization hooks | Tests / optional debug | NO (debug only) | NO | NO | KEEP |
| `geometry/state.py` | B | Shared | Dataclasses `DepthState`, `GeometryState` | Throughout `geometry/` | YES | NO | NO | KEEP |
| `geometry/transforms.py` | B | P3 | Explicit coordinate transformations (resize, crop, letterbox) | `camera.py`, tests | YES | NO | NO | KEEP |
| `geometry/validation.py` | C | P3 | `validate_renderer_geometry`, `GeometryValidationReport`, synthetic shapes | Tests, tools | NO (validation only) | NO | NO | KEEP |
| `tools/native_live_app.py` | A | Shared | CLI wrapper for `NativeLiveApp` competition UI | `main.py`, CLI | YES | NO | NO | KEEP |
| `tools/p123_live_app.py` | C | Shared | Real-time OpenCV viewer for P123 diagnostics (Modes 1–6) | CLI | NO (diagnostic only) | NO | NO | KEEP |
| `tools/physical_camera_gate.py` | C | Shared | Authoritative physical camera validation gate (fail-closed, no synthetic fallback) | CI / Hardware testing | NO (gate tool) | NO | NO | KEEP |
| `tools/live_camera_acceptance.py` | C | Shared | Continuous soak test with FPS/latency/RSS metrics and optional `--synthetic` mode | CLI / CI | NO (soak test) | NO | NO | KEEP (document distinction from hardware gate) |
| `tools/webcam_geometry_app.py` | E | Shared | Legacy 4-panel OpenCV browser/window app | `webcam_e2e.py`, legacy tests | NO (legacy) | YES (superseded by `native_live_app` & `p123_live_app`) | NO | KEEP / ISOLATE (retain for legacy compatibility) |
| `tools/webcam_e2e.py` | E | Shared | Legacy single-frame snapshot capture over `WebcamGeometrySession` | CLI | NO (legacy) | NO | NO | KEEP / ISOLATE (legacy snapshot test) |
| `tools/calibrate_camera.py` | C | P3 | Interactive checkerboard camera calibration CLI | CLI | NO | NO | NO | KEEP |
| `tools/cuda_smoke.py` | C | P3 | Quick CUDA backprojection / normal verification smoke script | CLI | NO | NO | NO | KEEP |
| `tools/geometry_demo.py` | C | P3 | Synthetic geometry demo (tilted plane, sphere) used by `--smoke` | CLI | NO | NO | NO | KEEP |
| `tools/geometry_benchmark.py` | C | P3 | Benchmarking backprojection and normals | CLI | NO | NO | NO | KEEP |
| `tools/geometry_stress.py` | C | P3 | Multi-frame stress test for memory and compute stability | CLI | NO | NO | NO | KEEP |
| `tools/geometry_capabilities.py` | C | Shared | Hardware capabilities probe (CUDA, OpenCV, PyTorch) | CLI | NO | NO | NO | KEEP |
| `tools/temporal_geometry_demo.py` | C | P3 | Synthetic temporal reprojection demo | CLI | NO | NO | NO | KEEP |
| `tools/flow_resolution_benchmark.py` | D | P3 | Benchmark optical flow compute at various resolutions | CLI | NO | NO | NO | KEEP |
| `tools/persistent_geometry_demo.py` | D | P3 | Demo for Phase 5 persistent surfel map | CLI | NO | NO | NO | KEEP |
| `tools/persistent_geometry_benchmark.py` | D | P3 | Benchmark Phase 5 pose and surfel update latency | CLI | NO | NO | NO | KEEP |
| `tools/setup_gpu_ui.sh` | B | Shared | Shell script to provision model checkpoints and verify GPU setup | CLI | NO | NO | NO | KEEP |
| `integrations/colleague_depth/` | G | P2 (Meriam) | Preserved source tree from `feature/depth` (`f8ecd09d`) | `geometry/colleague_depth.py` | YES (optional backend) | NO | NO | **KEEP UNTOUCHED** (preserved team provenance) |
| `integrations/colleague_hand/` | G | P1 (Talel) | Preserved source tree from `talel-hand` (`f6e65831`) | `geometry/hand_control.py` | YES (optional backend) | NO | NO | **KEEP UNTOUCHED** (preserved team provenance) |
