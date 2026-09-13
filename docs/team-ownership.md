# Team Ownership & Subsystem Allocation

**Repository**: Rami-Troudi/Computer-Vision-NRW  
**Date**: 2026-09-13  
**Pass**: Strict Decluttering and Cleanup Pass  

---

## Subsystem Ownership Breakdown

### Person 1 (P1): Hand Tracking & 3D Gestures
- **Responsibilities**:
  - Hand landmark detection and tracking (21 3D landmarks)
  - Stable hand ID assignment across frames
  - One-Euro / EMA jitter filtering
  - Palm 2D UV detection and 3D camera-space $(X, Y, Z)$ position estimation
  - 3D hand velocity and open-palm width estimation
- **Primary Source Files**:
  - [`geometry/hand_control.py`](file:///home/rami/Computer-Vision-NRW/geometry/hand_control.py) (`HandControlEngine`, `TrackedHand`, `GestureState`)
  - [`geometry/depth_sampling.py`](file:///home/rami/Computer-Vision-NRW/geometry/depth_sampling.py) (`sample_depth`, `camera_uv_to_depth_uv` — shared with P3)
  - [`integrations/colleague_hand/`](file:///home/rami/Computer-Vision-NRW/integrations/colleague_hand) (Preserved upstream implementation from `talel-hand` / Talel)
  - [`models/hand_landmarker.task`](file:///home/rami/Computer-Vision-NRW/models/hand_landmarker.task) (MediaPipe Tasks model asset)
- **Tests**:
  - [`tests/test_hand_lighting.py`](file:///home/rami/Computer-Vision-NRW/tests/test_hand_lighting.py)
  - Hand tests in [`tests/test_p123_remediation.py`](file:///home/rami/Computer-Vision-NRW/tests/test_p123_remediation.py)

---

### Person 2 (P2): Monocular Depth Estimation
- **Responsibilities**:
  - Monocular depth model evaluation, benchmarking, and selection
  - Depth Anything V2 local checkpoint execution (PyTorch CUDA FP16)
  - Colleague depth pipeline integration (`feature/depth`)
  - Depth normalization (median relative metric scale contract ~2.0 m)
  - Depth inference diagnostics (latency, VRAM, min/max depth)
- **Primary Source Files**:
  - [`geometry/depth_provider.py`](file:///home/rami/Computer-Vision-NRW/geometry/depth_provider.py) (`DepthAnythingProvider`, `DepthInferenceDiagnostics`)
  - [`geometry/colleague_depth.py`](file:///home/rami/Computer-Vision-NRW/geometry/colleague_depth.py) (`ColleagueDepthProvider` — adapter for Meriam's model)
  - [`integrations/colleague_depth/`](file:///home/rami/Computer-Vision-NRW/integrations/colleague_depth) (Preserved upstream package from `feature/depth` / Meriam)
  - [`models/depth-anything-v2-small/`](file:///home/rami/Computer-Vision-NRW/models/depth-anything-v2-small) (Local model weights & config)
- **Tests**:
  - Depth provider tests in [`tests/test_p123_remediation.py`](file:///home/rami/Computer-Vision-NRW/tests/test_p123_remediation.py)

---

### Person 3 (P3): Camera Geometry, Normals & Temporal Stabilization
- **Responsibilities**:
  - Camera pinhole intrinsics model, projection, and unprojection
  - Depth to camera-space $(X, Y, Z)$ 3D backprojection
  - Camera-facing surface normal estimation with discontinuity protection
  - CUDA GPU geometry acceleration (`TorchGeometryBackend`)
  - Optical flow (OpenCV DIS / Farneback)
  - Temporal geometry engine: depth-aware backward warping, history reprojection, confidence fusion
  - Camera calibration (checkerboard)
- **Primary Source Files**:
  - [`geometry/camera.py`](file:///home/rami/Computer-Vision-NRW/geometry/camera.py) (`CameraModel`)
  - [`geometry/backproject.py`](file:///home/rami/Computer-Vision-NRW/geometry/backproject.py) (`backproject_depth`, `DepthScaleMode`)
  - [`geometry/normals.py`](file:///home/rami/Computer-Vision-NRW/geometry/normals.py) (`estimate_normals`, `normals_to_rgb`, `geometry_from_depth_state`)
  - [`geometry/cuda_backend.py`](file:///home/rami/Computer-Vision-NRW/geometry/cuda_backend.py) (`TorchGeometryBackend`)
  - [`geometry/motion.py`](file:///home/rami/Computer-Vision-NRW/geometry/motion.py) (`MotionState`, `OpenCVFlowProvider`)
  - [`geometry/alignment.py`](file:///home/rami/Computer-Vision-NRW/geometry/alignment.py) (`align_history_depth`, `align_inverse_depth`)
  - [`geometry/warp.py`](file:///home/rami/Computer-Vision-NRW/geometry/warp.py) (`warp_depth_backward`, `warp_field_backward`)
  - [`geometry/temporal.py`](file:///home/rami/Computer-Vision-NRW/geometry/temporal.py) (`TemporalGeometryEngine`, `TemporalConfig`)
  - [`geometry/calibration.py`](file:///home/rami/Computer-Vision-NRW/geometry/calibration.py) (Checkerboard calibration)
  - [`geometry/transforms.py`](file:///home/rami/Computer-Vision-NRW/geometry/transforms.py) (Letterbox & image scaling)
- **Experimental P3 Modules (Phase 5 / Infinity — Out of Critical Path)**:
  - [`geometry/pose.py`](file:///home/rami/Computer-Vision-NRW/geometry/pose.py) (`PoseEstimator`, `CameraPoseState`, `PoseEstimateResult`)
  - [`geometry/persistent.py`](file:///home/rami/Computer-Vision-NRW/geometry/persistent.py) (`SurfelMap`, `PersistentGeometryMapper`)
  - [`geometry/persistent_worker.py`](file:///home/rami/Computer-Vision-NRW/geometry/persistent_worker.py) (`PersistentMapWorker`)
- **Tests**:
  - [`tests/test_camera.py`](file:///home/rami/Computer-Vision-NRW/tests/test_camera.py)
  - [`tests/test_backprojection.py`](file:///home/rami/Computer-Vision-NRW/tests/test_backprojection.py)
  - [`tests/test_normals.py`](file:///home/rami/Computer-Vision-NRW/tests/test_normals.py)
  - [`tests/test_cuda_backend.py`](file:///home/rami/Computer-Vision-NRW/tests/test_cuda_backend.py)
  - [`tests/test_motion.py`](file:///home/rami/Computer-Vision-NRW/tests/test_motion.py)
  - [`tests/test_temporal.py`](file:///home/rami/Computer-Vision-NRW/tests/test_temporal.py)
  - [`tests/test_warp.py`](file:///home/rami/Computer-Vision-NRW/tests/test_warp.py)
  - [`tests/test_alignment.py`](file:///home/rami/Computer-Vision-NRW/tests/test_alignment.py)

---

### Person 4 (P4): Lighting, Shading, Shadows & Volumetrics
- **Responsibilities**:
  - Diffuse Lambertian relighting
  - Blinn-Phong specular highlights and surface roughness
  - Dynamic screen-space ray-marched self-shadows
  - Multi-light rendering (two colored lights)
  - Volumetric light shafts and atmospheric scattering
- **Primary Source Files**:
  - [`geometry/lighting.py`](file:///home/rami/Computer-Vision-NRW/geometry/lighting.py) (`LightState`, `shade_geometry`, `render_volumetric_scattering`, `light_from_palm`)
- **Status**: **BLACK BOX — DO NOT MODIFY IMPLEMENTATION**. Protected under the strict project scope guard.
- **Tests**:
  - [`tests/test_phase4_contracts.py`](file:///home/rami/Computer-Vision-NRW/tests/test_phase4_contracts.py)
  - Shading unit tests in [`tests/test_gpu_lighting.py`](file:///home/rami/Computer-Vision-NRW/tests/test_gpu_lighting.py)

---

### Shared Infrastructure, Contracts & Runtimes
- **Responsibilities**:
  - Intersystem contract definition and handoff verification
  - Thread orchestration and non-blocking frame buffers
  - Physical camera capture worker and V4L2 device handling
  - Native display UI (GLFW / PyOpenGL) and diagnostic viewers
  - Authoritative validation gates
- **Primary Source Files**:
  - [`geometry/state.py`](file:///home/rami/Computer-Vision-NRW/geometry/state.py) (`DepthState`, `GeometryState`)
  - [`geometry/p123_contract.py`](file:///home/rami/Computer-Vision-NRW/geometry/p123_contract.py) (`HandXYZ`, `P4InputState`)
  - [`geometry/camera_worker.py`](file:///home/rami/Computer-Vision-NRW/geometry/camera_worker.py) (`CameraCaptureWorker`, `LatestFrameSlot`)
  - [`geometry/async_pipeline.py`](file:///home/rami/Computer-Vision-NRW/geometry/async_pipeline.py) (`DepthWorker`, `LatestFrameBuffer`, `LatestDepthBuffer`)
  - [`geometry/p123_live_runtime.py`](file:///home/rami/Computer-Vision-NRW/geometry/p123_live_runtime.py) (`P123LiveRuntime`)
  - [`geometry/visualization.py`](file:///home/rami/Computer-Vision-NRW/geometry/visualization.py) (Diagnostic color mappers)
  - [`geometry/validation.py`](file:///home/rami/Computer-Vision-NRW/geometry/validation.py) (`validate_renderer_geometry`)
  - [`tools/physical_camera_gate.py`](file:///home/rami/Computer-Vision-NRW/tools/physical_camera_gate.py) (Hardware gate tool)
  - [`tools/p123_live_app.py`](file:///home/rami/Computer-Vision-NRW/tools/p123_live_app.py) (single live UI)
  - [`main.py`](file:///home/rami/Computer-Vision-NRW/main.py) (Application launcher)
- **Tests**:
  - [`tests/test_p123_remediation.py`](file:///home/rami/Computer-Vision-NRW/tests/test_p123_remediation.py)
  - [`tests/test_state.py`](file:///home/rami/Computer-Vision-NRW/tests/test_state.py)

---

## Uncertain / Legacy Files
