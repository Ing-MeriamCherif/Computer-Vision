# Runtime Path & Critical Handoff Architecture

**Repository**: Rami-Troudi/Computer-Vision-NRW  
**Date**: 2026-09-13  
**Pass**: Strict Decluttering and Cleanup Pass  

---

## 1. Application Launch Hierarchy

### Default Competition Application
Executing:
```bash
python main.py
```
delegates directly to:
```python
from tools.native_live_app import main as _native_main
```
which constructs and runs [`NativeLiveApp`](file:///home/rami/Computer-Vision-NRW/geometry/native_app.py) with a native GLFW/OpenGL window on `:0`.

### Diagnostic & Validation Entry Points
- **P123 Physical Diagnostic UI**:
  ```bash
  python -m tools.p123_live_app
  ```
  Runs [`P123LiveRuntime`](file:///home/rami/Computer-Vision-NRW/geometry/p123_live_runtime.py) with an OpenCV diagnostic window. It terminates strictly at the P123 contract and contains zero P4 code.
- **Authoritative Physical Camera Gate**:
  ```bash
  python -m tools.physical_camera_gate --gate p123
  ```
  Strict fail-closed physical validation tool. Rejects all synthetic fallbacks.
- **Unit / Regression Suite**:
  ```bash
  pytest
  ```

---

## 2. End-to-End Critical Path Diagram

```
[ Physical V4L2 Camera ] (/dev/video0, 640x480 @ 30 FPS)
         │
         ▼
[ CameraCaptureWorker ] (geometry/camera_worker.py)
         │ Writes newest frame to LatestFrameSlot (capacity=1)
         │ Assigns capture_timestamp (time.monotonic()) & capture_sequence_id
         │
         ├───────────────────────────────┬───────────────────────────────┐
         ▼                               ▼                               ▼
[ P2: Depth Worker ]           [ P1: Hand Worker ]             [ P3: Camera Model ]
geometry/async_pipeline.py     geometry/hand_control.py        geometry/camera.py
DepthWorker                    HandTrackingWorker              CameraModel (intrinsics)
         │                              │                                │
         ├─ Local: Depth Anything V2    ├─ MediaPipe Tasks (.task)       │
         │  (FP16 CUDA)                 │  (21 3D landmarks)             │
         └─ Opt-in: Colleague Depth     └─ One-Euro filter               │
            (integrations/colleague)       (2-hand support)              │
         │                              │                                │
         ▼                              ▼                                │
   [ DepthState ]                [ GestureState ]                        │
   depth_map (relative)          hands: tuple[TrackedHand, ...]          │
   source_frame_id               palm_uv, depth_z, handedness            │
         │                              │                                │
         ├──────────────────────────────┴────────────────────────────────┤
         ▼
[ P3: Geometry Reconstruction & Surface Normals ]
geometry/cuda_backend.py (TorchGeometryBackend) or geometry/normals.py
- Unprojects (u, v, depth) to camera-space (X, Y, Z) positions via CameraModel
- Computes camera-facing surface normals with discontinuity protection
- Produces GeometryState (positions_3d, normals, valid_mask, confidence)
         │
         ▼
[ P3: Temporal Geometry Engine (Optional / Configurable) ]
geometry/temporal.py (TemporalGeometryEngine)
- Optical flow via OpenCVFlowProvider (DIS / Farneback)
- Forward/backward consistency verification
- Depth-aware history reprojection & confidence fusion
         │
         ▼
[ P1: Hand-Depth Fusion & Hand XYZ ]
geometry/depth_sampling.py (sample_depth) & geometry/hand_control.py
- Maps palm UV to depth space via camera_uv_to_depth_uv
- Bilinear / neighborhood sampling of depth buffer at palm location
- CameraModel.unproject(palm_u, palm_v, palm_z) -> HandXYZ.xyz_camera
         │
         ▼
========================================================================
[ AUTHORITATIVE P123 HANDOFF BOUNDARY: P4InputState ]
geometry/p123_contract.py
Contains ONLY:
  - rgb_frame (H, W, 3)
  - rgb_capture_id (monotonically increasing integer)
  - rgb_timestamp (float monotonic seconds)
  - camera_model (CameraModel)
  - geometry_state (GeometryState: XYZ positions, normals, valid mask)
  - hand_states (tuple[HandXYZ, ...]: palm UV, XYZ camera, confidence)
  - data_age_metrics (depth_age_ms, geometry_age_ms, hand_age_ms)
  - confidence_metadata
ZERO rendering, lighting, shadows, or volumetrics!
========================================================================
         │
         ▼
[ P4: Lighting, Shading & Renderer Boundary (Black Box) ]
geometry/lighting.py (shade_geometry, render_volumetric_scattering)
- Diffuse Lambertian shading (Level 2)
- Blinn-Phong specular highlights & roughness (Level 2)
- Hand-driven 3D gesture light positioning (Level 3)
- Dynamic screen-space ray-marched self-shadows (Level 4)
- Multi-hand dual colored lights & volumetric scattering (Level 5)
         │
         ▼
[ Display / Output ]
geometry/native_window.py (NativeOpenGLWindow - GLFW / OpenGL on :0)
or tools/p123_live_app.py (OpenCV imshow for diagnostics)
```

---

## 3. Subsystem Hand-Off Contracts

### Stage P2: Monocular Depth
- **Input**: RGB frame (`uint8`, $H \times W \times 3$), `source_frame_id`, `timestamp`.
- **Output**: [`DepthState`](file:///home/rami/Computer-Vision-NRW/geometry/state.py) with `depth` array (relative metric median-normalized to ~2.0 m), `scale_mode="relative"`, `valid_mask`, `confidence`.
- **Bypasses**: None in competition mode.

### Stage P3: Camera Geometry & Normals
- **Input**: `DepthState` + [`CameraModel`](file:///home/rami/Computer-Vision-NRW/geometry/camera.py).
- **Output**: [`GeometryState`](file:///home/rami/Computer-Vision-NRW/geometry/state.py) with `positions_3d` ($H \times W \times 3$, camera coordinates in metres), `normals` ($H \times W \times 3$, camera-facing unit vectors), `confidence`, `valid_mask`.
- **Bypasses**: If temporal smoothing is disabled, raw per-frame geometry passes straight through.

### Stage P1: Hand Tracking & Hand XYZ
- **Input**: RGB frame (`uint8`), `GeometryState` (for palm depth sampling), intrinsics [`CameraModel`](file:///home/rami/Computer-Vision-NRW/geometry/camera.py).
- **Output**: Tuple of [`HandXYZ`](file:///home/rami/Computer-Vision-NRW/geometry/p123_contract.py) dataclasses:
  - `hand_id: int` (stable integer ID, 0 or 1)
  - `palm_uv: tuple[float, float]` (2D normalized or pixel UV coordinates)
  - `xyz_camera: tuple[float, float, float] | None` (3D camera coordinates in metres)
  - `confidence: float`
  - `source_frame_id: int | str`
  - `age_ms: float`
  - `handedness: str | None` ("Left" or "Right")
- **Bypasses**: When no hands are visible, `hand_states` is an empty tuple `()`.

### Stage P123 → P4 Boundary: [`P4InputState`](file:///home/rami/Computer-Vision-NRW/geometry/p123_contract.py)
- **Role**: Clean isolation boundary. P1/P2/P3 have zero awareness of lighting, shaders, or rendering parameters.
- **Consumer**: [`shade_geometry()`](file:///home/rami/Computer-Vision-NRW/geometry/lighting.py) in P4 constructs lights from `hand_states` and applies diffuse, specular, shadows, and volumetrics.

---

## 4. Documented Bypasses & Optional Paths

1. **Experimental Phase 5 Persistent Mapping (Infinity Mode)**:
   - **Status**: OUT OF THE CRITICAL PATH.
   - **Isolation**: Handled by [`PersistentMapWorker`](file:///home/rami/Computer-Vision-NRW/geometry/persistent_worker.py) as an optional background sidecar. Disabled in P123 diagnostic tools (`p123_live_app.py`, `physical_camera_gate.py`). Can be disabled in `native_live_app.py` via `--no-persistent`.
2. **Synthetic Camera Fallback**:
   - **Status**: Permitted ONLY in unit/smoke testing (`main.py --smoke`, `tests/test_native_live_integration.py`).
   - **Strictly Banned**: In [`physical_camera_gate.py`](file:///home/rami/Computer-Vision-NRW/tools/physical_camera_gate.py), which enforces physical camera availability and fails closed.
3. **Colleague Depth Backend**:
   - **Status**: Preserved opt-in alternative via `NRW_DEPTH_SOURCE=colleague` or `--depth-backend colleague`. Default is the tested local `Depth Anything V2 Small` checkpoint.
