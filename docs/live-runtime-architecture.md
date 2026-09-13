# Live Runtime Architecture

**Repository**: Rami-Troudi/Computer-Vision-NRW  
**Revision**: Phase 5 Production Hardening — Native Live UI Integration  
**Date**: 2026-09-13

---

## Overview

The NRW competition application is a **native, live, low-latency, camera-first, GPU-first** computer-vision pipeline.
It runs continuously from a physical camera feed.
The production application is NOT based on uploaded images, manually-selected files, HTTP servers, or browser UIs.

---

## Architecture Diagram

```
Physical Camera (cv2.VideoCapture)
        │
        ▼  [CameraCaptureWorker – dedicated thread]
 LatestFrameSlot  ──────────────────────────────────────────────────────┐
  (capacity = 1)                                                         │
        │  full-res RGB                                                   │  hand tracking BGR
        ▼                                                                 ▼
 DepthWorker  (LatestFrameBuffer in)      HandTrackingWorker (dedicated thread)
 [Depth-Anything V2 on CUDA]              [MediaPipe Tasks backend]
        │  DepthState                              │  GestureState
        ▼                                          ▼
 TorchGeometryBackend.process_depth()     HandControlEngine
  (CUDA: normals, confidence)              (stable 2-hand IDs)
        │  GeometryState                           │
        ▼                                          ▼
        TemporalGeometryEngine (optional)     HandXYZ worker
  motion / temporal fusion                 palm-width Z + camera intrinsics
        │                                        │
        └────────────── latest geometry ─────────┤
                                                 ▼
                              P123 Mode 7: HandXYZ → LightState
                              OpenGL 3.3 surface/shadow/volume passes
                              CPU lighting is reference/fallback
                                                 │
                                                 ▼
                              OpenCV Material UI or shared GL texture path
        │
        ▼  [async, 2-10 Hz]
 PersistentMapWorker  (sidecar thread)
  SurfelMap → PersistentGeometryState
```

---

## Component Reference

### `geometry/camera_worker.py`

| Symbol | Purpose |
|--------|---------|
| `LatestFrameSlot` | Thread-safe single-slot buffer (capacity = 1). `put()` always overwrites. `get_latest()` is non-blocking. Exposes `dropped_count`, `total_arrived`. |
| `CameraCaptureWorker` | Dedicated thread wrapping `cv2.VideoCapture`. Writes full-res RGB to `LatestFrameSlot` with monotonic timestamps and monotonically-increasing sequence IDs. Exposes `actual_fps`, `dropped_frames`. |

### `geometry/native_window.py`

| Symbol | Purpose |
|--------|---------|
| `NativeOpenGLWindow` | GLFW + PyOpenGL double-buffered window. `render_frame(rgb_u8)` uploads texture and presents. Supports fullscreen toggle (`F`). `is_open` tracks state. |

Mode 7 ray marches soft area-light visibility and sample-to-light volumetric
occlusion through camera-space depth. It is a screen-space relighter, not RTX
hardware ray tracing. HandXYZ uses palm size and camera intrinsics, independent
of the depth worker.

### `geometry/native_app.py`

| Symbol | Purpose |
|--------|---------|
| `AppMode` | Enum: FEED=1, DEPTH=2, NORMALS=3, DIFFUSE=4, SPECULAR=5, SHADOWS=6, GESTURE=7, MULTILIGHT=8, INFINITY=9 |
| `QualityProfile` | LOW / BALANCED / HIGH — governs depth_size, shadow_steps, volumetric_steps, persistent_hz |
| `PROFILE_CONFIGS` | `dict[QualityProfile, QualityProfileConfig]` |
| `HandTrackingWorker` | Dedicated thread running `HandControlEngine`. Pushes `GestureState` to `LatestFrameSlot`. |
| `SyntheticCameraProvider` | Animated gradient camera for headless CI testing. |
| `NativeLiveApp` | Top-level application: starts all workers, non-blocking `step()`, keyboard handler, HUD overlay, 9 render modes. |

### `geometry/persistent_worker.py`

| Symbol | Purpose |
|--------|---------|
| `PersistentMapWorker` | Async sidecar thread updating `SurfelMap` at `map_update_hz`. Single pending job slot — excess submits are dropped (never queued). `get_latest_snapshot()` is non-blocking. |

### Key Phase 1–5 geometry modules (read-only stable API)

| Module | API |
|--------|-----|
| `geometry/backproject.py` | `backproject_depth()`, `DepthScaleMode`, cached ray grid |
| `geometry/camera.py` | `CameraModel` (frozen): `unproject()`, `project()`, `scaled_intrinsics()`, `load_json()` |
| `geometry/state.py` | `DepthState`, `GeometryState` |
| `geometry/cuda_backend.py` | `TorchGeometryBackend.process_depth()` → `GeometryState` with CUDA normals |
| `geometry/normals.py` | `estimate_normals()`, `normals_to_rgb()` |
| `geometry/lighting.py` | `LightState`, `shade_geometry()`, `render_volumetric_scattering()`, `light_from_palm()` |
| `geometry/hand_control.py` | `HandControlEngine`, `TrackedHand`, `GestureState`, `transform_hand_uv()` |
| `geometry/motion.py` | `MotionState`, `OpticalFlowProvider` |
| `geometry/temporal.py` | `TemporalGeometryEngine` |
| `geometry/pose.py` | `PoseEstimator`, `CameraPoseState`, `compute_rigid_flow_residual()`, `classify_static_geometry()` |
| `geometry/persistent.py` | `SurfelMap`, `AdvancedGeometryEngine`, `PersistentGeometryState` |

---

## Thread Model

```
Main thread              Camera thread         Depth thread          Hand thread
────────────────         ─────────────         ─────────────         ─────────────
NativeLiveApp.run()      CameraCaptureWorker   DepthWorker           HandTrackingWorker
  └─ step() loop           cv2.VideoCapture      TorchGeometry         HandControlEngine
       non-blocking         LatestFrameSlot       LatestDepthBuffer     LatestFrameSlot
       get_latest()         dropped_count         last_inference_ms     tracking_fps

Persistent thread         GLFW/OpenGL thread
─────────────────         ─────────────────
PersistentMapWorker       NativeOpenGLWindow
  SurfelMap update          texture upload
  2-10 Hz                   glfwSwapBuffers
  single pending job        fullscreen toggle
```

All inter-thread communication uses **lock-protected single-slot buffers** (capacity = 1).
**No unbounded queues exist** anywhere in the production path.

---

## Frame ID Contract

Every frame gets a monotonically-increasing integer ID assigned by `CameraCaptureWorker`:

```
CameraCaptureWorker  →  LatestFrameSlot  (frame, seq_id, timestamp)
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
          DepthWorker                   HandTrackingWorker
          DepthState.source_frame_id    GestureState.source_frame_id
               │                                 │
               ▼                                 ▼
          GeometryState.source_frame_id  LightState.source_hand
               │
               ▼
          PoseEstimateResult.source_frame_id / target_frame_id
```

The same `source_frame_id` propagates through the full pipeline, enabling exact frame-pair matching.

---

## Depth Convention

- `DepthAnythingProvider.compute()` normalizes output to **median = 2.0 metres** (relative forward-Z depth)
- `DepthScaleMode.RELATIVE` is the canonical scale mode
- Depth is **NOT disparity** — it is already in the same direction as camera `+Z`
- `ColleagueDepthProvider` uses the same median-normalization pipeline and output contract

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `1` – `9` | Switch display mode (FEED → DEPTH → … → INFINITY) |
| `F` | Toggle fullscreen |
| `D` | Cycle HUD detail level |
| `P` | Cycle quality profile (LOW → BALANCED → HIGH) |
| `V` | Toggle volumetric scattering |
| `S` | Toggle shadows |
| `H` | Toggle hand skeleton overlay |
| `R` | Reset persistent surfel map |
| `C` | Reset camera pose |
| `Q` / `ESC` | Quit |

---

## Quality Profiles

| Profile | Depth size | Shadow steps | Volumetric steps | Persistent Hz |
|---------|-----------|-------------|-----------------|--------------|
| LOW | 256×192 | 4 | 4 | 2 |
| BALANCED | 384×288 | 8 | 8 | 5 |
| HIGH | 512×384 | 16 | 12 | 10 |

---

## Display Modes

| Mode | Key | Description |
|------|-----|-------------|
| FEED | 1 | Raw camera RGB |
| DEPTH | 2 | Depth map (inferno colormap) |
| NORMALS | 3 | Surface normals (RGB-encoded) |
| DIFFUSE | 4 | Lambertian diffuse shading |
| SPECULAR | 5 | Specular highlight shading |
| SHADOWS | 6 | Ray-marched soft shadow shading |
| GESTURE | 7 | Hand skeleton + gesture overlay on raw feed |
| MULTILIGHT | 8 | Multi-hand lighting (up to 2 lights: cyan + amber) |
| INFINITY | 9 | Persistent surfel map depth overlay (Phase 5) |

---

## Worker Health HUD

The HUD overlay (toggled with `D`) shows:
- Mode name, quality profile
- Render FPS, render latency
- Camera capture FPS, dropped frames
- Depth inference latency, depth age (seconds since last depth)
- Hand tracking Hz, hand count, stale state indicator
- GPU memory usage
- Surfel count (Mode 9 only)
- Temporal geometry age

---

## Smoke Test

```bash
python main.py --smoke
```

Runs 10 frames in headless + synthetic camera mode. Exit code `0` = pass.
Used by CI to verify the full pipeline without requiring physical hardware.

---

## Entry Points

```bash
# Full native live app (default mode 8 = MULTILIGHT)
python main.py

# Specify camera and quality
python main.py --camera 0 --quality high --fullscreen

# Headless smoke test (CI)
python main.py --smoke

# Live camera acceptance test (30 second soak)
python -m tools.live_camera_acceptance --duration 30 --json-report report.json

# Soak test with stricter thresholds
python -m tools.live_camera_acceptance --soak --duration 300 --min-render-fps 28
```
