# Computer-Vision-NRW

National Robotics Week (NRW) AI & Vision Challenge — competition repository.

A **native, live, low-latency, camera-first, GPU-first** computer-vision system that operates continuously from a physical camera feed, performing real-time depth estimation, surface normal reconstruction, hand-driven relighting, and persistent 3D geometry mapping.

---

## Quick Start

```bash
# Headless smoke test (CI — no camera or display required)
python main.py --smoke

# Full native live application (default: multilight mode)
python main.py

# Specify camera device, quality, fullscreen
python main.py --camera 0 --quality high --fullscreen

# P123 live diagnostics (Mariem depth + Talel hand XYZ)
PYTHONPATH=. .venv/bin/python -m tools.p123_live_app --camera /dev/video0

# All options
python main.py --help
```

On an RTX GPU, the P123 launcher defaults to FP16, an aspect-preserving
`252x336` depth input, CUDA-resident depth handoff, and CUDA normals on every
second depth frame. The equivalent Windows command is:

```powershell
.\.venv\Scripts\python.exe -m tools.p123_live_app --camera 0
```

Launch directly into the hand-held relight view with a latency-focused profile:

```powershell
.\.venv\Scripts\python.exe -m tools.p123_live_app --camera 0 --mode 7 --display-fps 30 --depth-size 168x224 --normal-every-n 2
```

Use `--depth-size 336x448 --normal-every-n 1` for maximum spatial/normal
quality, or `--depth-size 210x280 --normal-every-n 3` when latency matters
most. Generate checkerboard calibration with `tools.calibrate_camera` and load
it with `--calibration camera.json`; calibrated frames are rectified before
inference. `--metric-depth` selects the metric Depth Anything checkpoint, and
`--exposure` plus `--camera-backend` expose camera cadence controls.

---

## Display Modes (keys 1–9)

| Key | Mode | Description |
|-----|------|-------------|
| `1` | FEED | Raw camera RGB |
| `2` | DEPTH | Depth map (inferno colormap) |
| `3` | NORMALS | Surface normals (RGB-encoded) |
| `4` | DIFFUSE | Lambertian diffuse shading |
| `5` | SPECULAR | Specular highlight shading |
| `6` | SHADOWS | Ray-marched soft shadow shading |
| `7` | GESTURE | Hand skeleton overlay |
| `8` | MULTILIGHT | Hand-driven dual lighting (default) |
| `9` | INFINITY | Persistent 3D surfel map overlay |

**Keyboard controls**: `F` fullscreen · `D` HUD · `P` profile · `V` volumetrics · `S` shadows · `H` skeleton · `R` reset map · `Q/ESC` quit

---

## Live Camera Acceptance Test

```bash
# 30-second soak with JSON report
python -m tools.live_camera_acceptance --duration 30 --json-report report.json

# Strict 5-minute soak
python -m tools.live_camera_acceptance --soak --duration 300 --min-render-fps 28
```

## P1/P2/P3 Physical Gates

The authoritative remediation gates use only a physical camera and never
substitute synthetic frames. They stop at the renderer-independent handoff
contract in [`geometry/p123_contract.py`](geometry/p123_contract.py):

```bash
PYTHONPATH=. .venv/bin/python -m tools.physical_camera_gate --gate camera --camera /dev/video0
PYTHONPATH=. .venv/bin/python -m tools.physical_camera_gate --gate depth --camera /dev/video0
PYTHONPATH=. .venv/bin/python -m tools.physical_camera_gate --gate geometry --camera /dev/video0
PYTHONPATH=. .venv/bin/python -m tools.physical_camera_gate --gate temporal --camera /dev/video0
PYTHONPATH=. .venv/bin/python -m tools.physical_camera_gate --gate hands --camera /dev/video0
PYTHONPATH=. .venv/bin/python -m tools.physical_camera_gate --gate xyz --camera /dev/video0
```

## P1/P2/P3 Asynchronous Live Diagnostics (Google Material 3 UI)

The P123 diagnostic interface features a modern Google Material 3 design system:
dark theme (`#101216` obsidian surface), anti-aliased rounded cards, clean typography
hierarchy, responsive sidebar navigation, compact telemetry header, and polished
connection/loading states:

```bash
PYTHONPATH=. .venv/bin/python -m tools.p123_live_app --camera /dev/video0
```

### Layout Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ [● P123]  M3 NORMALS      CAM 30 FPS  UI 30 FPS  DEPTH 14Hz  XYZ Fresh  CUDA│
├──────────────┬──────────────────────────────────────────────────────────────┤
│ VIEW MODES   │                                                              │
│ [1] RGB Cam  │  [● LIVE]  MODE 3 — SURFACE NORMALS                          │
│ [2] Depth    │                                                              │
│ [3] Normals* │                                                              │
│ [4] Temporal │               CENTRAL VIEWPORT (Letterboxed / Aspect-Fit)    │
│ [5] Hands    │                                                              │
│ [6] XYZ      │                                                              │
│ [7] Relight* │                                                              │
│              │                                                              │
│ [1-7] Mode   │                                                              │
│ [D] HUD      │                                                              │
│ [F] Fullscr  │  Normals: +X Right (Red) | +Y Down (Green) | +Z Forward (Blue)│
└──────────────┴──────────────────────────────────────────────────────────────┘
```

### Dedicated View Modules

Organized under [`p123/views/`](p123/views):
1. **RGB Camera** (`p123/views/rgb/`): Physical sensor feed (`/dev/video0`), live frame rate, and sensor metadata.
2. **Depth Map** (`p123/views/depth/`): Mariem CUDA Depth Anything visualization (warm=near, cool=far) with scale bar.
3. **Surface Normals** (`p123/views/normals/`): Multiscale surface normals (R=Nx, G=Ny, B=Nz) computed via CUDA backend.
4. **Temporal Confidence** (`p123/views/temporal/`): Scale-invariant temporal stability consistency map.
5. **Hand Tracking** (`p123/views/hands/`): Talel MediaPipe 21-point skeletal joints, palm tracking, confidence pills, and coasting state.
6. **XYZ Contract** (`p123/views/xyz/`): Camera-space metric 3D coordinates (m), 3D coordinate legend card, and freshness pill.
7. **Hand Relight** (`p123/views/relight/`): Cached low-resolution preview with a shared 3D light source and visible hand-held orb.

### Controls & Navigation

- **Mouse Navigation**: Click any item in the left sidebar to instantly switch views.
- **Keyboard Shortcuts**:
  - `1`–`6`: Direct view selection (RGB, Depth, Normals, Temporal, Hands, XYZ)
  - `D`: Toggle engine telemetry HUD overlay (capture, inference, and latency breakdown)
  - `F`: Toggle borderless fullscreen
  - `Q` or `ESC`: Clean shutdown

---

## Tests

```bash
python -m pytest                     # all 190+ tests
python -m pytest -q --tb=short       # compact output
python -m pytest tests/test_native_live_integration.py  # integration tests only
```

---

## Geometry Stack (Phases 1–5)

The `geometry/` package implements the complete calibrated geometry pipeline:

| Phase | Features |
|-------|---------|
| 1 | Calibrated camera model · depth → camera-space reconstruction · coordinate transforms |
| 2 | Camera-facing surface normals · edge-aware / multi-scale · discontinuity protection · spatial confidence |
| 3 | Optical flow (DIS/Farneback) · forward/backward consistency · depth-aware temporal warping · temporal fusion |
| 4 | Hand tracking (MediaPipe) · palm depth fusion · diffuse/specular/shadow shading · hand-driven lights |
| 5 | PnP-RANSAC pose estimation · bounded voxel surfel map · persistent geometry · async sidecar |

**Depth convention**: forward-Z, median normalized to ~2.0 m, `DepthScaleMode.RELATIVE`.  
**Camera convention**: `+X` image-right, `+Y` image-down, `+Z` forward.

---

## Architecture

```
Physical Camera ─→ LatestFrameSlot (cap=1) ─┬─→ DepthWorker (CUDA) ─→ GeometryState
                                             ├─→ HandTrackingWorker  ─→ GestureState ─→ LightState
                                             └─→ NativeOpenGLWindow  ←─ shade_geometry()
                                                        ↑
                                              PersistentMapWorker (async, 2-10 Hz)
```

See [`docs/live-runtime-architecture.md`](docs/live-runtime-architecture.md) for the full thread model.

---

## Team Integration

This repository integrates work from two team members:

- **Rami Troudi** — geometry stack (Phases 1–5), native live app, persistent mapping
- **Mariem Cherif** — colleague depth provider ([PR #2](https://github.com/Ing-MeriamCherif/Computer-Vision/pull/2))

See [`docs/team-integration.md`](docs/team-integration.md) for integration contracts and contribution guide.

---

## Documentation

| Doc | Contents |
|-----|---------|
| [`docs/repository-inventory.md`](docs/repository-inventory.md) | File classification (Critical, Shared, Diagnostic, Experimental, Legacy) |
| [`docs/runtime-path.md`](docs/runtime-path.md) | End-to-end critical runtime path & subsystem hand-off diagram |
| [`docs/team-ownership.md`](docs/team-ownership.md) | Subsystem ownership (P1, P2, P3, P4, Shared) & provenance |
| [`docs/known-issues.md`](docs/known-issues.md) | Active technical debt & known issues across all phases |
| [`docs/live-runtime-architecture.md`](docs/live-runtime-architecture.md) | Thread model, frame ID contract, component reference |
| [`docs/team-integration.md`](docs/team-integration.md) | Shared contracts, contribution workflow |
| [`docs/geometry-phase1.md`](docs/geometry-phase1.md) | Calibration, backprojection |
| [`docs/geometry-phase2.md`](docs/geometry-phase2.md) | Normal algorithms |
| [`docs/geometry-phase3.md`](docs/geometry-phase3.md) | Temporal fusion contracts |
| [`docs/geometry-phase5.md`](docs/geometry-phase5.md) | Persistent geometry, surfel map |
| [`docs/geometry-performance.md`](docs/geometry-performance.md) | Benchmark guidance |
| [`docs/gpu-hand-relighting.md`](docs/gpu-hand-relighting.md) | Mode 7 OpenGL install, run, profiles, and presentation paths |
