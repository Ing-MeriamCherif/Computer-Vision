# Computer-Vision-NRW

National Robotics Week (NRW) AI & Vision Challenge — competition repository.

A **native, live, low-latency, camera-first, GPU-first** computer-vision system that operates continuously from a physical camera feed, performing real-time depth estimation, surface normal reconstruction, hand-driven relighting, and persistent 3D geometry mapping.

---

## Quick Start

```bash
# Single live Material 3 interface
python main.py

# Start hand relighting (Mode 7)
python main.py --camera /dev/video2 --mode 7 --lighting-quality balanced

# P123 live diagnostics (depth + hand XYZ)
PYTHONPATH=. .venv/bin/python -m tools.p123_live_app --camera /dev/video0

# All options
python main.py --help
```

The single P123 live path defaults to local Depth Anything V2 at `336x448` FP32 with
latest-only depth handoff and CUDA normals. Mode 7 uses an OpenGL 3.3
screen-space/camera-space renderer when available and an explicit CPU reference
fallback otherwise. It is not hardware RTX ray tracing or full path tracing.

```powershell
.\.venv\Scripts\python.exe -m tools.p123_live_app --camera 0
```

Launch directly into the hand-held relight view:

```powershell
.\.venv\Scripts\python.exe -m tools.p123_live_app --webcam-camera 0 --phone-camera 2 --mode 7 --lighting-quality balanced
```

Use the header selector or `C` to switch WEBCAM/PHONE; device IDs are
configurable with `--webcam-camera` and `--phone-camera`. Lighting quality can
be selected with `--lighting-quality low|balanced|high`.

---

## P123 Display Modes (keys 1–7)

| Key | Mode | Description |
|-----|------|-------------|
| `1` | RGB | Raw camera RGB |
| `2` | DEPTH | Mariem CUDA depth map |
| `3` | NORMALS | CUDA surface normals |
| `4` | TEMPORAL | Temporal confidence |
| `5` | HANDS | Talel hand tracking |
| `6` | XYZ | Camera-space hand coordinates |
| `7` | RELIGHT | Hand-controlled GPU relighting |

**Keyboard controls**: `1–7` views · `F` fullscreen · `D` HUD · `Q/ESC` quit

---

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
7. **Hand Relight** (`p123/views/relight/`): Full-resolution GPU surface relighting, reduced-resolution volumetrics, and a projected 3D light orb; CPU reference fallback preserves camera detail.

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
python -m pytest tests/test_p123_modern_ui.py  # P123 UI integration tests
```

---

## Geometry Stack (Phases 1–5)

The `geometry/` package implements the complete calibrated geometry pipeline:

| Phase | Features |
|-------|---------|
| 1 | Calibrated camera model · depth → camera-space reconstruction · coordinate transforms |
| 2 | Camera-facing surface normals · edge-aware / multi-scale · discontinuity protection · spatial confidence |
| 3 | Optical flow (DIS/Farneback) · forward/backward consistency · depth-aware temporal warping · temporal fusion |
| 4 | Hand tracking (MediaPipe) · palm-width HandXYZ · GPU diffuse/additive-specular/shadow/volume relighting |
| 5 | PnP-RANSAC pose estimation · bounded voxel surfel map · persistent geometry · async sidecar |

**Depth convention**: forward-Z, median normalized to ~2.0 m, `DepthScaleMode.RELATIVE`.  
**Camera convention**: `+X` image-right, `+Y` image-down, `+Z` forward.

---

## Architecture

```
Physical Camera ─→ LatestFrameSlot (cap=1) ─┬─→ Depth/Geometry workers ─→ P123Snapshot
                                             ├─→ HandTrackingWorker ─→ HandXYZ
                                             └─→ P123 Material 3 OpenCV UI
                                                        ↑
                                              Mode 7 GPU relight renderer
```

See [`docs/live-runtime-architecture.md`](docs/live-runtime-architecture.md) for the full thread model.

---

## Team Integration

This repository integrates work from two team members:

- **Rami Troudi** — geometry stack (Phases 1–5), P123 live app, persistent mapping
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
