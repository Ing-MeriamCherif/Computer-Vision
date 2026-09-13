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

## P1/P2/P3 Asynchronous Live Diagnostics

This diagnostic UI is the live P123 path. It uses a physical camera, separate
latest-only depth/geometry/hand workers, and stops at `P4InputState`; it does
not render lighting, shadows, volumetrics, or any other Person 4 output:

```bash
PYTHONPATH=. .venv/bin/python -m tools.p123_live_app --camera /dev/video0
```

Use keys `1`–`6` for RGB, depth, normals, temporal confidence, hands, and XYZ
contract diagnostics; press `q` to exit. The P123 live path now uses Mariem's
vendored CUDA Depth Anything module exclusively, with Talel's hand tracker and
palm-size XYZ convention. FP32 is the default because it is faster than FP16
on the GTX 1650 Ti. The default Mariem input is 336px (use `--depth-size 420`
for higher spatial quality); use `--headless --duration 10` for a bounded
physical-camera smoke measurement. The older local/colleague depth providers
remain only for legacy non-P123 tools and are not selectable by this live app.
Native normals use the CUDA geometry stream
independently of the slower temporal CPU diagnostics; mode 4 uses a lightweight
native temporal-consistency map at the same live cadence.
XYZ mode samples the same fresh CUDA geometry state and reports “Waiting for a
detected hand...” when MediaPipe has no hand in view. Native depth and XYZ are
EMA-smoothed, and mode 6 displays the camera coordinate axes (+X right, +Y
down, +Z forward) in the upper-left corner beside each hand annotation. Depth,
normals, temporal confidence, hand positions, and XYZ all use the postprocessed
live states rather than raw worker buffers.

XYZ follows Talel's hand-branch back-projection convention: palm size provides
a bounded metric Z estimate (0.2–3.0 m) for relative-depth runs, then the
calibrated camera model computes `+X right, +Y down, +Z forward`. Metric depth
maps are used directly only when explicitly supplied.

For the GTX 1650 Ti, the measured physical-camera sweet spot is FP32 at
`--depth-size 336x448` (rectangular HxW syntax). The default runtime keeps the
slower full CPU temporal reference disabled; enable it explicitly with
`--full-temporal`. The HUD shows CAM/DEPTH/NORMALS/TEMP/HANDS rates and XYZ
fresh/degraded age. V4L2 format negotiation prefers advertised MJPG and uses a
capacity-one capture buffer; override with `--fourcc MJPG` or `--fourcc YUYV`
when required by a specific camera.

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
