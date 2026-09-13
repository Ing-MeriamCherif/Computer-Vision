# Team Integration Guide

**Repository**: Rami-Troudi/Computer-Vision-NRW  
**Branch policy**: all work on feature branches; merge to `main` only after tests pass  
**Reference PR**: https://github.com/Ing-MeriamCherif/Computer-Vision/pull/2

---

## Overview

This repository now integrates the work of two team members into a unified production pipeline.
The integration is achieved via **provenance-tracking wrappers** that preserve each contributor's original code
while making both interoperable with the shared geometry stack.

---

## Merged Components

### Rami's Geometry Stack (Phase 1–5)

| Phase | Modules | Status |
|-------|---------|--------|
| 1 | `geometry/backproject.py`, `geometry/camera.py`, `geometry/state.py` | ✅ Frozen |
| 2 | `geometry/normals.py`, `geometry/cuda_backend.py` | ✅ Frozen |
| 3 | `geometry/motion.py`, `geometry/temporal.py` | ✅ Frozen |
| 4 | `geometry/async_pipeline.py`, `geometry/pose.py`, `geometry/lighting.py`, `geometry/hand_control.py` | ✅ Frozen |
| 5 | `geometry/persistent.py`, `geometry/camera_worker.py`, `geometry/native_window.py`, `geometry/native_app.py`, `geometry/persistent_worker.py` | ✅ Production-hardened |

### Mariem's Depth Pipeline (Colleague PR #2)

| Component | Integration Point | Status |
|-----------|------------------|--------|
| `ColleagueDepthProvider` | `geometry/colleague_depth.py` | ✅ Integrated |
| Median depth normalization | `DepthAnythingProvider` (local) + `ColleagueDepthProvider` — same contract | ✅ Unified |
| `last_diagnostics` field | Added to `ColleagueDepthProvider` for HUD reporting | ✅ Fixed |

---

## Depth Provider Selection

Set environment variable `NRW_DEPTH_SOURCE` before running:

```bash
# Use Rami's local Depth Anything V2 (default)
NRW_DEPTH_SOURCE=local python main.py

# Use Mariem's colleague depth model
NRW_DEPTH_SOURCE=colleague python main.py

# CLI override
python main.py --depth-backend colleague
```

Both providers produce:
- `DepthState` with `scale_mode = DepthScaleMode.RELATIVE`
- Median depth normalized to ~2.0 metres
- Valid mask from finite depth checks
- `last_diagnostics` dict for HUD

---

## Adding a New Team Member's Code

1. **Create a wrapper module** in `geometry/` following the `ColleagueDepthProvider` pattern
2. **Implement the provider contract**:
   - `compute(rgb: np.ndarray, frame_id: int, timestamp: float) -> DepthState`
   - `last_diagnostics: dict[str, Any]`
3. **Add provider selection** to `geometry/depth_provider.py` or equivalent
4. **Write tests** in `tests/` following `test_native_live_integration.py` patterns
5. **Open a PR** to `main`; CI (smoke test) must pass

---

## Shared Contracts

### `DepthState`

```python
@dataclass(frozen=True, slots=True)
class DepthState:
    depth: np.ndarray           # (H, W) float32 forward-Z depth in metres (relative)
    timestamp: float            # monotonic seconds
    source_frame_id: int | str  # from CameraCaptureWorker
    scale_mode: str             # "relative"
    valid_mask: np.ndarray      # (H, W) bool
    confidence: np.ndarray | None = None  # (H, W) float32 in [0, 1]
```

### `GeometryState`

Produced by `TorchGeometryBackend.process_depth()` from a `DepthState`.
Carries: `depth`, `positions_3d` (H×W×3 camera-space), `normals` (H×W×3), `confidence` (H×W), `valid_mask`, `camera`.

### `LightState`

For P123 Mode 7, adapted from fresh `HandXYZ.xyz_camera` (palm-width metric-Z
proxy plus camera intrinsics), without sampling palm scene depth. The legacy
`light_from_palm(hand, geom)` remains available to the native CPU/reference path.
Carries: `position_camera` (3-vector metres), `intensity`, `color_rgb`, `confidence`, `source_hand`, `light_id`, and the separate range/source/orb radii.
The GPU renderer and CPU reference both attenuate from `position_camera` to the surface.

### `GestureState`

Produced by `HandControlEngine.update()`.
Carries: `hands: tuple[TrackedHand, ...]` (up to 2 hands), `timestamp`, `source_frame_id`, `backend`, `tracker_ms`.

---

## Running the Full Pipeline

```bash
# Install dependencies (PyTorch, MediaPipe, GLFW, OpenCV)
pip install -r requirements.txt

# Download model assets
# (Depth Anything V2 → models/depth-anything-v2-small)
# (MediaPipe Tasks → models/hand_landmarker.task)

# CI smoke test (headless, no camera required)
python main.py --smoke

# Full live demo
python main.py

# Run all unit + integration tests
python -m pytest

# Live camera acceptance (30-second soak)
python -m tools.live_camera_acceptance --duration 30
```

---

## Test Organisation

| File | Coverage |
|------|---------|
| `tests/test_native_live_integration.py` | LatestFrameSlot, workers, lighting, pose, persistent, quality profiles, full app smoke |
| `tests/test_phase5_persistent.py` | SurfelMap, PersistentGeometryMapper, AdvancedGeometryEngine |
| `tests/test_phase4_contracts.py` | LightState, hand-depth fusion contracts |
| `tests/test_phase4_hardening.py` | Static confidence, coordinate domains |
| `tests/test_motion.py` | OpticalFlowProvider, TemporalGeometryEngine |
| `tests/test_temporal.py` | Temporal fusion, photometric validation |
| `tests/test_normals.py` | All normal estimation modes |

---

## Known Limitations

- **Persistent geometry** (Mode 9) requires a valid camera pose estimate; falls back to identity if PnP fails
- **GLFW display** requires `DISPLAY` or `WAYLAND_DISPLAY` to be set; use `--headless` in CI
- **MediaPipe Tasks** backend requires `models/hand_landmarker.task`; falls back to legacy backend if absent
- **Colleague depth model** requires the weights to be downloaded separately; falls back to local model if absent

---

## Git Workflow

```
feature/my-feature  ─┐
                      ├─→ bug-fixes/... ─→ main (merge after tests pass)
mariem                ─┘
```

- All PRs must pass `python main.py --smoke` (exit 0)
- All PRs must pass `python -m pytest` (all tests green)
- No force pushes to `main`
- Branch deleted after merge
