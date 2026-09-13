# Computer-Vision-NRW

Competition-ready live computer vision pipeline for the NRW challenge. It reads a physical webcam continuously and renders the newest available frame.

## Quick start

```bash
./tools/setup_gpu_ui.sh
.venv/bin/python main.py --camera 0
```

The setup script installs `requirements.txt` and downloads Depth Anything V2 Small. For Mode 7:

```bash
.venv/bin/python main.py --camera 0 --mode 7 --relight-backend auto --lighting-quality balanced
.venv/bin/python -m pytest -q
```

Keys `1`–`7` select RGB, depth, normals, temporal confidence, hands, XYZ, and relighting. `L` cycles lighting stages, `I` toggles the Level Infinity palm spotlight, `D` toggles telemetry, `F` fullscreen, and `Q`/`Esc` exits. Mode 7 reports `NVIDIA_OPTIX_RT_CORES`, `OPENGL_RASTER`, or `CPU_FALLBACK`; RTX hardware fails closed when real OptiX is unavailable.

## Architecture

```text
CameraCaptureWorker → LatestFrameSlot(capacity=1)
                    ├─ DepthWorker (Depth Anything V2, CUDA when available)
                    ├─ TorchGeometryBackend (XYZ + edge-aware normals)
                    ├─ HandControlEngine (MediaPipe + smoothing) → HandXYZ
                    └─ P123LiveRuntime → p123/views/* → OpenCV UI
                                             └─ Mode 7: OptiX / OpenGL / CPU
```

The asynchronous workers are latest-only so slow inference never blocks the live display. See [`docs/live-runtime-architecture.md`](docs/live-runtime-architecture.md) and [`docs/gpu-hand-relighting.md`](docs/gpu-hand-relighting.md).

RTX production additionally requires NVIDIA's Python OptiX binding, CuPy,
`cuda-python`/NVRTC, OptiX SDK headers, and `OPTIX_INCLUDE_DIR`. The real
production implementation is `geometry/optix_relighting.py` plus
`geometry/optix_shadow.cu`; the native C++ directory is not the jury path.

## Verification

```bash
.venv/bin/python -m tools.cuda_smoke
.venv/bin/python -m tools.physical_camera_gate --gate camera --camera 0
.venv/bin/python -m pytest -q
```

The suite currently covers camera, depth, geometry, temporal, hand, XYZ, UI, raster, and OptiX routing. Physical OptiX output requires an RTX machine; GTX systems use the labeled raster fallback.

## Layout

- `geometry/`: camera, depth, normals, temporal, hands, contracts, lighting.
- `p123/views/`: one display module per live mode.
- `tools/`: setup, launcher, gates, diagnostics, and native build helper.
- `native/optix_relight/`: optional CMake/OptiX extension.
- `integrations/`: preserved colleague adapters and provenance.
- `tests/`: deterministic coverage.
