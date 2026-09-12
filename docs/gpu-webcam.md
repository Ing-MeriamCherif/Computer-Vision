# CUDA and webcam interface

The optional UI environment is isolated in `.venv` on the SATA checkout. It
does not change the Phase 1–4 CPU fallback or make PyTorch a base dependency.

```bash
./tools/setup_gpu_ui.sh
.venv/bin/python -m tools.webcam_geometry_app
```

The direct OpenCV window exposes:

- direct CUDA depth backprojection and camera-facing normals;
- the stable Phase 1–4 temporal path;
- optional Phase 5 PnP plus bounded persistent surfels;
- depth, normals, renderer confidence, persistent reprojection, and detailed
  stage/device diagnostics;
- colleague-style live hand vector overlays, filter/cadence/LK controls, and
  measured FPS/latency on every camera frame.

The processing path is direct V4L2/OpenCV (no Gradio or FastRTC transport).
The first frame includes lazy model startup; use the warm overlay values for
acceptance.

The setup downloads `depth-anything/Depth-Anything-V2-Small-hf` into the
ignored `models/depth-anything-v2-small` directory so later runs are offline.
The model supplies session-relative monocular depth; values are not metres.

## Verification

```bash
.venv/bin/python -m tools.cuda_smoke
.venv/bin/python -m tools.webcam_e2e --device /dev/video0
.venv/bin/python main.py --device /dev/video0
```

Validated host snapshot (2026-09-12): GTX 1650 Ti 4 GB, driver 595.84,
PyTorch 2.14.0+cu130, compute capability 7.5, OpenCV 5.0.0, and an HP Wide
Vision HD camera at `/dev/video0`. CUDA geometry and model
inference fall back explicitly or fail closed when unavailable; Phase 1–4
remains usable with the base `requirements.txt` environment.
