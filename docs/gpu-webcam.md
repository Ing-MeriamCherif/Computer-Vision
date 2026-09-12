# CUDA and webcam interface

The optional UI environment is isolated in `.venv` on the SATA checkout. It
does not change the Phase 1–4 CPU fallback or make PyTorch a base dependency.

```bash
./tools/setup_gpu_ui.sh
.venv/bin/python -m tools.webcam_geometry_app
```

Open <http://127.0.0.1:7860>. The interface exposes:

- direct CUDA depth backprojection and camera-facing normals;
- the stable Phase 1–4 temporal path;
- optional Phase 5 PnP plus bounded persistent surfels;
- depth, normals, renderer confidence, persistent reprojection, and detailed
  stage/device diagnostics;
- live browser webcam streaming and deterministic upload testing.

For remote access, the browser capture is constrained to 256 x 192 at 30 FPS
before upload. The server requests the latest frame (`always_last`) so internet
latency cannot build a stale replay queue. The first frame includes lazy model
startup; use the warm FPS/latency values in the Live metrics card for acceptance.

The setup downloads `depth-anything/Depth-Anything-V2-Small-hf` into the
ignored `models/depth-anything-v2-small` directory so later runs are offline.
The model supplies session-relative monocular depth; values are not metres.

## Verification

```bash
.venv/bin/python -m tools.cuda_smoke
.venv/bin/python -m tools.webcam_e2e --device /dev/video0
.venv/bin/python /home/rami/.codex/skills/webapp-testing/scripts/with_server.py \
  --server ".venv/bin/python -m tools.webcam_geometry_app" --port 7860 --timeout 60 \
  -- .venv/bin/python -m tools.ui_e2e
```

Validated host snapshot (2026-09-12): GTX 1650 Ti 4 GB, driver 595.84,
PyTorch 2.14.0+cu130, compute capability 7.5, OpenCV 5.0.0, Gradio 6.27.0,
and an HP Wide Vision HD camera at `/dev/video0`. CUDA geometry and model
inference fall back explicitly or fail closed when unavailable; Phase 1–4
remains usable with the base `requirements.txt` environment.
