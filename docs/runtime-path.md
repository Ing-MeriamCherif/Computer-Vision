# Runtime Path

The repository has one user-facing interface: the P123 Material 3 live app.

```bash
.venv/bin/python main.py --camera /dev/video2 --mode 7
# or
.venv/bin/python -m tools.p123_live_app --camera /dev/video2 --mode 7
```

The app captures a physical camera through `CameraCaptureWorker`, runs the
selected depth provider and CUDA geometry pipeline, tracks hands, builds the
`P4InputState` contract, and renders the selected view through `p123/views/`.
Mode 7 uses the GPU relighting renderer with its explicit raster emergency
path; this is an internal renderer choice, not a second UI.

The strict physical validation CLI remains separate and has no display unless
`--interactive` is explicitly requested:

```bash
.venv/bin/python -m tools.physical_camera_gate --gate xyz --camera /dev/video2
```
