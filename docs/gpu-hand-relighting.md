# GPU Hand Relighting

Mode 7 runs a full-resolution OpenGL 3.3 surface pass and a reduced-resolution
volumetric pass over the latest P123 snapshot. It uses `fast_geometry_state`
when available and positions lights from `HandXYZ`; it never samples the palm
depth to infer a production light position. The CPU Youssef renderer remains
the explicit fallback and reference path.

## Install and Run

On Linux, install the UI/GL dependencies in the repository environment. The
CUDA-enabled PyTorch wheel must match the installed driver/runtime used by the
P123 depth provider.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu-ui.txt
python -m tools.p123_live_app --camera /dev/video0 --depth-backend local --depth-size 336x448 --mode 7 --lighting-quality balanced
```

For colleague/TensorRT depth deployments, install the matching optional
dependencies from `requirements-depth-engine.txt` and use the existing
provider/export instructions. Ensure the configured hand-landmarker asset is
present before starting `--hand-backend auto`.

Quality can be changed with `--lighting-quality low|balanced|high`. Press `D`
for pipeline and relight telemetry. The Mode 7 status strip reports renderer,
active lights, render duration, shadow/volume profile, and geometry/HandXYZ
source ages. If GLFW, PyOpenGL, or the GL context is unavailable, startup logs
the reason and reports `CPU_FALLBACK` rather than presenting the CPU image as
GPU-rendered.

## Presentation Paths

The P123 Material UI currently consumes NumPy frames, so Mode 7 reads the
final RGBA8 framebuffer back to CPU RGB before the existing OpenCV compositor.
The renderer also exposes `render_to_texture()` for a native-window path that
avoids this copy. Create `NativeOpenGLWindow` first, pass its GLFW window as
`share_window` to `GPURelightRenderer`, then pass the returned texture ID to
`NativeOpenGLWindow.render_texture()`. The shared GL context owns the texture;
close the renderer and window during shutdown.

The renderer is screen-space/camera-space ray marched, not hardware ray
tracing. Balanced settings use four deterministic area-light shadow rays with
six samples each, quarter-resolution volumetrics with six camera samples and
four sample-to-light visibility steps, and conservative depth-rejected
temporal history. No per-frame depth normalization is performed.
