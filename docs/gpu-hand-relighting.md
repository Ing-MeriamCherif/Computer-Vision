# GPU Hand Relighting

Mode 7 is an OpenGL 3.3 GPU screen-space/camera-space ray-marched relighter over
monocular 2.5D geometry. It uses `fast_geometry_state` when available, falling
back to `geometry_state`, and positions lights from fresh `HandXYZ.xyz_camera`;
it never samples palm depth to infer a production light position. RGB is
converted from sRGB to linear before surface lighting, then converted back at
the final composite. Diffuse is albedo-weighted; specular is additive. The CPU
Youssef renderer remains the explicit fallback and reference path.

The GPU path combines full-resolution surface lighting, up to two independently
colored lights, camera-space soft area-light shadow rays, reduced-resolution
volumetric ray marching with sample-to-light visibility, temporal stabilization,
and projected 3D emitter orbs. Each orb and its lighting/shadow/volume rays use
the same camera-space `LightState` position. The working emitter range is
`0.30 m`; physical shadow source radius is `0.018 m`; visible orb radius is
`0.035 m`. These values are separate. This is not hardware RTX ray tracing or
full path tracing, and it does not reconstruct hidden geometry beyond the
monocular depth surface.

## Install and Run

On Linux, install the UI/GL dependencies in the repository environment. The
CUDA-enabled PyTorch wheel must match the installed driver/runtime used by the
P123 depth provider.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu-ui.txt
python -m tools.p123_live_app --webcam-camera /dev/video0 --phone-camera /dev/video2 --depth-backend local --depth-size 336x448 --mode 7 --lighting-quality balanced
```

For colleague/TensorRT depth deployments, install the matching optional
dependencies from `requirements-depth-engine.txt` and use the existing
provider/export instructions. Ensure the configured hand-landmarker asset is
present before starting `--hand-backend auto`.

Quality can be selected with `--lighting-quality low|balanced|high`. The
volume target is reallocated when its profile divisor changes. Press `D`
for pipeline and relight telemetry. The WEBCAM/PHONE header selector or `C`
restarts capture/runtime and invalidates camera-dependent renderer history.
The Mode 7 status strip reports renderer,
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

## Backend selection

The live app accepts `--relight-backend auto|rtx|raster`. `auto` selects the
optional NVIDIA OptiX module only when an RTX-class GPU and native module are
available; otherwise it reports `OPENGL_RASTER`. `raster` forces the CUDA/OpenGL
path. `rtx` is strict and reports `RTX_UNAVAILABLE` when unavailable; it never
labels a fallback as RTX.

The optional native build is under `native/optix_relight/` and requires the
NVIDIA OptiX SDK, CUDA Toolkit, CMake, a C++ compiler, and pybind11. Set
`OPTIX_ROOT` or pass `--optix-root` to `tools/build_optix_backend.py`. The
current GTX 1650 Ti is intentionally detected as non-RTX and uses the raster
backend.
