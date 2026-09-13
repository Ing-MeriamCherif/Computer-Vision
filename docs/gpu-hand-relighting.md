# GPU Hand Relighting

Mode 7 uses an OpenGL 3.3 relighter and, when NVIDIA OptiX is available, an
OptiX acceleration structure built from a 96x54 sample of the live depth map.
The RTX path traces a visibility ray per coarse sample and light on NVIDIA RT
cores; bilinear upsampling feeds the full-resolution surface pass. The
acceleration structure keeps fixed topology and updates its vertices each
frame. Depth discontinuities and invalid samples are excluded. Volumetric
scattering still uses the existing screen-space visibility marcher; this is
not full path tracing and monocular depth cannot reveal hidden geometry.

It uses `fast_geometry_state` when available, falling back to `geometry_state`,
and positions lights from fresh `HandXYZ.xyz_camera`; it never samples palm
depth to infer a production light position. RGB is converted from sRGB to
linear before surface lighting, then converted back at the final composite.
Diffuse is albedo-weighted; specular is additive. The CPU Youssef renderer
remains the explicit fallback and reference path.

The GPU path combines full-resolution surface lighting, up to two independently
colored lights, OptiX RT-core surface visibility (GLSL screen-space fallback),
reduced-resolution volumetric ray marching with sample-to-light visibility,
temporal stabilization, and projected 3D emitter orbs. Each orb and its
lighting/shadow/volume rays use the same camera-space `LightState` position.
The working emitter range is
`0.30 m`; physical shadow source radius is `0.018 m`; visible orb radius is
`0.035 m`. RTX mode uses OptiX RT cores when available; GTX/non-NVIDIA systems
use the explicitly labeled raster or CPU fallback. Hidden geometry beyond the
monocular depth surface is not reconstructed.

## Install and Run

The RTX route additionally needs the NVIDIA `pyoptix` binding, `cuda-python`,
CuPy, a working NVIDIA driver/CUDA runtime, and OptiX SDK headers for NVRTC
compilation. Install the binding from NVIDIA's `otk-pyoptix` project and set
`OPTIX_INCLUDE_DIR` to the SDK's `include` directory. If any part of that
optional stack is unavailable, stats report `GLSL_SCREEN_SPACE_FALLBACK`; the
Mode 7 app remains usable, but is not using RT cores.

On Linux, install the UI/GL dependencies in the repository environment. The
CUDA-enabled PyTorch wheel must match the installed driver/runtime used by the
P123 depth provider.

```bash
./tools/setup_gpu_ui.sh
python -m tools.p123_live_app --webcam-camera /dev/video0 --phone-camera /dev/video2 --depth-backend local --depth-size 336x448 --mode 7 --lighting-quality balanced
```

For colleague/TensorRT depth deployments, install the matching optional
dependencies from `requirements-depth-engine.txt` and use the existing
provider/export instructions. Ensure the configured hand-landmarker asset is
present before starting `--hand-backend auto`.

Quality can be selected with `--lighting-quality low|balanced|high`. The RT
grid defaults to 96x54, configurable with `NRW_RT_WIDTH` and `NRW_RT_HEIGHT`;
this keeps RT traversal light while maintaining a full-resolution final
image. The last relight/session stats expose `ray_backend`, `rt_trace_ms`, and
`gpu_render_ms`; on the RTX 4060 Laptop at 1280x720 the balanced relight pass
measured about 33 ms (~30 FPS), excluding camera capture and depth inference.
The
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

Balanced settings use one RT-core surface visibility ray per coarse sample,
quarter-resolution volumetrics with six camera samples and four
sample-to-light visibility steps, and conservative depth-rejected temporal
history. The surface fallback uses deterministic screen-space rays. No
per-frame depth normalization is performed.
