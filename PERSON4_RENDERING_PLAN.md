# NRW 8th Edition AI & Vision Challenge
# Person 4 — Rendering, Lighting, Shadows, Multi-Light & Volumetrics
## Codex Execution Plan

> **Role:** Person 4  
> **Primary ownership:** GPU rendering pipeline  
> **Challenge coverage:** Level 02, Level 04, rendering side of Level 05, and support for Level ∞  
> **Recommended implementation stack:** Python + ModernGL + GLSL + GLFW  
> **Target hardware:** NVIDIA RTX 4060-class laptop GPU  
> **Main rule:** build and benchmark your rendering subsystem independently with mock inputs before waiting for the other teammates.

---

# 0. Your mission

Your job is to transform the outputs of the perception team into the final image seen by the jury.

The other members estimate:

- where surfaces are in 3D;
- what direction each surface faces;
- where the virtual light is in 3D.

You turn those data into:

1. realistic dynamic lighting;
2. specular highlights;
3. geometry-aware dynamic shadows;
4. multiple independent lights;
5. volumetric light/haze;
6. the final composited real-time output.

In simple terms:

```text
RGB + DEPTH + NORMALS + LIGHT POSITION
                 |
                 v
          YOUR GPU RENDERER
                 |
      +----------+----------+
      |          |          |
   Lighting    Shadows   Volumetrics
      |          |          |
      +----------+----------+
                 |
                 v
          FINAL LIVE FRAME
```

The challenge specifically expects real-time Lambertian/specular relighting, geometry-aware shadows, and later multiple lights plus volumetric scattering.

Your work is therefore one of the most visible parts of the entire project.

---

# 1. What you own

You are responsible for the following modules.

```text
renderer/
├── __init__.py
├── renderer.py
├── resources.py
├── contracts.py
├── quality.py
├── profiler.py
│
└── shaders/
    ├── fullscreen.vert
    ├── lighting.frag
    ├── shadow.frag
    ├── volumetric.frag
    ├── composite.frag
    └── debug.frag

tools/
├── renderer_synthetic_demo.py
├── benchmark_renderer.py
└── validate_shadow_direction.py

tests/
├── test_renderer_contracts.py
├── test_projection_math.py
└── test_quality_profiles.py
```

You should **not** own:

- the depth model;
- hand tracking;
- gesture recognition;
- normal-estimation research;
- the main AI model selection.

You may write mock versions of those outputs for testing, but do not duplicate the other teammates' production implementations.

---

# 2. What you need from the other 3 people

Do not integrate by passing random dictionaries between modules.

Agree on these interfaces before full integration.

## 2.1 From Person 1 — Hand tracking / gesture control

You need the **final smoothed virtual light state**, not raw hand landmarks.

Minimum structure:

```python
@dataclass
class Light:
    position_camera_m: np.ndarray   # shape (3,), [x, y, z] in meters
    color_rgb: np.ndarray           # shape (3,), values 0..1
    intensity: float                # >= 0
    radius_m: float                 # optional, useful later
    active: bool
    confidence: float

@dataclass
class LightState:
    lights: list[Light]             # 1 light for L2/L4, max 2 for L5
    timestamp_s: float
```

### Required coordinate convention

Everyone should use this camera coordinate system at the integration boundary:

```text
+X = right
+Y = down
+Z = forward from the camera
units = meters
```

This matches normal computer-vision camera coordinates well and avoids unnecessary transforms.

### What Person 1 must guarantee

- Light position is already smoothed.
- Pinch/gesture state is already interpreted.
- If hand tracking is lost, `active=False` or the previous stable state is returned.
- Light positions are expressed in the same camera coordinate frame as the depth map.
- Person 1 should not implement lighting formulas.

---

## 2.2 From Person 2 — Depth estimation

You need:

```python
@dataclass
class DepthFrame:
    depth_m: np.ndarray       # HxW float32, depth in meters
    valid_mask: np.ndarray    # HxW bool
    fx: float
    fy: float
    cx: float
    cy: float
    timestamp_s: float
```

### Person 2 must guarantee

- Metric depth is strongly preferred.
- Invalid pixels are marked.
- `fx, fy, cx, cy` correspond to the image resolution you receive.
- Depth and RGB are spatially aligned.
- The meaning of the depth value is documented.
- Any resizing of depth updates the camera intrinsics correctly.

You do **not** need Person 2 to send a full XYZ point cloud.

Reconstruct the per-pixel 3D point directly in your shader:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth(u,v)
```

This avoids transferring an additional H×W×3 point map every frame.

---

## 2.3 From Person 3 — Surface normals

You need:

```python
@dataclass
class NormalFrame:
    normals_camera: np.ndarray   # HxWx3 float16/float32
    valid_mask: np.ndarray       # HxW bool
    timestamp_s: float
```

Normals must use the **same camera coordinate frame**:

```text
+X right
+Y down
+Z forward
```

They should be normalized to unit length before integration, or the shader must normalize them.

Optional but useful:

```python
edge_mask: np.ndarray  # depth discontinuity / unreliable-normal areas
```

You can use this later to reduce shadow/specular artifacts at depth boundaries.

---

# 3. Final integration packet

The cleanest shared frame contract is:

```python
@dataclass
class RenderPacket:
    rgb: np.ndarray                # HxWx3 uint8 RGB
    depth: DepthFrame
    normals: NormalFrame
    lights: LightState
    frame_id: int
    timestamp_s: float
```

Your public renderer API should be roughly:

```python
renderer = Renderer(config)

while running:
    packet = latest_render_packet()
    renderer.render(packet)
```

The renderer should always consume the **latest available packet**.

Do not allow an old-frame queue to grow.

---

# 4. Critical technical decision: use screen-space shadows first

The earlier team notes mention cube shadow maps.

**Do not make cube shadow mapping your first implementation.**

Traditional point-light shadow mapping works best when you have complete scene geometry that can be rendered from the light's point of view.

Here, you mainly have a **single camera-visible depth surface**.

That means rendering six complete cube-map views from the light requires you to first reconstruct a mesh and then deal with:

- holes;
- missing geometry behind objects;
- disocclusions;
- noisy mesh boundaries;
- extra rendering cost.

For this challenge, start with:

> **Screen-space visibility / ray marching against the live depth map.**

This directly uses the data produced by the depth model and is much faster to prototype.

If screen-space limitations become visible later, a reconstructed depth-mesh + light-space shadow map can be tested as an advanced alternative.

---

# 5. Your research phase

Do this before serious implementation.

The point is not to spend days reading papers.

The goal is to extract the algorithms you need.

---

## Research Task R1 — Real-time point-light shading

### Learn

- Lambertian diffuse shading;
- Blinn-Phong specular shading;
- view vector;
- inverse-distance attenuation;
- ambient term;
- gamma/sRGB vs linear RGB;
- tone mapping / clipping.

### Core equations

For a 3D surface point `P`, normal `N`, light position `Lp`, and camera at origin:

```text
L = normalize(Lp - P)
V = normalize(-P)
H = normalize(L + V)
```

Diffuse:

```text
diffuse = max(dot(N, L), 0)
```

Specular:

```text
specular = pow(max(dot(N, H), 0), shininess)
```

Distance:

```text
d = length(Lp - P)
```

Stable attenuation for the demo:

```text
attenuation = 1 / (1 + k * d*d)
```

Initial combined light:

```text
lighting =
    ambient
  + attenuation * light_intensity * diffuse
  + attenuation * light_intensity * specular_strength * specular
```

Initial compositing:

```text
relit_rgb =
    original_rgb_linear * (ambient + diffuse_term)
  + specular_term
```

Then convert back to display RGB.

### Deliverable

Create:

```text
docs/person4_research_lighting.md
```

with:

- equations;
- chosen constants;
- what each constant controls;
- screenshots/notes from a synthetic test.

---

## Research Task R2 — Screen-space shadow rays

### Primary paper

**Morgan McGuire & Michael Mara — Efficient GPU Screen-Space Ray Tracing, JCGT 2014**

Main page:

https://jcgt.org/published/0003/04/04/

What matters for you:

- tracing against an existing depth buffer;
- perspective-correct stepping;
- DDA-style traversal;
- avoiding oversampling;
- GLSL implementation details;
- jitter/stride trade-offs;
- limitations of screen-space information.

Do **not** try to implement their final optimized method immediately.

Use the paper to understand how to improve your first simple ray marcher.

### First algorithm to implement

For each visible surface point `P`, trace toward the light:

```text
Q(t) = P + t * (Lp - P)
```

for:

```text
0 < t < 1
```

For each sample `Q`:

1. Project `Q` back into the camera image.
2. Read scene depth at that projected pixel.
3. Determine whether a visible surface lies between `P` and the light.
4. If yes, mark the pixel shadowed.

Projection:

```text
u = fx * Q.x / Q.z + cx
v = fy * Q.y / Q.z + cy
```

### Important parameters

You will need to tune:

```text
shadow_steps
shadow_bias_m
shadow_thickness_m
shadow_resolution_scale
ray_start_offset
```

Start with:

```text
shadow_steps = 12
shadow_resolution_scale = 0.5
```

Do not treat these numbers as final.

Benchmark them.

### Deliverable

Create:

```text
docs/person4_research_shadows.md
```

containing:

- basic ray marcher;
- optimized DDA method summary;
- screen-space limitations;
- parameters to benchmark;
- chosen fallback behavior.

---

## Research Task R3 — Depth-aware AR rendering

### Reference

**DepthLab: Real-time 3D Interaction with Depth Maps for Mobile Augmented Reality**

Google Research:

https://research.google/pubs/depthlab-real-time-3d-interaction-with-depth-maps-for-mobile-augmented-reality/

Focus on:

- depth-aware occlusion;
- relighting;
- geometry-aware shadows;
- depth-based visual effects.

Why this is useful:

The project is close to the challenge's conceptual objective: take live depth information and use it for interactive graphics.

### Deliverable

Write a short section in:

```text
docs/person4_research_shadows.md
```

called:

```text
Lessons from DepthLab
```

Keep only techniques useful to the challenge.

---

## Research Task R4 — Volumetric light scattering

### Reference

**GPU Gems 3 — Chapter 13: Volumetric Light Scattering as a Post-Process**

NVIDIA:

https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-13-volumetric-light-scattering-post-process

Study:

- screen-space radial sampling;
- light position projected into screen space;
- sample density;
- decay;
- weight;
- exposure;
- occlusion.

This is an efficient way to create a convincing "light through haze" effect without full physical volumetric simulation.

### Deliverable

Create:

```text
docs/person4_research_volumetric.md
```

with:

- simplified algorithm;
- expected GPU cost;
- tunable parameters;
- limitations.

---

## Research Task R5 — ModernGL rendering architecture

### Reference

ModernGL documentation:

https://moderngl.readthedocs.io/

Learn only what you need:

- creating a context;
- textures;
- framebuffers;
- shader programs;
- uniforms;
- fullscreen triangle/quad;
- offscreen render targets;
- GPU timing queries if available.

### Your rendering pipeline should eventually look like

```text
RGB texture
Depth texture
Normal texture
      |
      v
[ Lighting pass ]
      |
      v
[ Shadow pass at 1/2 resolution ]
      |
      v
[ Optional volumetric pass at 1/4 resolution ]
      |
      v
[ Composite pass ]
      |
      v
Final framebuffer
```

---

# 6. Practical implementation plan

Do these steps in order.

Do not jump directly to Level 5.

---

# PHASE P0 — Create your branch and renderer skeleton

## Git

Create your own branch:

```bash
git checkout -b feat/person4-renderer
```

Recommended commits:

```text
feat(renderer): bootstrap ModernGL pipeline
feat(renderer): add camera-space lighting
feat(renderer): add depth-based shadows
feat(renderer): integrate shared contracts
feat(renderer): add multi-light
feat(renderer): add volumetric pass
perf(renderer): add adaptive quality and profiling
```

## Install initial dependencies

Keep the renderer dependencies small.

Likely:

```text
moderngl
glfw
numpy
PyYAML
```

Do not add a large graphics engine unless the team explicitly changes architecture.

## Definition of done

- A window opens.
- A fullscreen RGB image is rendered through a GLSL shader.
- No CPU per-pixel rendering.
- `python tools/renderer_synthetic_demo.py` works.

---

# PHASE P1 — Build a synthetic scene before integration

This is extremely important.

Do not wait for Person 2 or Person 3.

Generate fake inputs:

```text
RGB:
    simple image or webcam frame

Depth:
    background plane = 3.0 m
    foreground rectangle/circle = 1.5 m

Normals:
    background = [0, 0, -1] or your agreed orientation
    synthetic tilted planes for testing

Light:
    keyboard/mouse-controlled XYZ point
```

The exact normal sign depends on the convention chosen by the team.

Add a debug mode that displays:

```text
1 = RGB
2 = Depth
3 = Normals
4 = Lighting
5 = Shadow
6 = Final
```

## Definition of done

You can develop all rendering logic even if the other teammates' modules do not exist yet.

---

# PHASE P2 — GPU texture and data pipeline

Create GPU textures:

```text
rgb_tex       -> RGB8 / sRGB-compatible
depth_tex     -> R32F
normal_tex    -> RGB16F
shadow_tex    -> R8/R16F
volumetric    -> RGB16F
```

Avoid reallocating textures every frame.

Allocate once.

Then update their contents.

### Major performance rule

Bad:

```text
create texture -> upload -> destroy texture -> every frame
```

Good:

```text
create texture once
update existing texture every frame
```

## Handle texture origin exactly once

OpenCV / NumPy images usually use:

```text
origin = top-left
```

OpenGL texture coordinates are often treated as:

```text
origin = bottom-left
```

Decide exactly where the Y flip occurs.

Do not scatter `1.0 - uv.y` throughout five shaders.

Put it in one controlled place.

## Definition of done

Uploading RGB + depth + normals does not create increasing memory usage and does not dominate frame time.

---

# PHASE P3 — Per-pixel 3D reconstruction in GLSL

Do not require a full XYZ image from Person 3.

Inside the shader:

```glsl
float z = texture(depth_tex, uv).r;

float x = (pixel_x - cx) * z / fx;
float y = (pixel_y - cy) * z / fy;

vec3 P = vec3(x, y, z);
```

Uniforms:

```text
fx
fy
cx
cy
image_width
image_height
```

Reject invalid depth.

Example:

```glsl
if (z <= 0.0 || !valid_depth) {
    // preserve original RGB or use fallback
}
```

## Unit test

Write `tests/test_projection_math.py`.

Test known values.

Example:

At:

```text
u = cx
v = cy
z = 2m
```

you must get approximately:

```text
P = [0, 0, 2]
```

Do not proceed until projection and back-projection are correct.

---

# PHASE P4 — Level 02: Lambertian lighting

Start with diffuse only.

For every pixel:

```glsl
vec3 light_vec = light_pos - P;
float distance_to_light = length(light_vec);
vec3 L = normalize(light_vec);

float ndotl = max(dot(N, L), 0.0);
```

Initial:

```glsl
float attenuation = 1.0 / (1.0 + k * distance_to_light * distance_to_light);
```

Then:

```glsl
vec3 diffuse =
    base_rgb
    * light_color
    * light_intensity
    * attenuation
    * ndotl;
```

Add ambient:

```glsl
vec3 final = base_rgb * ambient + diffuse;
```

### Debug controls

During development allow:

```text
A/D -> light X
W/S -> light Y
Q/E -> light Z
```

Do this before hand control is integrated.

## Validation

Test a tilted plane.

Move light:

- left;
- right;
- up;
- down;
- forward;
- backward.

The highlight/brightness must move in the physically expected direction.

## Definition of done

Level 02 diffuse lighting works with a manually moved 3D point light at stable real-time FPS.

---

# PHASE P5 — Add Blinn-Phong specular

Compute view direction:

```glsl
vec3 V = normalize(-P);
```

Half vector:

```glsl
vec3 H = normalize(L + V);
```

Specular:

```glsl
float spec = pow(max(dot(N, H), 0.0), shininess);
```

Then:

```glsl
vec3 specular =
    light_color
    * light_intensity
    * attenuation
    * specular_strength
    * spec;
```

Final:

```glsl
vec3 final_linear =
    base_rgb_linear * ambient
    + diffuse
    + specular;
```

### Start parameters

Try:

```text
ambient = 0.15
specular_strength = 0.20
shininess = 48
```

These are starting points only.

Expose them in config.

## Important color rule

Do lighting in approximately linear color space.

Do not perform all lighting directly on gamma-encoded 8-bit RGB if avoidable.

At minimum:

```text
sRGB -> linear
perform lighting
linear -> sRGB
```

## Definition of done

You can clearly show:

- diffuse response;
- specular highlight;
- XYZ light movement.

---

# PHASE P6 — Create the baseline dynamic shadow

This is your most important technical phase.

The Level 04 result should be more important than volumetrics.

## Pass resolution

Start at:

```text
shadow_resolution = 0.5 * output resolution
```

For 1280×720:

```text
640×360 shadow pass
```

Then upscale.

## Basic ray algorithm

For a shaded point `P`:

```text
ray = Lp - P
```

Ignore a tiny region near `P`:

```text
start_t > 0
```

For each step:

```text
Q = P + t * ray
```

Project `Q`:

```text
u = fx * Q.x/Q.z + cx
v = fy * Q.y/Q.z + cy
```

Get:

```text
scene_z = depth(u,v)
```

If a reliable scene sample indicates the ray is blocked before reaching the light:

```text
visibility = 0
```

Else:

```text
visibility = 1
```

Then:

```text
direct_lighting *= visibility
```

Keep ambient light even in shadow.

---

# PHASE P7 — Fix screen-space shadow artifacts

Your first shadows will probably have problems.

Expect:

- self-shadow acne;
- holes at depth edges;
- flicker;
- disconnected shadows;
- missing off-screen occluders;
- incorrect results from noisy depth.

Fix them in this order.

## 7.1 Self-shadow bias

Add:

```text
shadow_bias_m
```

and/or begin the ray slightly away from `P`.

## 7.2 Thickness tolerance

Depth maps represent surfaces with no volume.

Use:

```text
shadow_thickness_m
```

so tiny depth differences are not treated as definite intersections.

## 7.3 Invalid-depth fallback

If the ray leaves the image or hits invalid depth:

Recommended demo behavior:

```text
treat unknown as unoccluded
```

Why:

Random black shadows are visually worse than a missing shadow.

## 7.4 Reduce depth-edge artifacts

Use Person 3's optional edge mask.

At strong depth discontinuities:

- reduce shadow confidence;
- soften the shadow;
- avoid aggressive specular.

## 7.5 Soft shadow approximation

Do not implement expensive physical area-light integration initially.

Try one of:

- small jitter of the shadow ray;
- 2–4 nearby rays;
- edge-aware blur of the shadow mask.

Benchmark all variants.

---

# PHASE P8 — Upgrade the ray marcher

Once the simple version works, use the ideas from McGuire & Mara.

Compare:

### Version A

```text
fixed 3D ray steps
```

### Version B

```text
perspective-aware / screen-space DDA traversal
```

Measure:

```text
shadow pass ms
visual holes
temporal stability
```

Do not replace A unless B is clearly better in your implementation.

Keep A as fallback.

---

# PHASE P9 — Integrate Person 1's real light state

Only after manual XYZ control is correct.

Map:

```python
packet.lights.lights[0].position_camera_m
```

directly to your shader uniform.

Do not interpret hand landmarks yourself.

### Lost hand behavior

Agree with Person 1 on one policy:

Recommended:

```text
short loss (< ~0.3 s):
    hold last stable light

long loss:
    gracefully fade direct light toward default/off
```

No teleporting light.

No NaNs.

No crash.

---

# PHASE P10 — Integrate real depth and normals

Integration order:

```text
1. RGB only
2. RGB + real depth
3. RGB + depth + real normals
4. manual virtual light
5. real hand-controlled light
```

When something breaks, this order tells you which module introduced the problem.

### Add debug visualizations

Always keep:

```text
RGB
Depth
Normals
N dot L
Shadow mask
Final
FPS / timings
```

This is useful both for debugging and for the jury.

---

# PHASE P11 — Level 05: two independent lights

After one light is perfect.

Support at most:

```text
MAX_LIGHTS = 2
```

Initial shader structure:

```glsl
for (int i = 0; i < light_count; ++i) {
    if (!light_active[i]) continue;

    // compute lighting
    // compute shadow visibility
    // accumulate contribution
}
```

Use visibly different colors during the demo:

```text
Light 1 = warm
Light 2 = cool
```

This makes independence visually obvious.

## Benchmark

Record:

```text
one-light lighting ms
one-light shadow ms

two-light lighting ms
two-light shadow ms
```

If two full shadow traces are too expensive:

- lower shadow resolution;
- lower steps;
- calculate lower-quality shadows for the second light;
- use adaptive quality.

Do not let Level 5 destroy Level 4 fluidity.

---

# PHASE P12 — Volumetric scattering

Only start this after:

```text
lighting = stable
shadows = stable
two lights = stable
```

## MVP method

Implement the GPU Gems-style screen-space post-process.

For a projected light position:

```text
pixel ----------------------> light_screen_pos
```

sample along that 2D direction.

Accumulate light using parameters such as:

```text
num_samples
density
decay
weight
exposure
```

Run at:

```text
1/4 resolution
```

and upscale.

## Make it depth-aware

Do not draw volumetric glow blindly over foreground objects.

Use depth/occlusion information to reduce the effect when a foreground surface blocks the light.

## Initial quality

Start around:

```text
12-16 samples
quarter resolution
```

Then benchmark.

## Definition of done

The volumetric effect is visually obvious but does not reduce the full system below the team's stable FPS target.

---

# PHASE P13 — Support a strong Level ∞ feature

Recommended renderer-owned Level ∞ contribution:

## Depth-occluded virtual light orb

Render a visible glowing orb at the 3D light position.

Project light:

```text
u_light = fx * L.x/L.z + cx
v_light = fy * L.y/L.z + cy
```

Sample scene depth at that location.

If:

```text
scene_depth < light_depth
```

the light is behind a real object.

Therefore the orb should be:

- hidden;
- clipped;
- or partially occluded.

If the light is in front:

- show orb;
- bloom/glow it.

This creates a very strong illusion that the virtual light actually exists in the reconstructed 3D scene.

Optional later:

- light radius controlled by pinch;
- light color controlled by a second gesture;
- GGX/Cook-Torrance specular;
- temporal shadow accumulation.

Do not implement these before the required levels work.

---

# 7. Your exact performance work

Your role is not finished when the image looks good.

You also need it to be fast.

Create:

```text
renderer/profiler.py
tools/benchmark_renderer.py
```

Track at least:

```text
frame upload ms
lighting pass ms
shadow pass ms
volumetric pass ms
composite pass ms
total renderer ms
final FPS
```

Write benchmark output to CSV:

```text
benchmarks/renderer_benchmark.csv
```

Suggested columns:

```text
timestamp
resolution
num_lights
shadow_scale
shadow_steps
volumetric_enabled
volumetric_samples
upload_ms
lighting_ms
shadow_ms
volumetric_ms
composite_ms
total_gpu_ms
fps
```

---

# 8. Adaptive quality profiles

Create three profiles.

## SAFE

```text
output: 1280x720
shadow scale: 0.5
shadow samples: 8
volumetric: off or very low
lights: 1-2
```

## BALANCED

```text
output: 1280x720
shadow scale: 0.5
shadow samples: 12-16
volumetric scale: 0.25
volumetric samples: 12
lights: 2
```

## CINEMATIC

```text
output: 1280x720
shadow scale: 0.5-1.0 depending benchmark
shadow samples: 20-24
volumetric samples: 16-24
extra softening / glow
```

Do not automatically use CINEMATIC on stage.

The stage profile should be the highest-quality profile proven stable.

---

# 9. Renderer acceptance criteria

Do not say your module is finished until these pass.

## Level 02

- [ ] Light moves in X.
- [ ] Light moves in Y.
- [ ] Light moves in Z.
- [ ] Diffuse shading changes correctly.
- [ ] Specular highlight changes correctly.
- [ ] No NaNs or flashes on invalid depth.
- [ ] Real-time performance is stable.

## Level 04

- [ ] Foreground object casts a visible shadow.
- [ ] Shadow direction changes with light X/Y.
- [ ] Shadow length/behavior changes with light Z.
- [ ] Shadow updates while object/person moves.
- [ ] No catastrophic flicker.
- [ ] Invalid depth does not create random black areas.
- [ ] Stable for at least 10 continuous minutes.

## Level 05

- [ ] Two lights are independent.
- [ ] Two colors are visually distinguishable.
- [ ] Both contribute to lighting.
- [ ] Overlapping shadow behavior is understandable.
- [ ] Volumetric effect is visible.
- [ ] FPS remains above team minimum.

## Level ∞ support

- [ ] Light orb projects correctly into image.
- [ ] Orb is occluded by closer real geometry.
- [ ] Orb becomes visible again when moved in front.

---

# 10. Tests you must perform

Create a repeatable test sheet.

## Test scene A — Flat wall

Purpose:

```text
basic lighting
specular stability
```

## Test scene B — Box in front of wall

Purpose:

```text
shadow direction
shadow length
depth discontinuity
```

## Test scene C — Human in front of background

Purpose:

```text
real silhouette
moving geometry
temporal shadow quality
```

## Test scene D — Hand close to camera

Purpose:

```text
extreme depth
light tracking interaction
occlusion
```

## Test scene E — Low-texture region

Purpose:

```text
depth noise robustness
normal noise robustness
```

## Stress test

For 10-20 minutes:

- move hand quickly;
- remove hand;
- re-enter;
- move objects;
- cover camera briefly;
- get very close;
- move light off screen;
- activate two lights;
- switch debug modes.

The renderer must never crash.

---

# 11. What to ask each teammate to deliver

Send them this checklist.

## Message to Person 1

```text
I need a stable LightState API.
Please give me:
- position [x,y,z] in camera coordinates, meters
- RGB light color 0..1
- intensity
- active/confidence
- timestamp
- max 2 lights
Do the hand/gesture smoothing before sending the state.
Coordinate convention: +X right, +Y down, +Z forward.
```

## Message to Person 2

```text
I need:
- depth map in meters
- valid-depth mask
- fx, fy, cx, cy for the exact depth/RGB resolution
- aligned RGB/depth
- timestamp
Please document invalid-depth values and any resize performed.
```

## Message to Person 3

```text
I need:
- HxWx3 unit normals
- camera-space convention +X right, +Y down, +Z forward
- valid-normal mask
- timestamp
Optional: depth-discontinuity/edge mask.
```

---

# 12. Integration rules for the team

These rules will prevent wasted time.

## Rule 1

No teammate changes another teammate's module API without discussion.

## Rule 2

Shared contracts live in one file:

```text
contracts/render_types.py
```

or equivalent.

## Rule 3

Every data object includes:

```text
frame_id or timestamp
```

## Rule 4

The rendering loop consumes latest state.

No long frame queue.

## Rule 5

Every major module can be disabled from config.

Example:

```yaml
renderer:
  lighting: true
  shadows: true
  volumetric: true
  visible_light_orb: true
```

This gives you stage fail-safes.

---

# 13. Proposed configuration

Create something similar to:

```yaml
renderer:
  output_width: 1280
  output_height: 720

  ambient_strength: 0.15
  specular_strength: 0.20
  shininess: 48.0
  attenuation_k: 0.6

  shadows:
    enabled: true
    resolution_scale: 0.5
    steps: 12
    bias_m: 0.015
    thickness_m: 0.05
    soften: true

  volumetric:
    enabled: false
    resolution_scale: 0.25
    samples: 12
    density: 0.8
    decay: 0.95
    exposure: 0.5

  max_lights: 2

  debug:
    show_fps: true
    mode: final
```

All numbers are starting points.

Benchmark and tune them.

---

# 14. Codex workflow

Put this file in the repository root as:

```text
PERSON4_RENDERING_PLAN.md
```

Then use Codex in small tasks.

Do not ask Codex to "build the whole renderer" in one giant request.

---

# 15. First Codex prompt — repository analysis

Use this first:

```text
Read PERSON4_RENDERING_PLAN.md completely.

You are helping me implement Person 4's renderer for the NRW AI & Vision Challenge.

First inspect the entire repository and report:
1. current project structure,
2. how main.py currently works,
3. existing shared data structures,
4. existing RGB/depth/normal/light APIs,
5. dependencies and graphics libraries already installed,
6. likely integration conflicts,
7. the exact files you propose to create or modify.

Do not implement anything yet.
Do not refactor teammates' modules.
Use PERSON4_RENDERING_PLAN.md as the source of truth for my responsibilities.
```

Review Codex's answer before coding.

---

# 16. Codex prompt — renderer bootstrap

```text
Read PERSON4_RENDERING_PLAN.md and the repository.

Implement PHASE P0 and P1 only.

Requirements:
- create an isolated renderer module,
- use ModernGL + GLSL,
- create a simple window/context,
- render a fullscreen synthetic RGB image,
- build reusable textures rather than reallocating each frame,
- create a synthetic depth map and synthetic normal map,
- create manual XYZ keyboard controls for one point light,
- add debug modes for RGB/depth/normals,
- add a runnable tools/renderer_synthetic_demo.py,
- do not integrate teammates' AI modules yet,
- do not modify their code unless absolutely necessary.

Add basic tests for contracts/projection math where applicable.
Run the demo/tests and report the results.
```

---

# 17. Codex prompt — Level 02 lighting

```text
Read PERSON4_RENDERING_PLAN.md.

Implement PHASE P3, P4, and P5 on the existing synthetic renderer.

Requirements:
- reconstruct per-pixel camera-space XYZ from depth using fx/fy/cx/cy,
- implement Lambertian diffuse lighting in GLSL,
- implement Blinn-Phong specular lighting in GLSL,
- use a 3D point-light position in camera coordinates,
- include distance attenuation,
- keep ambient illumination in dark regions,
- perform lighting in linear color space as far as practical,
- expose all important constants in config,
- preserve debug views,
- add projection/back-projection tests,
- add timing for the lighting pass.

Do not implement shadows yet.

After implementation:
- run tests,
- run a benchmark,
- summarize files changed and measured timings.
```

---

# 18. Codex prompt — baseline screen-space shadows

```text
Read PERSON4_RENDERING_PLAN.md, especially PHASE P6 and P7.

Implement the first working geometry-aware screen-space shadow system.

Requirements:
- use the live/synthetic depth texture,
- for each shaded point trace samples toward the point light,
- project ray samples back to the depth map,
- compare against scene depth,
- add self-shadow bias,
- add configurable thickness tolerance,
- handle invalid/off-screen samples safely,
- compute shadows in a separate framebuffer,
- start at half resolution,
- make sample count configurable,
- keep ambient light visible in shadow,
- add a shadow-mask debug view,
- measure shadow-pass GPU/CPU timing.

Do not implement cube shadow maps.
Do not implement volumetrics yet.

Use the simplest correct ray marcher first.
Keep the code structured so we can later replace it with DDA traversal.

Run tests/demo and report artifacts and timings.
```

---

# 19. Codex prompt — shadow optimization

```text
Read PERSON4_RENDERING_PLAN.md and review the existing shadow implementation.

Research the implementation details in McGuire & Mara's
"Efficient GPU Screen-Space Ray Tracing" only as needed.

Improve the existing shadow pass without changing its public API.

Benchmark at least:
- 8 ray steps,
- 12 ray steps,
- 16 ray steps,
and at least:
- 0.5 shadow resolution,
- 1.0 shadow resolution if feasible.

Implement only optimizations that give a clear quality/performance benefit.

Add:
- optional jitter or softening,
- edge handling,
- safe invalid-depth behavior.

Keep the simple implementation available as a fallback.

Write benchmark results to benchmarks/renderer_benchmark.csv.
```

---

# 20. Codex prompt — teammate integration

```text
Read PERSON4_RENDERING_PLAN.md and inspect the current teammate APIs.

Integrate the renderer with the real pipeline using a RenderPacket-style contract.

Requirements:
- accept RGB,
- accept metric depth + intrinsics,
- accept camera-space normals,
- accept the final smoothed LightState,
- do not duplicate depth, normal, or gesture inference,
- validate shapes/dtypes,
- reject or gracefully handle stale/invalid inputs,
- avoid blocking queues,
- use latest-frame semantics,
- keep the synthetic demo working.

If the existing teammate APIs differ from PERSON4_RENDERING_PLAN.md,
adapt at the integration boundary instead of rewriting their modules.

Run integration smoke tests and document any API mismatch.
```

---

# 21. Codex prompt — two-light Level 05

```text
Read PERSON4_RENDERING_PLAN.md.

Add Level 05 multi-light support.

Requirements:
- maximum 2 active point lights,
- independent position/color/intensity,
- accumulate diffuse + specular contribution,
- support geometry-aware shadows for each light,
- expose per-light debug controls,
- use a warm default for light 1 and cool default for light 2,
- benchmark 1 light versus 2 lights,
- preserve the stable single-light path.

Do not add volumetrics until multi-light is stable.
```

---

# 22. Codex prompt — volumetric pass

```text
Read PERSON4_RENDERING_PLAN.md and GPU Gems 3 Chapter 13 concepts.

Implement a lightweight Level 05 volumetric-scattering post-process.

Requirements:
- separate low-resolution framebuffer,
- projected screen-space light position,
- configurable samples/density/decay/weight/exposure,
- start at quarter resolution,
- make the result depth/occlusion aware where possible,
- composite additively but avoid clipping,
- make the feature instantly disableable,
- add timing and debug view.

Target visual impact with low GPU cost, not full physical simulation.

Benchmark with volumetrics on/off.
```

---

# 23. Codex prompt — Level ∞ light orb

```text
Read PERSON4_RENDERING_PLAN.md.

Implement the renderer-owned Level Infinity feature:
a visible 3D virtual light orb that is correctly occluded by real scene depth.

Requirements:
- project the point-light 3D position into the camera image,
- render a glowing orb/halo,
- compare its depth against scene depth,
- hide or clip it when real geometry is in front,
- reveal it when moved in front of the geometry,
- work with one or two lights,
- keep it optional through config.

Add a debug visualization showing the light's projected coordinates and depth comparison.
```

---

# 24. Final Codex prompt — hardening

```text
Read PERSON4_RENDERING_PLAN.md.

Do a final renderer hardening pass.

Do not redesign the architecture.

Tasks:
1. profile all renderer passes,
2. find avoidable allocations/copies,
3. verify textures/framebuffers are reused,
4. verify invalid depth/normals cannot produce NaNs,
5. verify hand/light disappearance cannot crash rendering,
6. verify 1-light and 2-light modes,
7. verify shadow and volumetric fallbacks,
8. add SAFE/BALANCED/CINEMATIC quality profiles,
9. run a long-duration stress test,
10. write a concise docs/person4_renderer_final_report.md.

Report:
- average FPS,
- renderer ms/frame,
- lighting ms,
- shadow ms,
- volumetric ms,
- GPU/CPU bottlenecks,
- recommended stage configuration.
```

---

# 25. Your daily order of work

If you need a very concrete order:

## Day 1

- Read R1, R2, R3.
- Create branch.
- Build ModernGL window.
- Fullscreen texture.
- Synthetic depth/normals.
- Manual XYZ light.

## Day 2

- Camera back-projection.
- Lambertian lighting.
- Specular.
- Linear/sRGB handling.
- Benchmark.

## Day 3

- Basic screen-space shadow ray marcher.
- Shadow debug mask.
- Bias/thickness tuning.
- Half-resolution pass.

## Day 4

- Shadow stabilization.
- Edge behavior.
- DDA/screen-space optimization experiment.
- Benchmark quality vs speed.

## Day 5

- Integrate real depth.
- Integrate real normals.
- Integrate real light state.
- Fix coordinate/UV problems.

## Day 6

- Two-light support.
- Performance optimization.
- Adaptive quality.

## Day 7

- Volumetric scattering.
- Light orb / Level ∞.
- Final hardening.

If the team has more time, extend each phase rather than adding random features.

---

# 26. Priority order if time becomes short

If the deadline approaches, prioritize exactly like this:

```text
1. Stable GPU renderer
2. Correct Level 02 lighting
3. Excellent Level 04 shadows
4. Smooth real hand integration
5. Two-light Level 05
6. Volumetric effect
7. Light-orb Level Infinity
8. Extra visual polish
```

Do not sacrifice Level 04 stability for a fancy volumetric effect.

---

# 27. Stage fail-safe strategy

Your renderer must have hotkeys or config toggles.

Recommended:

```text
F1 -> final mode
F2 -> depth
F3 -> normals
F4 -> shadow mask
F5 -> toggle shadows
F6 -> toggle volumetrics
F7 -> toggle second light
F8 -> SAFE quality
F9 -> BALANCED quality
F10 -> CINEMATIC quality
```

If volumetrics cause a problem on stage:

```text
disable volumetrics
```

without restarting.

If two lights cause performance problems:

```text
disable second light
```

without restarting.

The core demo must continue.

---

# 28. What your final contribution should look like

At the end, your subsystem should support this sequence:

```text
Raw RGB
   |
   + Depth + Intrinsics
   |
   + Surface Normals
   |
   + 3D Light State
   |
   v
GPU CAMERA-SPACE RECONSTRUCTION
   |
   v
LAMBERT + SPECULAR
   |
   v
DEPTH-BASED SCREEN-SPACE SHADOW
   |
   v
MULTI-LIGHT
   |
   v
VOLUMETRIC SCATTERING
   |
   v
DEPTH-OCCLUDED VIRTUAL LIGHT ORB
   |
   v
FINAL LIVE OUTPUT
```

The jury should be able to move a hand and instantly understand that:

- the light exists in 3D;
- surfaces react according to their orientation;
- the light can move closer/farther;
- foreground objects cast shadows;
- two hands can control two lights;
- light interacts with real geometry.

That is your target.

---

# 29. Research references

## Challenge-related / depth-aware rendering

**DepthLab — Real-time 3D Interaction with Depth Maps for Mobile Augmented Reality**  
Google Research / UIST 2020  
https://research.google/pubs/depthlab-real-time-3d-interaction-with-depth-maps-for-mobile-augmented-reality/

Use it for:
- depth-aware rendering concepts;
- occlusion;
- geometry-aware effects;
- shadows and relighting inspiration.

## Screen-space ray tracing

**Morgan McGuire & Michael Mara — Efficient GPU Screen-Space Ray Tracing**  
Journal of Computer Graphics Techniques, 2014  
https://jcgt.org/published/0003/04/04/

Use it for:
- optimized depth-buffer ray traversal;
- DDA;
- screen-space visibility;
- GLSL reference implementation.

## Volumetric light

**Kenny Mitchell — Volumetric Light Scattering as a Post-Process**  
GPU Gems 3, Chapter 13  
https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-13-volumetric-light-scattering-post-process

Use it for:
- cheap real-time "god ray"/haze effect;
- radial screen-space sampling;
- occlusion-aware post-processing.

## Graphics API

**ModernGL documentation**  
https://moderngl.readthedocs.io/

Use it for:
- textures;
- framebuffers;
- shaders;
- full-screen GPU passes;
- resource management.

---

# 30. One final rule

**Your first goal is not the most realistic renderer.**

Your first goal is:

```text
correct
+
fast
+
stable
+
easy to integrate
```

Then improve quality.

For this challenge, a stable Level 04 implementation at high FPS is more valuable than an unstable renderer with many advanced effects.
