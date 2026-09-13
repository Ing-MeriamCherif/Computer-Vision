# P123 QUALITY/PERFORMANCE CONSOLIDATION

> **LATEST RAMI INTEGRATION STATE — 2026-09-13**
>
> Branch: `rami` (integration branch), starting from `eb2cfb0`.
> The production default is the preserved local Depth Anything provider at
> **336×448 FP32**; `--depth-backend colleague` and `--depth-backend mariem`
> remain selectable alternatives, and `--depth-size native` is explicit.
> The local provider now returns a CPU-compatible `DepthState` plus a private
> device-resident state for CUDA normals, preserving frame IDs and latest-only
> pairing. Display depth bounds use an EMA of robust 2–98% bounds; this is
> visualization-only and never changes geometry. XYZ reports capture/source age
> separately from worker completion age.
>
> Fresh physical run after this pass (`/dev/video0`, 640×480 MJPG, GTX 1650 Ti,
> CUDA, local 336×448 FP32, 18 s): capture **30.65 Hz**, depth **17.17 Hz**,
> CUDA normals **17.07 Hz**, hands **27.66 Hz**, XYZ **25.00 Hz**, with no
> depth errors or overwritten frames. The GUI was inspected live in the depth
> panel and rendered native-resolution structure continuously. Focused P123,
> CUDA, normal, camera, state, and temporal tests: **77 passed**.

The sections below are retained as historical run records.

Base SHA: `85cae0e6deae1da8eff252907631de805392a531`

Final SHA (code): `70f9ede`

Branch: `bug-fixes/p123-quality-performance-consolidation`

## LATEST AUTHORITATIVE LIVE RUN (2026-09-13)

GPU: NVIDIA GeForce GTX 1650 Ti (4 GB), CUDA, FP32 depth (measured faster than FP16)

Camera: HP Wide Vision HD Camera `/dev/video0`, V4L2, MJPG, negotiated 640x480 at 30 FPS

Depth input comparison on the physical webcam (10 seconds each):

| Input | Capture Hz | Depth Hz | CUDA normals Hz | Hands Hz |
|---|---:|---:|---:|---:|
| 336x448 | 28.4 | 17.5 | 17.7 | 24.6 |
| 378x504 | 27.7 | 14.6 | 14.7 | 24.4 |
| 420x560 | 30.6 | 11.2 | 11.3 | 23.8 |
| native 480x640 | 28.0 | 8.8 | 8.8 | 25.0 |

Selected live sweet spot: **336x448 FP32**. Depth and normals are one
latest-only CUDA-backed stream; full CPU temporal geometry is opt-in with
`--full-temporal` and is not on the default live path. XYZ freshness is capped
at 200 ms (adaptive to depth cadence), with motion-responsive smoothing and
expiry of held coordinates. No physical hand was present during the bounded
headless comparison, so XYZ coordinates were not claimed as hardware-validated.

Direct visual inspection of rendered panels found and corrected display-only
depth speckle, normal speckle, and an obscured XYZ-axis legend. A low-confidence
MediaPipe false hand near the face was also rejected; the post-fix webcam panel
showed zero hands when no hand was present.

### Mariem `main` depth module integration

The complete `depth_module/` package from Mariem's `upstream/main` commit
`a7ed804` is vendored unchanged and is now the **sole P123 live depth
provider**. The old local and colleague adapters are not selectable by
`tools.p123_live_app`; they remain only for legacy non-P123 tools. On this GTX
1650 Ti, Mariem FP32 is materially faster and cleaner than its FP16 path
(approximately 9–10 Hz at the default 336px input versus approximately 3 Hz
FP16). Talel's hand tracker and palm-size XYZ math remain the P123 hand path.

### Talel XYZ correction

XYZ now uses Talel's `light_vector.py` math for relative-depth runs: apparent
palm width estimates metric Z with `z = fx * 0.085 / palm_pixels`, clamped to
0.2–3.0 m, followed by camera-model back-projection. A sampled depth value is
used directly only for an explicitly metric depth state. This prevents the
previous relative-depth values (for example `z≈0.02`) from being mislabeled as
camera-space meters. The pending-state bug was a timestamp-domain error:
freshness now uses the CUDA geometry completion timestamp rather than the older
camera capture timestamp, so valid hand XYZ is not rejected while inference is
still within the adaptive 180–200 ms window.

## P123 CONSOLIDATION NOTES

The runtime now negotiates camera dimensions after `start()`, rebuilds the
uncalibrated camera model to the negotiated size, prefers advertised MJPG, and
requests a capacity-one V4L2 buffer. Metrics use bounded deques, RGB history is
limited to 24 frames, and worker waits use condition/version notifications
instead of busy polling. The HUD reports camera/depth/normals/temporal/hands
rates plus XYZ fresh/degraded age. CUDA normals use edge-aware radii 1/2 and
unit-vector normalization without per-frame synchronization or peak-memory
resets.

Remaining measured limitations: Mariem's preserved Hugging Face pipeline remains
the depth hot path and publishes a CPU-compatible `DepthState`; CUDA normals
consume that latest state without a queue or backlog. TensorRT/ONNX conversion
and architectural replacement were not introduced. Physical XYZ hand
coordinates require a hand in view and were not claimed from a no-hand webcam
soak.

============================================================

# P123 MODE-BY-MODE LIVE REMEDIATION REPORT

Starting commit: `e5b1fec`

Ending commit (historical section): `5b882a9` (superseded by the authoritative run above)

Branch: `bug-fixes/live-p123-physical-camera-remediation`

============================================================
NATIVE-RESOLUTION PERFORMANCE UPDATE (2026-09-13)
============================================================

The P123 live path now defaults to the physical camera's native 640x480 input
for depth and all geometry/hand stages. Depth inference is asynchronous and
latest-only; FP32 is the default on this GTX 1650 Ti because its measured
native latency is lower than FP16. Mariem's preserved provider is selectable
with `--depth-backend colleague --depth-size 420`.

Physical webcam smoke (`--depth-size native`, FP32, 10 seconds): capture
31.18 Hz, depth 9.24 Hz, CUDA normals 8.76 Hz, temporal geometry 1.76 Hz,
hands 30.14 Hz. A follow-up run measured CUDA normals 8.43 Hz and the
lightweight live temporal-confidence map 8.43 Hz, while the full CPU temporal
geometry remained 1.37 Hz. The live normals and mode-4 panels consume the
independent fast stream; full temporal diagnostics remain available for the
P4 contract without blocking them. The camera/display cadence is independent
from all slower workers; each worker publishes only its newest completed state.
XYZ now samples that same fast geometry stream instead of stale full-temporal
geometry; it remains empty by design when no physical hand is detected. Native
depth and XYZ use EMA smoothing, and mode 6 overlays the fixed camera frame
(+X right, +Y down, +Z forward) with each hand's coordinates.
Mode 2/3/4 consume the same postprocessed native depth state, CUDA normals, and
fast temporal-confidence state; mode 5 uses the hand tracker's One-Euro output.

============================================================
SCOPE GUARD
============================================================

P4 code modified: NO

P4 functionality implemented: NO

Lighting modified: NO

Specular modified: NO

Shadows modified: NO

Volumetrics modified: NO

Expected: all NO

P123→P4 dependency remaining: legacy `geometry/__init__.py` exports/imports existing P4 modules; the new P123 runtime and hand-depth path do not call them. Person 4 remains a black box.

============================================================
MODE 1 — CAMERA
============================================================

Physical device: `/dev/video0` via V4L2/OpenCV

Resolution: 640x480

Capture Hz: 14.996 Hz measured in the current run (device reports 30 FPS but delivered ~15 Hz)

Unique frames: 154 / 154

Motion proof: FAIL — no deliberate camera movement was supplied

Timestamp correctness: PASS (`time.monotonic()` at successful read)

Capture IDs: physical sequence IDs preserved

Processing IDs: separate contiguous IDs in temporal processing

Result: NOT READY

============================================================
MODE 2 — DEPTH
============================================================

Provider: Mariem's vendored Depth Anything V2 module (`MariemDepthProvider`)

Input resolution: 192x192 gate input

Inference p50: 61.60 ms

Inference p95: 74.13 ms

Actual depth Hz: 14.66 Hz

Near depth: 2.8887 (un-staged run)

Far depth: 2.5122 (un-staged run)

Forward-Z verified: NO — the required same-object near→far staged interaction was not performed

Static noise before: NOT HARDWARE VALIDATED

Static noise after: NOT HARDWARE VALIDATED

Scale breathing before: NOT HARDWARE VALIDATED

Scale breathing after: session-scale EMA implemented; physical foreground-entry test pending

Edge quality: NOT HARDWARE VALIDATED visually

Depth reliability: gradient-derived `depth_reliability`, invalid pixels zeroed

Color palette: dedicated warm-near / cool-far palette

Near color: warm red/yellow

Far color: cool blue/purple

Visual inspection: NOT RUN

Result: NOT READY

============================================================
MODE 3 — NORMALS
============================================================

Algorithm: existing edge-aware/multi-scale `estimate_normals`

Multi-scale: existing configurable normal mode retained

Planar angular jitter before: NOT HARDWARE VALIDATED

Planar angular jitter after: NOT HARDWARE VALIDATED

Edge bleed before: NOT HARDWARE VALIDATED

Edge bleed after: NOT HARDWARE VALIDATED

Temporal normal jitter: NOT HARDWARE VALIDATED

Visualization convention: R=Nx, G=Ny, B=Nz, [-1,1]→[0,255], invalid black

Visual inspection: NOT RUN

Result: NOT HARDWARE VALIDATED

============================================================
TEMPORAL
============================================================

Capture IDs observed: physical IDs may skip under latest-only processing

Processing IDs observed: contiguous 0,1,2,… per temporal observation

Unexpected resets: 0 in the updated temporal regression path

Raw depth jitter: 0.1584 (normalized units)

Stabilized depth jitter: 0.1718 (no deliberate motion; improvement not inferred)

Moving-object ghosting: NOT HARDWARE VALIDATED

Disocclusion: PASS for mask-shape/engine invariant; physical reveal test pending

Temporal Hz: 1.87 Hz in the current gate run

Result: NOT HARDWARE VALIDATED

============================================================
HANDS
============================================================

Backend: `mediapipe-tasks`, `models/hand_landmarker.task`

0 hand: NOT HARDWARE VALIDATED

1 hand: NOT HARDWARE VALIDATED

2 hands: NOT HARDWARE VALIDATED

Stable IDs: software invariant covered; physical multi-hand test pending

Detector Hz: 15.28 Hz (previous physical run)

Effective tracking Hz: 27.61 Hz (previous physical run)

LK intermediate motion: software regression PASS; no physical hand presented

Velocity correctness: software regression PASS

Result: NOT READY

============================================================
XYZ
============================================================

UV domain conversion: explicit camera-UV→depth-UV mapping

Depth age threshold: 120 ms

X test: NOT HARDWARE VALIDATED

Y test: NOT HARDWARE VALIDATED

Z test: NOT HARDWARE VALIDATED

XYZ Hz: none (no hand samples)

XYZ age p50: NOT HARDWARE VALIDATED

XYZ age p95: NOT HARDWARE VALIDATED

Result: NOT READY

============================================================
P4 INPUT CONTRACT
============================================================

RGB correspondence: source capture ID retained

Camera model: `CameraModel` carried explicitly

Depth: `DepthState`/`GeometryState` with source frame provenance

XYZ: `HandXYZ` camera-space tuple with confidence

Normals: carried by `GeometryState`

Confidence: depth reliability and hand confidence metadata

Frame provenance: capture IDs and monotonic timestamps

Temporal target: contiguous processing ID plus physical source ID

Hands: tuple of real tracked-hand states only

Ages: depth/hand/XYZ age metadata

Contains rendering: NO

Contract: PASS

============================================================
P4 OBSERVATIONAL BLOCKERS — NO CODE CHANGES
============================================================

Diffuse:

P123 inputs valid: NOT AUDITED

Observed output: NOT RUN

Blocker: none asserted; P4 observation is deferred until P123 physical gates pass.

Specular:

P123 inputs valid: NOT AUDITED

Observed output: NOT RUN

Blocker: none asserted; P4 observation is deferred until P123 physical gates pass.

Dynamic Shadows:

P123 inputs valid: NOT AUDITED

Observed output: NOT RUN

Blocker: none asserted; P4 observation is deferred until P123 physical gates pass.

============================================================
TESTS
============================================================

Unit passed: 199

Unit failed: 0

Physical gates passed: geometry; temporal structural gate

Physical gates failed: camera motion proof; depth direction; hands; XYZ

Visual physical gates passed: none

Not hardware validated: staged depth semantics, visual depth/normal quality, temporal motion/ghosting, 0/1/2-hand behavior, physical XYZ motion

============================================================
PERFORMANCE
============================================================

Camera Hz: 14.996 measured current run

Depth Hz: 14.66 measured current run

Geometry Hz: 4.30 measured current geometry run; 1.54 in asynchronous runtime smoke

Temporal Hz: 1.87 measured current run

Hand Hz: 15.28 detector / 27.61 effective (previous physical run)

XYZ Hz: none without a hand

Depth age p95: 131.76 ms

Geometry age p95: NOT MEASURED by the gate

Hand age p95: NOT MEASURED by the gate

============================================================
REMAINING P1 ISSUES
============================================================

1. Complete the visible 0-hand, 1-hand, 2-hand, cross, and separation stages.

2. Repeat XYZ with deliberate X/Y/Z motion and verify real two-hand IDs.

============================================================
REMAINING P2 ISSUES
============================================================

1. Run `physical_camera_gate --gate depth --interactive` and complete the near/far protocol.

2. Benchmark depth input sizes and complete physical static-noise, edge, and scale-breathing measurements.

============================================================
REMAINING P3 ISSUES
============================================================

1. Supply real camera calibration instead of the explicitly labelled approximate model.

2. Perform physical wall/edge/moving-object temporal and normal visual acceptance tests.

============================================================
P4 BLOCKERS — DO NOT MODIFY
============================================================

1. P4 diffuse/specular/shadow outputs were not observationally audited because their upstream physical gates are incomplete.

2. Existing P4 implementation remains unchanged and outside this branch.

============================================================
FINAL VERDICT
============================================================

Camera: NOT READY

Depth: NOT READY

Depth visualization: NOT READY (visual inspection pending)

Normals: NOT READY (physical quality pending)

Temporal: NOT READY (motion/ghosting pending)

One hand: NOT READY

Two hands: NOT READY

XYZ: NOT READY

P123 asynchronous runtime: NOT READY (runtime executes, but measured camera cadence is ~15 Hz and no physical hand samples were produced)

P123 input contract: READY (renderer-independent contract and tests pass)
