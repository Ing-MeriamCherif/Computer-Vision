# P123 MODE-BY-MODE LIVE REMEDIATION REPORT

Starting commit: `e5b1fec`

Ending commit (code): `5b882a9`; report finalization follows in the report-only commit

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

Provider: `DepthAnythingProvider`

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
