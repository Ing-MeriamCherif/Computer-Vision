# P1/P2/P3 LIVE PHYSICAL CAMERA REMEDIATION REPORT

Starting main commit: `e5b1fec`

Working branch: `bug-fixes/live-p123-physical-camera-remediation`

Ending commit: code commit recorded below; report commit follows it

============================================================
SCOPE GUARD
============================================================

P4 files modified: NO

P4 algorithms implemented: NO

Lighting implemented/modified: NO

Shadows implemented/modified: NO

Volumetrics implemented/modified: NO

Renderer implemented/modified: NO

Any P4 blockers discovered: Existing native renderer/lighting path remains outside this branch and was intentionally not exercised or changed. The remediation ends at the renderer-independent P4 input contract.

============================================================
GATE 0 — PHYSICAL CAMERA
============================================================

Executed: YES

Input: PHYSICAL

Device: `/dev/video0`

Backend: V4L2/OpenCV

Resolution: 640x480 negotiated

Requested FPS: 30

Actual capture Hz: 29.79 (10.5 s run)

Unique frames: 303

Capture failures: 0

Motion proof: FAIL — the operator did not perform the required move-left/right stage; frames were unique but the measured MAD did not exceed the motion threshold.

Result: FAIL (physical camera is open and producing frames; motion stage remains unverified)

============================================================
GATE 1 — LIVE DEPTH
============================================================

Executed: YES

Provider: `DepthAnythingProvider` (local checkpoint, FP16-capable path)

Depth convention: forward-Z, larger value means farther

Convention physically verified: NO — no near/far object movement was performed during the run.

Near value: 2.5892 (first-third center median)

Far value: 2.6079 (last-third center median)

Larger means: NOT PHYSICALLY VERIFIED

Canonical forward-Z conversion: PASS (inverse-depth model output is converted to forward-Z before session normalization)

Actual depth update Hz: 14.98

Inference p50: 59.09 ms

Inference p95: 78.21 ms

Depth age p50: 82.87 ms

Depth age p95: 107.47 ms

Scale breathing test: PASS in software (session-scale EMA is stable); physical near/far validation pending.

Result: FAIL / NOT HARDWARE VALIDATED for physical depth semantics

============================================================
GATE 2 — LIVE P3 GEOMETRY
============================================================

Executed: YES

Calibration: APPROXIMATE (no real calibration file supplied)

Depth source frame preservation: PASS

XYZ convention: PASS

Pz≈depth: PASS

Normals: PASS

Edge handling: PASS

CPU/GPU agreement: PASS (0.0° median normal angle on the measured overlap)

Geometry update Hz: 4.45

Result: PASS for the live geometry gate with approximate calibration; not a calibrated metric-quality claim.

============================================================
GATE 3 — LIVE TEMPORAL P3
============================================================

Executed: YES

Canonical temporal engine used: YES

Flow provider: OpenCV Farneback

Raw depth jitter: 0.1101 (normalized depth units)

Stabilized depth jitter: 0.1101 (no operator motion was supplied, so no improvement can be inferred)

Raw normal jitter: NOT MEASURED by the current gate

Stabilized normal jitter: NOT MEASURED by the current gate

Disocclusion rejection: PASS

Moving-object ghosting: NOT HARDWARE VALIDATED (no moving-object stage performed)

Frame contract: PASS

Temporal output Hz: 2.67

Result: NOT HARDWARE VALIDATED for motion/ghosting quality; frame-contract and engine checks pass.

============================================================
GATE 4 — LIVE P1 HANDS
============================================================

Executed: YES

Backend: `mediapipe-tasks` using `models/hand_landmarker.task`

Physical 0-hand test: NOT HARDWARE VALIDATED

Physical 1-hand test: NOT HARDWARE VALIDATED

Physical 2-hand test: NOT HARDWARE VALIDATED

True simultaneous hands detected: 0 (no hands presented during the run)

Detector Hz: 15.28

Effective tracking Hz: 27.61

Skipped-frame LK: PASS

Stable IDs: PASS

Result: FAIL — the real backend loaded, but the required one/two-hand physical stages were not performed.

============================================================
GATE 5 — LIVE P1/P2/P3 XYZ
============================================================

Executed: YES

Hand UV→depth UV: PASS

Same/compatible frame association: PASS

Hand depth age bounded: PASS

Robust hand depth: FAIL (no hand samples)

X physical motion: NOT HARDWARE VALIDATED

Y physical motion: NOT HARDWARE VALIDATED

Z physical near/far: NOT HARDWARE VALIDATED

XYZ update Hz: 4.01 state-loop rate

Result: FAIL — no physical hand was present, so no XYZ sample could be emitted.

============================================================
P4 HANDOFF CONTRACT
============================================================

Defined: YES

Contains rendering code: NO

Expected: NO

Provides: RGB, CameraModel, GeometryState, TrackedHand XYZ, confidence, frame IDs, timestamps, age metadata

Contract tests: PASS

============================================================
MOCK/SYNTHETIC TESTS
============================================================

Unit tests passed: 195

Unit tests failed: 0

Smoke tests passed: 0 separately classified smoke tests

Smoke tests failed: 0

Do any synthetic tests claim hardware readiness? NO

Expected: NO

============================================================
PHYSICAL HARDWARE STATUS
============================================================

Camera hardware available: YES

Physical gates executed: camera, depth, geometry, temporal, hands, xyz

Physical gates passed: geometry; temporal contract/engine checks; depth inference execution (semantic direction not verified)

Physical gates failed: camera motion proof; depth physical direction; hands; XYZ

Unexecuted gates: None (operator motion/hand stages were not supplied)

============================================================
REMAINING P1 ISSUES
============================================================

#1 Run the hands gate while visibly presenting 0, then 1, then 2 simultaneous hands.

#2 Repeat XYZ with deliberate X/Y movement and near/far Z movement to validate depth fusion and two-hand behavior.

============================================================
REMAINING P2 ISSUES
============================================================

#1 Perform the instructed near/far physical object stage to verify that larger forward-Z values truly mean farther.

#2 Improve live throughput beyond the measured ~15 Hz depth update rate if the target requires camera-rate depth output; retain measured completion and age metrics.

============================================================
REMAINING P3 ISSUES
============================================================

#1 Supply real camera calibration; current geometry gate uses an explicitly labelled approximate model.

#2 Repeat temporal validation with deliberate camera/object motion to measure normal jitter reduction and moving-object ghosting.

============================================================
P4 BLOCKERS OBSERVED — NOT MODIFIED
============================================================

#1 The existing renderer/lighting implementation is outside this corrective branch and was not tested as part of P1/P2/P3.

#2 Person 4 must consume the contract only after the operator-completed hand and depth physical stages pass.

============================================================
FINAL VERDICT
============================================================

Physical camera pipeline: NOT READY (camera is available, but motion proof is incomplete)

P2 live depth: NOT HARDWARE VALIDATED

P3 live geometry: READY for the tested approximate-calibration invariants

P3 temporal: NOT HARDWARE VALIDATED

P1 one-hand: NOT HARDWARE VALIDATED

P1 two-hand: NOT HARDWARE VALIDATED

P1/P2/P3 XYZ contract: NOT HARDWARE VALIDATED

Ready to hand off to Person 4: NO
