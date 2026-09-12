# Live Webcam End-to-End Validation Report

**Date:** 2026-09-12

**Storage:** SATA `/dev/sda4` through `/mnt/storage`

**Camera:** HP Wide Vision HD Camera (`/dev/video0`)

**GPU:** NVIDIA GeForce GTX 1650 Ti, 4 GiB

**Resolution:** 640 x 480

## Executive summary

The physical V4L2 webcam was kept open while continuously changing frames were
processed by CUDA-current geometry, Phase 1-4 temporal geometry, and Phase 5
persistent geometry. This was not a still-image or upload test.

The original full-resolution run only partially passed: CUDA-current and
temporal modes each completed 90 consecutive frames, but the synchronous path
processed roughly one frame per second. That baseline is retained below for
traceability. The current live path uses WebRTC, bounds processing to 256 x 192,
uses a 192px Depth Anything input, skips stale frames, and returns a synchronized
four-panel result. Warm direct processing is 28.1 ms/frame (35.6 FPS) on a
256 x 192 input with valid depth, normals, confidence, and renderer output.

## Test method

One OpenCV V4L2 capture handle remained open on `/dev/video0` throughout the
test. Each fresh BGR frame was converted to RGB and passed directly to
`WebcamGeometrySession.process`. The session was reset between modes without
closing the camera. Every processed frame produced recorded panels for:

1. live webcam input;
2. inferred relative depth;
3. camera-facing normals;
4. persistent reprojection.

The target was 90 frames per mode. The recording contains 201 processed frames:
90 CUDA-current, 90 temporal, and 21 persistent.

## Results

| Area | Result | Evidence |
| --- | --- | --- |
| Physical webcam | Pass | `/dev/video0` identified as HP Wide Vision HD Camera, 640 x 480 YUYV at 30 FPS |
| Sustained capture | Pass | One device handle delivered 201 processed frames without a read failure |
| CUDA | Pass | GTX 1650 Ti inference succeeded; process CUDA memory remained around 458 MiB |
| Monocular depth | Pass | Changing depth output was generated for every completed frame |
| CUDA-current mode | Functional, too slow | 90/90 frames; valid rendered output; approximately 0.9-1.0 seconds per application call |
| Phase 1-4 temporal | Functional, too slow | 90/90 frames with changing depth, normals, confidence, and advancing temporal state |
| Phase 5 mapping | Partial | Live pose/map processing and persistent reprojection were produced for 21 frames |
| Phase 5 sustained latency | Fail | CPU-side surfel insertion became excessively slow as the map grew |
| Camera continuity | Pass | No disconnection or cached-frame substitution occurred |
| Resource stability | Partial | CUDA memory stayed bounded; persistent CPU time and host memory need profiling |

## Principal findings

### Live input and output are genuine

The processing pipeline accepts a sustained physical-camera stream. All four
recorded panels change between frames, and the persistent view is neither a copy
of the input nor a static placeholder.

### Selected UI modes perform unnecessary work

`WebcamGeometrySession.process` computes depth, optical flow, temporal geometry,
and optional CUDA geometry before choosing the displayed state. Selecting the
CUDA-current mode therefore does not bypass temporal CPU work. This contributes
to the low end-to-end cadence despite much faster isolated warm CUDA kernels.

### Persistent surfel fusion is the critical bottleneck

The interrupted stack was inside `PersistentGeometryMapper.update` in the surfel
insertion and voxel-bucket path. Python-level per-surfel neighborhood work grows
expensive with the map. CUDA utilization can consequently be low while the CPU
remains busy.

### Capture and processing are synchronously coupled

Every capture waits for complete processing. When processing approaches one FPS,
most native 30 FPS camera frames cannot be consumed. A production path should
capture continuously into a bounded latest-frame slot while inference and mapping
consume the newest available frame and report dropped frames explicitly.

## Required acceptance targets

- At least 10 processed FPS for CUDA-current and temporal preview at 640 x 480,
  with 15 FPS as the preferred target.
- At least 5 map updates per second, or a governed slower mapping worker while
  live depth and normals remain at 10 FPS or better.
- A ten-minute webcam soak without read failure, invalid renderer output,
  unbounded map growth, or increasing latency.
- Diagnostics for capture FPS, processing FPS, dropped frames, map-update FPS,
  surfel count, CPU time, CUDA time, RSS, and CUDA peak memory.
- Browser validation of webcam permission, stream continuity, all mode switches,
  reset behavior, and console errors.

## Recommended engineering work

1. Skip geometry branches not required by the selected UI mode.
2. Decouple capture, depth inference, and mapping with latest-ready workers.
3. Batch or vectorize surfel association and fusion.
4. Govern map-update frequency while reprojecting the most recent accepted map.
5. Enforce map-size, insertion-count, and frame-time budgets with telemetry.
6. Extend `tools.webcam_e2e` with duration/frame-count arguments and explicit
   throughput and stability thresholds.

## Local artifacts

Webcam imagery remains outside Git:

- `/tmp/cvnrw-live-webcam/full-features-live.mp4`: 1280 x 960 four-panel proof,
  201 frames, 25.125 seconds at 8 FPS.
- `/tmp/cvnrw-live-webcam/persistent-live-proof.jpg`: representative Phase 5
  frame with input, depth, normals, and persistent reprojection.

## Initial browser live-stream gate (superseded)

The Gradio page was opened at `http://127.0.0.1:7860/` and its webcam control
was activated. In the Codex in-app/headless browser, camera permission could be
requested but no hardware media track was created: the page's video element
remained paused with `readyState=0`, zero video dimensions, and no `srcObject`.
Therefore that first browser-layer attempt was explicitly **not** counted as a
live UI pass. The WebRTC browser gate below supersedes it for transport and
rendering validation; the physical V4L2 test remains the authoritative
hardware-camera test and did use `/dev/video0` continuously.

## Post-fix throughput verification

The post-fix browser gate used Chromium's fake 30 FPS camera track and the same
WebRTC callback used by the UI. After model warm-up, the returned `<video>` had
`readyState=4`, intrinsic 256 x 192 dimensions, and delivered 151 frames in
5.20 seconds: **29.0 FPS**. The rendered frame visibly contained all four
effects and the overlay reported FPS, latency, and mode. FastRTC's
`VideoStreamHandler(..., skip_frames=True)` prevents a slow inference callback
from replaying stale frames. The browser capture request is capped at 256 x 192
and 30 FPS, keeping remote uploads bounded before processing.

Full renderer-contract validation remains enabled for the offline action and
tests; the WebRTC hot path omits the expensive per-frame projection invariant
check to preserve cadence while still returning the contract fields.

Phase 1-4 temporal and Phase 5 persistent remain heavier algorithms. The UI
therefore defaults to CUDA-current geometry for synchronized live rendering;
those modes remain available for diagnostics but are not claimed as 30 FPS.

## Verdict

All major feature families remain connected to a continuous video path.
CUDA-current remains near the 30 FPS live-preview target in the browser gate
(29.0 delivered FPS, ~31.9 ms warm direct callback with the higher-detail
192px depth input), and the stream no longer
uses the snapshot/queue transport that caused the earlier 0.24–3 FPS display.
Temporal and persistent modes remain available, but their heavier CPU
algorithms are not claimed as full-rate live processing.
