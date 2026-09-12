```markdown
# Cahier de Charge — NRW 8th Edition AI & Vision Challenge

**Project:** Spatial Perception & Real-Time Dynamic Relighting System
**Competition:** National Robotics Week — 8th Edition
**Organizers:** IEEE INSAT Student Branch & IEEE RAS INSAT Chapter
**Document Version:** 1.0
**Date:** 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Challenge Requirements Mapping](#2-challenge-requirements-mapping)
3. [System Architecture](#3-system-architecture)
4. [Module Specifications](#4-module-specifications)
5. [Testing & Validation Plan](#5-testing--validation-plan)
6. [Hardware & Software Environment](#6-hardware--software-environment)
7. [Performance Targets](#7-performance-targets)
8. [Deliverables](#8-deliverables)
9. [Timeline](#9-timeline)
10. [Anti-Cheat Compliance](#10-anti-cheat-compliance)
11. [Risk Register](#11-risk-register)
12. [Team Roles](#12-team-roles)

---

## 1. Project Overview

### 1.1 Context

The 8th Edition of National Robotics Week (NRW) challenges teams to build a **real-time spatial perception system** that transforms a flat 2D RGB camera feed into a responsive, mathematically sound 3D environment — without dedicated LiDAR hardware.

The system must:
- Estimate **pixel-accurate 3D depth geometry** from a monocular camera.
- Extract **surface normal vectors** in real time.
- Render a **user-controlled virtual light source** that dynamically illuminates the scene.
- Cast **realistic dynamic shadows** based on relative light positions.
- React to **real-time hand gestures** for 3D light placement (X, Y, Z axes).

### 1.2 Vision

Prove that modern monocular depth and segmentation models can run **synchronously with graphics pipelines at ultra-low latency**, delivering a seamless interactive experience on commodity hardware.

### 1.3 Primary Objective

Develop a **high-throughput, real-time computer vision system** that:
1. Takes a standard camera feed.
2. Extracts pixel-accurate 3D depth geometry.
3. Renders a user-controlled virtual light source.
4. Casts realistic dynamic shadows.
5. Reacts seamlessly to real-time hand gestures (X, Y, Z).

### 1.4 Design Philosophy

**Do not over-engineer.** Build the simplest working implementation first. Add complexity only when a specific, measurable problem appears.

**Anti-patterns to avoid:**
- Do not implement D2NT before testing a Sobel filter on the depth map.
- Do not convert to TensorRT before measuring PyTorch/ONNX latency.
- Do not integrate multi-task models before the separate pipeline works end-to-end.
- Do not build a WebRTC demo platform before the local pipeline runs at ≥30 FPS.

---

## 2. Challenge Requirements Mapping

### 2.1 Technical Progression Ladder

| Level | Objective | Visual Benchmark | Priority |
|:---|:---|:---|:---|
| **Level 01** | Extract continuous 3D surface normal vectors N(x,y) from depth gradients in real time. | Live depth-to-normal map visualization running synchronously with camera feed. | **Critical** |
| **Level 02** | Apply real-time Lambertian diffuse and specular shading models onto 3D surface geometry. | Realistic scene illumination responsive to interactive virtual point light. | **Critical** |
| **Level 03** | Integrate real-time hand-tracking algorithms to control light placement using physical gestures. | Smooth spatial control across X, Y, Z axes. | **Critical** |
| **Level 04** | Project geometry-aware occlusion shadows cast behind foreground subjects. | Realistic, dynamic shadow vectors adjusting smoothly with gesture movement. | **High** |
| **Level 05** | Support multi-hand gesture tracking for multiple independent light sources. | Complex light interactions with overlapping shadow paths and volumetric haze. | **Medium** |
| **Level ∞** | Push beyond defined levels with creative extensions. | Jury-defined. | **Bonus** |

### 2.2 Evaluation Protocol

**Phase 01 — Pre-Selection Demo Submission:**
- Teams submit a short, continuous, unedited video demo showing the live system.
- Evaluators screen based on: real-time rendering quality, hand gesture responsiveness, dynamic shading/shadow precision, and framerate fluidity.
- Top-ranked teams with seamless real-time interaction advance as finalists.

**Phase 02 — Live Stage Defense & Jury Presentation:**
- Live interactive demonstration on stage.
- Jury physically tests real-time gesture control (X, Y, Z light manipulation), specular shading, and dynamic shadow projection.
- Technical presentation covering vision pipeline, depth/normal estimation methods, latency optimizations, and architectural choices.
- **Live Verification & Anti-Cheat:** Live demo must match pre-selection video. Pre-rendered footage, hardcoded depth maps, or unhandled frame drops → immediate disqualification.

---

## 3. System Architecture

### 3.1 High-Level Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                      CAMERA FEED (RGB)                           │
└──────────────────────────┬───────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
           ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│  HAND TRACKING      │         │  DEPTH ESTIMATION   │
│  (MediaPipe Hands)  │         │  (DepthAnythingV2)  │
│  21 landmarks/hand  │         │  Monocular depth    │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           │                               ▼
           │                    ┌─────────────────────┐
           │                    │  NORMAL TRANSLATOR  │
           │                    │  (D2NT / Sobel)     │
           │                    │  N(x,y) per pixel   │
           │                    └──────────┬──────────┘
           │                               │
           ▼                               │
┌─────────────────────┐                    │
│  LIGHT POSITION     │                    │
│  Back-project palm  │                    │
│  Sample depth at    │                    │
│  landmark → (X,Y,Z) │                    │
└──────────┬──────────┘                    │
           │                               │
           └───────────────┬───────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  LIGHT VECTOR COMPUTE   │
              │  L = normalize(L_pos    │
              │      - P_pixel)         │
              │  (per-pixel, parallel)  │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  LAMBERTIAN DIFFUSE     │
              │  I = max(dot(N,L),0)    │
              │  + Specular (Blinn-Phong)│
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  SHADOW MAPPING         │
              │  Light POV depth pass   │
              │  Per-pixel comparison   │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  FINAL RENDER           │
              │  + Intensity modulation │
              │  + Color grading        │
              └─────────────────────────┘
```

### 3.2 Data Flow Summary

| Stage | Input | Output | Parallelism |
|:---|:---|:---|:---|
| Hand Tracking | RGB frame | 21 landmarks × N hands | CPU/GPU |
| Depth Estimation | RGB frame | Depth map D(x,y) | GPU (CNN/ViT) |
| Normal Translation | Depth map | Normals N(x,y) | Patch-level GPU |
| Light Position | Landmark + depth | 3D point (X,Y,Z) | Single op |
| Light Vector | Light pos + pixel 3D | Direction L | Per-pixel GPU |
| Diffuse Shading | N, L, albedo | Intensity | Per-pixel GPU |
| Specular Shading | N, H, V | Highlight | Per-pixel GPU |
| Shadow Mapping | Light POV depth | Shadow factor | Per-pixel GPU |
| Final Render | All above | RGB frame | Full-screen GPU |

### 3.3 GPU-Resident Architecture (Target)

To achieve ultra-low latency, all tensors must remain on the GPU. Data transfer between CPU and GPU is the primary bottleneck. The target architecture:

- Depth model runs on GPU (TensorRT / ONNX Runtime GPU).
- Normal translation runs as a GPU shader or CUDA kernel.
- Lighting and shadow computation run as fragment shaders.
- Only the final rendered frame is copied to CPU for display.

---

## 4. Module Specifications

### 4.1 Module 1 — Hand Tracking & Light Control

**Objective:** Detect hand landmarks in real time, back-project to 3D, and use as virtual light source position.

**Model:** MediaPipe Hands (industry standard).

**Specifications:**

| Parameter | Value |
|:---|:---|
| Landmarks per hand | 21 |
| Max hands | 2 (for Level 05 multi-light) |
| Input resolution | 480p / 720p |
| Target latency | < 15 ms |
| Target FPS | ≥ 30 |
| Temporal filter | EMA (α = 0.3–0.5) |

**Back-Projection Formula:**

Given landmark pixel `(u, v)` and depth `z_c` sampled from depth map:

```
x_c = z_c · (u - u₀) / f_x
y_c = z_c · (v - v₀) / f_y
```

Where `(u₀, v₀)` is the principal point and `f_x, f_y` are focal lengths in pixels.

**Intensity Metric:**

```
intensity = I₀ / (1 + (d / d_ref)²)
```

Where `d = ||L_pos||` (distance from camera), `d_ref = 0.5 m`, `I₀ = 1.0`.

**Temporal Filtering:**

```python
light_pos_smoothed = α · light_pos_new + (1 - α) · light_pos_previous
```

**Color Filter Modulation:**

As intensity grows, modulate RGB channels differently to simulate warm point light:
- `R_out = R · intensity · 1.1`
- `G_out = G · intensity`
- `B_out = B · intensity · 0.9`

**Implementation Notes:**
- Use palm center (landmark index 9) as the light position reference.
- Sample depth with bilinear interpolation for sub-pixel accuracy.
- If depth at landmark is unreliable (edge discontinuity), apply a 5×5 median filter around the landmark.

---

### 4.2 Module 2 — Depth Estimation

**Objective:** Extract a per-pixel depth map from a single RGB frame in real time.

**Candidate Models:**

| Model | Params | GFLOPs | Latency | Accuracy | Recommendation |
|:---|:---|:---|:---|:---|:---|
| **DepthAnythingV2-Small** | 24.7M | 53.6 @ 420×560 | ~35 FPS (TensorRT FP16, 1280×720) | High | **Primary choice** |
| **Flex-Nano** | ~1.5M | 0.7 | 37.6 FPS mobile | Moderate | **Fallback for constrained HW** |
| **DepthPro** | ~100M+ | ~50–100 | < 1s (unoptimized) | Highest (0.0635 AbsRel) | **If metric depth needed** |
| **ZoeDepth** | — | — | ~2.2 FPS desktop | Lower | Not recommended |

**Benchmark Protocol:**
1. Test all models at 336×252, 420×560, and 518×518.
2. Measure median latency (10 runs per config).
3. Measure VRAM usage.
4. Evaluate depth quality visually and with AbsRel if ground truth available.

**Optimization Ladder (apply only if needed):**

1. **PyTorch baseline** — measure first.
2. **ONNX Runtime GPU** — if PyTorch is slow.
3. **TensorRT FP16** — if ONNX is still slow.
4. **TensorRT INT8** — only if FP16 is insufficient.

**Reference performance data:**
- DepthAnythingV2-Small: FP16 quantization yields up to **66% speed improvement** on Jetson Nano.
- TensorRT is **2–2.7× faster** than ONNX Runtime.
- ONNX Runtime is **1.4× faster** than PyTorch.

**Local Depth Refinement (optional):**

To sharpen depth edges without global computation, apply a local gradient accumulator:

```
If |∇D| > τ:
    D_corrected(x,y) = D(x,y) + λ · Σ_Ω |D(x,y) - D(x',y')|
```

Where `Ω` is a local neighborhood. This mimics D2NT's discontinuity-aware behavior at minimal cost.

**Temporal Smoothing:**

```
D_t = β · D_t + (1 - β) · D_{t-1},  β ≈ 0.7–0.8
```

---

### 4.3 Module 3 — Surface Normal Translation

**Objective:** Convert depth map D(x,y) into surface normal map N(x,y) in real time.

**Primary Method:** D2NT (Depth-to-Normal Translator)

| Metric | D2NT | Sobel Filter |
|:---|:---|:---|
| Latency | 1.82 ms | 0.1–0.5 ms |
| Angular error | 0.89° | Higher at edges |
| Edge handling | Discontinuity-aware (DAG filter) | Poor (blurs edges) |
| Recommendation | **Primary** | Fast preview / fallback |

**Sobel Baseline (implement first):**

```python
grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)

nx = -fx * grad_x
ny = -fy * grad_y
nz = np.ones_like(depth_map)

norm = sqrt(nx² + ny² + nz²)
N = (nx/norm, ny/norm, nz/norm)
```

**Normal Orientation Correction:**

After normal estimation, apply a rule: if `dot(N, view_direction) < 0`, flip `N = -N`. This fixes occasional orientation errors.

**Temporal Filtering on Normals:**

Apply EMA across frames to reduce flickering:

```
N_t = γ · N_t + (1 - γ) · N_{t-1},  γ ≈ 0.6–0.7
```

**Upgrade Decision Rule:**

Implement D2NT only if:
- Sobel normals produce visible artifacts in the final shaded output.
- The artifacts are severe enough to affect the jury's perception.
- The 1.82 ms latency overhead is acceptable.

---

### 4.4 Module 4 — Lighting & Shadow Rendering

**Objective:** Apply real-time Lambertian diffuse and specular shading, and project dynamic shadows based on light position.

#### 4.4.1 Light Vector Calculation

Per-pixel light direction:

```
L = normalize(L_pos - P_pixel)
```

Where:
- `L_pos` = 3D light position from hand tracking (Module 1).
- `P_pixel` = 3D position of the pixel, reconstructed from depth map.

This is a per-pixel operation, fully parallelizable.

#### 4.4.2 Lambertian Diffuse

```
I_diffuse = k_d · I_light · max(dot(N, L), 0)
```

Where:
- `k_d` = material albedo (use original RGB as approximation).
- `I_light` = intensity from Module 1.

#### 4.4.3 Specular (Blinn-Phong)

```
H = normalize(L + V)        // halfway vector
I_specular = k_s · I_light · max(dot(N, H), 0)^n
```

Where:
- `V` = view direction (toward camera).
- `n` = shininess exponent (32–128).

#### 4.4.4 Combined Shading

```
I_final = I_ambient + I_diffuse + I_specular
```

#### 4.4.5 Shadow Mapping

**Standard Approach:**
1. Render scene depth from light's point of view → shadow map.
2. During main render, compare pixel distance from light to shadow map value.
3. If pixel is farther → in shadow.

**Optimization:**
- Use **Shadow Cache** technique (210 FPS @ 2000m vs 160 FPS for cascaded shadow maps).
- For single point light: **cube shadow map** (6 faces, 512×512 each).
- Add **shadow bias** (0.001–0.005) to prevent self-shadowing artifacts.

**Implementation Order:**
1. **Fake shadow** (screen-space distance darkening) — for Phase 1 testing.
2. **Basic shadow map** — 1 face, 512×512.
3. **Cube shadow map** — for point light.
4. **Shadow Cache** — only if performance drops.

---

### 4.5 Module 5 — Multi-Task Model Exploration (Research)

**Objective:** Evaluate whether a single multi-task model can replace the separate depth + normal + pose pipeline.

**Candidates:**

| Model | Outputs | Latency | Backends | Notes |
|:---|:---|:---|:---|:---|
| **Y-MAP-Net** | Depth, normals, human pose, segmentation, captions | ~20 ms (est.) | TF, TFLite, JAX, ONNX | Replaces 3 models in one pass |
| **M2H** | Segmentation, depth, edge, normals | ~33 ms (est.) | PyTorch | 30 FPS on RTX 3080 laptop |
| **UniDepth V2** | Depth only | Near-real-time | PyTorch | Efficient, consistent |
| **Metric3D v2** | Metric depth | Slower | PyTorch | Higher accuracy trade-off |

**Decision Rule:**

Adopt a multi-task model only if:
1. The separate pipeline works end-to-end at ≥30 FPS.
2. The multi-task model's latency is **lower** than the sum of separate models.
3. Its accuracy is **sufficient** for the challenge (no visible degradation).

**Do not** adopt multi-task models before the separate pipeline is stable.

---

## 5. Testing & Validation Plan

### 5.1 Test Harness — `test_runner.py`

**Single entry point for all testing:**

```bash
python test_runner.py --mode phase1      # Depth + hand tracking only
python test_runner.py --mode phase2      # Full pipeline with lighting
python test_runner.py --mode benchmark   # Run model benchmarks
python test_runner.py --mode profile     # Per-stage latency profile
python test_runner.py --mode record      # Record demo video
```

**Features:**
- Runs full pipeline with live camera.
- Logs latency per stage (min/avg/max/p95).
- Saves results as JSON to `results/`.
- Records video in record mode.
- Displays 2×2 grid: RGB+landmarks | Depth | Normals | Shaded.

### 5.2 Test Scenarios

#### Scenario A — Phase 1 Test

**Setup:** Sit at desk, good lighting, hand visible.

**Run:** `python test_runner.py --mode phase1`

**Pass Criteria:**
- FPS ≥ 25.
- Hand landmarks track smoothly.
- Depth map shows clear hand/background distinction.

**Failure Fixes:**

| Symptom | Fix |
|:---|:---|
| FPS < 20 | Reduce depth input to 336×252. |
| Landmarks jitter | Add EMA filter. |
| Depth map uniform | Check normalization. |
| Camera won't open | Change CAMERA_ID. |

#### Scenario B — Phase 2 Test

**Setup:** Same as A, move hand left/right/closer/farther.

**Run:** `python test_runner.py --mode phase2`

**Pass Criteria:**
- FPS ≥ 20.
- Light responds within 2–3 frames.
- No shadow artifacts.

#### Scenario C — Benchmark Mode

**Run:** `python test_runner.py --mode benchmark`

**Protocol:**
1. Load each depth model.
2. Run 50 frames of static scene.
3. Log median and p95 latency, VRAM.
4. Save comparison table.

#### Scenario D — Profile Mode

**Run:** `python test_runner.py --mode profile`

**Output Example:**

```
Stage          Avg (ms)   Min (ms)   Max (ms)   P95 (ms)
hand            12.3        8.1        28.4       22.1
depth           18.7       15.2        35.1       28.9
normals          2.1        1.8         4.2        3.5
lighting         5.4        4.1        12.3        9.8
total           38.5       29.2        80.0       64.3
```

**Use this to identify bottlenecks.** Optimize the largest stage first.

#### Scenario E — Record Mode

**Run:** `python test_runner.py --mode record`

**Recording Checklist:**
- [ ] Good lighting on hand.
- [ ] Move hand: left → right → closer → farther.
- [ ] Show 10 seconds of each level (01–04).
- [ ] No frame drops (FPS ≥ 25).
- [ ] Video is continuous and unedited.
- [ ] FPS counter visible.

### 5.3 Remote Demo Testing

**Architecture:** Phone browser ↔ WebRTC ↔ PC GPU pipeline.

**Latency Budget:**

| Stage | Latency |
|:---|:---|
| Phone camera capture | ~10 ms |
| WebRTC uplink | ~30–50 ms |
| Server processing | ~25 ms |
| WebRTC downlink | ~30–50 ms |
| **Total** | **~95–135 ms** |

**Setup:**
1. Run `uvicorn server:app --host 0.0.0.0 --port 8000` on PC.
2. Find PC local IP.
3. Open `http://<PC_IP>:8000` on phone (same Wi-Fi).
4. Allow camera access.

**No internet required.** Everything on local network.

---

## 6. Hardware & Software Environment

### 6.1 Hardware Requirements

| Component | Minimum | Recommended |
|:---|:---|:---|
| GPU | RTX 3060 (12GB) | RTX 4070 / 4080 |
| CPU | Ryzen 5 / i5 (6 cores) | Ryzen 7 / i7 (8+ cores) |
| RAM | 16 GB | 32 GB |
| Camera | 720p USB webcam | 1080p 60fps webcam |
| Phone | Any modern Android/iOS | — |
| Network | Local Wi-Fi (5GHz) | Same router, wired PC |

### 6.2 Software Stack

```bash
# Python 3.10 or 3.11 (not 3.12)
conda create -n nrw python=3.10
conda activate nrw

# PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core CV
pip install opencv-python mediapipe numpy scipy

# Depth model
pip install huggingface_hub timm

# ONNX / TensorRT (install AFTER PyTorch)
pip install onnx onnxruntime-gpu
# TensorRT: install via NVIDIA's official package

# Utilities
pip install tqdm matplotlib pandas
```

**Verification:**

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import mediapipe; print(mediapipe.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

### 6.3 Project Structure

```
nrw_challenge/
├── main.py                  # Entry point (launches test_runner)
├── test_runner.py           # Single entry point for all testing
├── hand_tracker.py          # MediaPipe wrapper
├── depth_estimator.py       # DepthAnythingV2 / FlexDepth wrapper
├── normal_translator.py     # Sobel / D2NT normal estimation
├── lighting.py              # Lambertian + specular shading
├── shadow.py                # Shadow mapping
├── utils.py                 # Camera intrinsics, back-projection, filters
├── server.py                # WebRTC remote demo server
├── client.html              # Phone browser client
├── requirements.txt
├── .env                     # Configuration
├── README.md
└── results/                 # Test outputs (JSON, video)
    ├── phase1_*.json
    ├── phase2_*.json
    ├── benchmark_*.json
    └── demo.mp4
```

### 6.4 Configuration — `.env`

```bash
CAMERA_ID=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=30
DEPTH_MODEL=vits
DEPTH_INPUT_SIZE=518
DEVICE=cuda
LOG_LEVEL=INFO
```

---

## 7. Performance Targets

### 7.1 Latency Budget

| Stage | Target (ms) | Maximum (ms) |
|:---|:---|:---|
| Hand tracking | 12 | 20 |
| Depth estimation | 18 | 25 |
| Normal translation | 2 | 5 |
| Lighting | 5 | 10 |
| Shadow mapping | 5 | 10 |
| **Total (serial)** | **42** | **70** |
| **Total (pipelined)** | **~25** | **~40** |

**Frame budget for 30 FPS:** 33.3 ms
**Frame budget for 60 FPS:** 16.6 ms

### 7.2 Quality Targets

| Metric | Target |
|:---|:---|
| Depth AbsRel (indoor) | < 0.10 |
| Normal angular error | < 5° |
| Hand landmark jitter | < 2 px std dev |
| Shadow smoothness | No visible flickering |
| Light response latency | < 3 frames |

### 7.3 Optimization Decision Tree

```
Is FPS ≥ 30?
├── YES → Ship it.
└── NO → Which stage is slowest?
    ├── Depth → ONNX → TensorRT FP16 → INT8
    ├── Hand → GPU delegate → reduce resolution
    ├── Normals → Sobel → D2NT (if quality issue)
    └── Lighting → shader → full-screen pass
```

---

## 8. Deliverables

### 8.1 Required Artifacts

| # | Artifact | Format | Deadline |
|:---|:---|:---|:---|
| 1 | **Source Code Repository** | Clean codebase, `requirements.txt` or `Dockerfile`, launches via `python main.py` | Before countdown |
| 2 | **Technical Architecture Brief** | System report: neural model selection, normal extraction formulas, temporal filtering, FPS optimization | Before countdown |
| 3 | **Pitch Deck** | Presentation for jury: pipeline, trade-offs, FPS benchmarks, live demo config | Before countdown |

### 8.2 Submission Instructions

**Email:** nrw8challenges@gmail.com
**Subject format:** `[Challenge Name] - [Team Name]`

### 8.3 Pre-Selection Video Requirements

- Short, continuous, unedited.
- Shows live system in action.
- Demonstrates real-time rendering quality.
- Shows hand gesture responsiveness.
- Shows dynamic shading/shadow precision.
- Shows framerate fluidity.

---

## 9. Timeline

### 9.1 Seven-Day Sprint Plan

| Day | Morning | Afternoon | Evening |
|:---|:---|:---|:---|
| **1** | Setup environment, verify CUDA/MediaPipe | Implement `hand_tracker.py`, test | Implement `depth_estimator.py`, test |
| **2** | Integrate hand + depth in `test_runner.py` | Run `--mode phase1`, fix issues | Run `--mode profile`, find bottlenecks |
| **3** | Implement `normal_translator.py` (Sobel) | Test normals visually | Implement `lighting.py` (diffuse) |
| **4** | Run `--mode phase2`, fix light direction | Add intensity + color filter | Add EMA temporal filter |
| **5** | Run `--mode benchmark` on all depth models | Decide final model | Optimize if needed (ONNX → TensorRT) |
| **6** | Implement shadow mapping (basic) | Run `--mode record`, review video | Fix frame drops |
| **7** | Build WebRTC remote demo | Test phone-to-PC streaming | Record final pre-selection video |

### 9.2 Milestones

| Milestone | Date | Deliverable |
|:---|:---|:---|
| M1: Environment ready | Day 1 | Working camera + MediaPipe |
| M2: Phase 1 pipeline | Day 2 | Depth + hand visualization at 30 FPS |
| M3: Lighting pipeline | Day 4 | Interactive light with diffuse shading |
| M4: Model selection | Day 5 | Benchmark report, final model chosen |
| M5: Full pipeline | Day 6 | Shadows + record mode working |
| M6: Submission ready | Day 7 | Video + code + brief + pitch deck |

---

## 10. Anti-Cheat Compliance

### 10.1 Rules

| Rule | Compliance Strategy |
|:---|:---|
| No pre-rendered footage | All outputs computed live from camera feed. |
| No hardcoded depth maps | Depth always from neural model inference. |
| No unhandled frame drops | Maintain ≥ 30 FPS; monitor with FPS counter. |
| Live demo matches pre-selection | Same code, same pipeline, same performance. |

### 10.2 Verification Checklist

- [ ] FPS counter visible in demo.
- [ ] No pre-recorded video in pipeline.
- [ ] Depth computed from camera feed only.
- [ ] Hand tracking computed from camera feed only.
- [ ] Lighting responds to live gestures.
- [ ] Shadows update with light movement.
- [ ] No hardcoded values in rendering path.

---

## 11. Risk Register

| Risk | Probability | Impact | Mitigation |
|:---|:---|:---|:---|
| Depth model too slow | Medium | High | ONNX → TensorRT FP16 → INT8 |
| Hand tracking jitter | High | Medium | EMA + Kalman filter |
| Normals noisy at edges | Medium | High | D2NT if Sobel insufficient |
| Shadow flickering | Medium | Medium | Temporal smoothing on light pos |
| Frame drops in demo | Medium | High | Reduce resolution, pipeline stages |
| WebRTC latency too high | Low | Low | Reduce stream resolution |
| GPU memory overflow | Low | High | Reduce batch size, use FP16 |
| MediaPipe false positives | Medium | Low | ROI cropping, confidence threshold |

### 11.1 Fallback Plans

| Component | Primary | Fallback 1 | Fallback 2 |
|:---|:---|:---|:---|
| Depth | DepthAnythingV2-Small | Flex-Nano | ZoeDepth |
| Normals | D2NT | Sobel + Kalman | Raw Sobel |
| Hand | MediaPipe | YOLO-pose | Manual ROI |
| Shadows | Cube shadow map | Single shadow map | Fake screen-space |
| Remote demo | WebRTC | Local screen share | Recorded video |

---

## 12. Team Roles

| Role | Responsibility | Suggested Skills |
|:---|:---|:---|
| **Project Lead** | Coordination, timeline, deliverables | Management, communication |
| **CV Engineer** | Depth estimation, normal translation | PyTorch, ONNX, TensorRT |
| **Graphics Engineer** | Lighting, shadows, shaders | GLSL, CUDA, rendering |
| **Integration Engineer** | Pipeline, test harness, WebRTC | Python, FastAPI, aiortc |
| **Documentation Lead** | Technical brief, pitch deck, video | Technical writing |

### 12.1 Contact Information

**FERJANI Yasmine** — Team Leader of Technical Programs Department
- yasmine.ferjani2@insat.ucar.tn
- +216 28 057 308

**BEN ALAYA Asma** — Technical Manager
- asma.benalaya@insat.ucar.tn
- +216 92 689 001

**ELOUED Abdelwahed** — Technical Manager
- eloued.abdelwahed@gmail.com
- +216 27 252 019

---

## Appendix A — Key Formulas Reference

### A.1 Back-Projection

```
x_c = z_c · (u - u₀) / f_x
y_c = z_c · (v - v₀) / f_y
```

### A.2 Light Direction

```
L = normalize(L_pos - P_pixel)
```

### A.3 Lambertian Diffuse

```
I_diffuse = k_d · I_light · max(dot(N, L), 0)
```

### A.4 Specular (Blinn-Phong)

```
H = normalize(L + V)
I_specular = k_s · I_light · max(dot(N, H), 0)^n
```

### A.5 Intensity Falloff

```
intensity = I₀ / (1 + (d / d_ref)²)
```

### A.6 Sobel Normals

```
nx = -fx · ∂D/∂x
ny = -fy · ∂D/∂y
nz = 1
N = (nx, ny, nz) / ||(nx, ny, nz)||
```

---

## Appendix B — Model Quick Reference

| Model | Task | Key Metric | Source |
|:---|:---|:---|:---|
| **DepthAnythingV2-Small** | Monocular depth | ~35 FPS @ 1280×720 (TensorRT FP16) | HuggingFace |
| **Flex-Nano** | Monocular depth | 0.7 GFLOPs, 37.6 FPS mobile | Self-supervised |
| **DepthPro** | Metric depth | 0.0635 AbsRel (DAIR-V2X-I) | Apple ML |
| **D2NT** | Depth → normals | 1.82 ms, 0.89° error | Real-time normal estimator |
| **MediaPipe Hands** | Hand tracking | 21 landmarks, ~40 FPS | Google |
| **Y-MAP-Net** | Multi-task | Depth + normals + pose + seg | Multi-teacher distillation |
| **M2H** | Multi-task | Depth + normals + seg + edge | 30 FPS RTX 3080 laptop |

---

## Appendix C — Testing Checklist

### Phase 1 Checklist
- [ ] Camera opens and displays feed.
- [ ] MediaPipe detects hand landmarks.
- [ ] DepthAnythingV2 produces depth map.
- [ ] Sobel produces normal map.
- [ ] 2×2 grid displays correctly.
- [ ] FPS ≥ 25.
- [ ] JSON results saved.

### Phase 2 Checklist
- [ ] Light position back-projected from hand.
- [ ] Diffuse shading responds to hand movement.
- [ ] Intensity metric modulates brightness.
- [ ] Temporal filter reduces jitter.
- [ ] Shadow mapping produces visible shadows.
- [ ] FPS ≥ 20.

### Submission Checklist
- [ ] `python main.py` launches cleanly.
- [ ] `requirements.txt` present.
- [ ] Pre-selection video recorded.
- [ ] Technical architecture brief written.
- [ ] Pitch deck prepared.
- [ ] Email sent to nrw8challenges@gmail.com with correct subject.

---

**End of Cahier de Charge**
```