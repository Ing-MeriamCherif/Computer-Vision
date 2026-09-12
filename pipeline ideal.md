This is a solid implementation plan. Let me structure it into a clear technical document you can share with your team, filling in the details for each of the five parts and adding the remote demo platform idea.

---

# NRW 8th Edition — Implementation Plan & Technical Document

## Overview

The pipeline has five workstreams that map directly to the challenge levels. Each has a clear benchmark target, a latency budget, and a validation method.

| Part | Workstream | Challenge Levels | Target Latency |
|:---|:---|:---|:---|
| 1 | Hand Tracking & Gesture Control | L03, L05 | < 15 ms |
| 2 | Depth Estimation | L01 (foundation) | < 20 ms |
| 3 | Surface Normal Translation | L01 | < 5 ms |
| 4 | Lighting & Shadow Rendering | L02, L04 | < 10 ms |
| 5 | Multi-Task Model Exploration | All (pipeline optimization) | Research phase |

**Total frame budget for 60 FPS: 16.6 ms.** You will need to pipeline stages (frame N's rendering overlaps frame N+1's depth inference) to fit within this budget.

---

## Part 1: Hand Tracking & Light Vector Control

### 1.1 Model Selection & Benchmarking

**Primary candidate: MediaPipe Hands.** This is the industry standard for real-time hand tracking. It delivers 21 2D/2.5D landmarks per hand and runs at **~40 FPS on Apple M2 Air** , with benchmarked latency as low as **18.2 ms on mid-tier hardware (54.9 FPS)** . A 2026 ACM paper integrating MediaPipe with TouchDesigner achieved **50 ± 8 ms end-to-end latency at 60 FPS** .

**Alternative: YOLOv8-pose or YOLOv11-Hand.** A comparative study found that MediaPipe achieves the best overall performance and processing speed (20.09 FPS average), while YOLO variants offer better robustness to background clutter . For your challenge, MediaPipe's speed advantage is decisive.

**Benchmark plan:**
- Test MediaPipe Hands at 480p, 720p, and 1080p on your target hardware.
- Measure: inference latency (ms), landmark jitter (pixel std dev frame-to-frame), and FPS.
- Test with 1 hand and 2 hands (for Level 05 multi-light).
- Record CPU vs GPU delegate performance.

**Expected results (from published benchmarks):**

| Resolution | Latency | FPS | Notes |
|:---|:---|:---|:---|
| 480p | ~18 ms | ~54 FPS | Theoretical upper limit  |
| 720p | ~27 ms | ~37 FPS | Standard demo resolution |
| 1080p | ~38 ms | ~26 FPS | Diminishing returns; background false positives increase  |

### 1.2 Light Direction Vector & Intensity Metric

Once you have the hand's 2D landmarks, you need three things:

**(a) Direction vector.** Back-project the palm center landmark to 3D using the pinhole model:

```
x_c = z_c · (u - u₀) / f_x
y_c = z_c · (v - v₀) / f_y
```

Where `z_c` comes from **sampling the depth map** at the palm's pixel coordinates (not from MediaPipe's relative z values, which are not metric). The light direction for each pixel is then `L = normalize(light_pos - pixel_pos_3d)`.

**(b) Intensity metric based on distance.** Define:

```
intensity = I₀ / (1 + (d / d_ref)²)
```

Where `d` is the Euclidean distance from the hand to the camera center, `d_ref` is a reference distance (e.g., 0.5m), and `I₀` is the base intensity. This is an inverse-square falloff approximation. When the hand moves closer (Z decreases), intensity grows; when it moves away, intensity falls.

**(c) Color filter modulation.** Apply the intensity metric as a global multiplier on the rendered image. For a warm light effect, modulate the RGB channels differently (e.g., `R * intensity * 1.1`, `G * intensity`, `B * intensity * 0.9`) to simulate a warm point light.

### 1.3 Temporal Filtering & Stabilization

Raw MediaPipe landmarks jitter significantly frame-to-frame. You need **two layers of filtering**:

**Layer 1: Per-landmark Kalman filter.** Maintain an independent constant-velocity Kalman filter for each of the 21 landmarks. This is the standard approach in hand tracking systems . The Kalman filter predicts the next position based on velocity and corrects it with the new measurement, suppressing high-frequency jitter while preserving smooth motion .

**Layer 2: Exponential Moving Average (EMA) on the light position.** After computing the 3D light position, apply a lightweight EMA:

```
light_pos_smoothed = α · light_pos_new + (1 - α) · light_pos_previous
```

With `α ≈ 0.3–0.5`, this provides a first-order recursive smoothing that reduces fluctuations in the final light position . This is critical for stable shadows—jittery light positions cause flickering shadow maps.

### 1.4 Parallelization Strategy

MediaPipe itself is optimized for CPU/GPU inference. For maximum FPS:

- Run **palm detection** and **landmark regression** as separate stages on the GPU where possible.
- Overlap hand tracking with depth inference: MediaPipe runs on CPU while the depth model runs on GPU.
- Use a **double-buffered pipeline**: frame N's hand tracking result feeds into frame N's lighting, while frame N+1's depth is already being computed.

---

## Part 2: Depth Estimation

### 2.1 Model Selection & Benchmarking

You will benchmark **FlexDepth (Flex-Nano)**, **DepthPro**, and **DepthAnythingV2**, plus additional SOTA models. Here is the published data to guide your benchmarking:

| Model | Params | GFLOPs | Key Benchmark | Latency | Notes |
|:---|:---|:---|:---|:---|:---|
| **Flex-Nano** | ~1.5M | **0.7** | Generalization | **37.6 FPS mobile** | Ultra-lightweight, self-supervised |
| **DepthPro** | ~100M+ | ~50–100 | **0.0635 AbsRel** (DAIR-V2X-I) | < 1 s (unoptimized) | Metric depth, sharp boundaries  |
| **DepthAnythingV2 (ViT-S)** | 24.7M | 53.6 @ 420×560 | Strong generalization | **7.5 ms** (TensorRT INT8) | NASA VPEngine basis  |
| **ZoeDepth** | — | — | Fastest baseline | **2.2 FPS** (desktop) | Lower accuracy  |

**Critical finding from recent benchmarks:** DepthAnythingV2's inference time is heavily resolution-dependent. On KITTI, inference increases by **~50 ms on desktop and ~1180 ms on Jetson** compared to lower resolutions . TensorRT is **2–2.7× faster than ONNX Runtime**, which is **1.4× faster than PyTorch** .

**Benchmark plan:**
- Test all models at 336×252, 420×560, and 518×518 input resolutions.
- Measure latency on your exact GPU (TensorRT FP16 and INT8).
- Evaluate depth quality on a fixed set of indoor/outdoor scenes (look for edge sharpness and temporal consistency).
- Record VRAM usage for each model.

### 2.2 The "Cumulative Depth Between Two Points" Method

Your idea—accumulating depth differences between two points, and if the difference is high, treating it as "even deeper"—is a **local relative depth refinement** approach. This can be implemented as a **local depth gradient accumulator**:

For each pixel, compute the depth gradient magnitude `|∇D|`. If `|∇D|` exceeds a threshold `τ`, it indicates a depth discontinuity (object boundary). In that region, apply a local correction:

```
D_corrected(x,y) = D(x,y) + λ · Σ_Ω |D(x,y) - D(x',y')|
```

Where `Ω` is a local neighborhood. This effectively sharpens depth edges without global computation. It is analogous to the discontinuity-aware filtering in D2NT and can improve edge quality at minimal cost.

**Validation:** Compare `D_corrected` against ground-truth depth (NYUv2 indoor, KITTI outdoor) using AbsRel and RMSE.

### 2.3 Smoothing & Accumulation Strategy

To maximize frame-to-frame stability:

1. **Temporal depth smoothing:** Apply an EMA across consecutive depth maps: `D_t = β · D_t + (1-β) · D_{t-1}` with `β ≈ 0.7–0.8`. This reduces flickering in the depth map, which propagates to smoother normals and shadows.
2. **Bilateral filter on depth:** Apply a small (5×5) bilateral filter after depth inference to smooth flat regions while preserving edges. This is cheap and dramatically improves normal quality.
3. **Spatial accumulation for static scenes:** For regions with low depth variance (static background), accumulate depth over multiple frames to reduce noise. Use a frame counter per pixel: if `|D_t - D_{t-1}| < ε`, increment the accumulator; otherwise reset. This is a lightweight temporal fusion.

---

## Part 3: Surface Normal Translation

### 3.1 D2NT vs. Filter-Based Approaches

**D2NT is the recommended primary method.** It achieves **1.82 ms runtime and 0.89° average angular error**, and it translates depth maps directly to normals without computing 3D coordinates . Its **Discontinuity-Aware Gradient (DAG) filter** adaptively generates gradient kernels, which is critical for handling depth edges correctly .

**Sobel and Kalman filters as auxiliary tools:**

| Method | Latency | Edge Quality | Use Case |
|:---|:---|:---|:---|
| **D2NT (DAG + MRF)** | 1.82 ms | Excellent at discontinuities | Primary normal estimation  |
| **Sobel filter** | ~0.1–0.5 ms | Poor at edges (blurs normals) | Fast preview / fallback |
| **Kalman filter on normals** | ~0.5 ms | Temporal smoothing | Post-processing stabilization |

**Rule-based correction:** After D2NT, apply a simple rule: if the normal's dot product with the camera direction is negative (pointing away), flip it. This is a one-line correction that fixes D2NT's occasional orientation errors.

**Benchmark plan:**
- Implement D2NT in a GPU shader or CUDA kernel.
- Compare its output against Sobel-based normals on the same depth maps.
- Measure angular error against ground-truth normals (available in NYUv2 and synthetic datasets).
- Measure the latency impact of adding Kalman temporal filtering on normals.

### 3.2 Parallel Normalization

Normalizing each surface normal to unit length is a per-pixel operation that is trivially parallelizable. After D2NT produces the raw normal components `(n_x, n_y, n_z)`, compute:

```
N = (n_x, n_y, n_z) / sqrt(n_x² + n_y² + n_z²)
```

This runs on the GPU as a single shader pass with one thread per pixel. No synchronization needed.

---

## Part 4: Lighting & Shadow Computation

### 4.1 Light Vector Calculation

Once you have:
- **Light position** `L_pos` (from hand tracking, Part 1)
- **Pixel 3D position** `P_pixel` (from depth back-projection, Part 2)
- **Surface normal** `N` (from D2NT, Part 3)

Compute the light direction vector per pixel:

```
L = normalize(L_pos - P_pixel)
```

This is a per-pixel vector subtraction and normalization—fully parallelizable.

### 4.2 Diffuse & Specular Shading

**Lambertian diffuse:**

```
I_diffuse = k_d · I_light · max(dot(N, L), 0)
```

Where `k_d` is the material albedo (you can use the original RGB pixel value as an approximation) and `I_light` is the light intensity from Part 1.

**Specular (Blinn-Phong):**

```
H = normalize(L + V)        // halfway vector
I_specular = k_s · I_light · max(dot(N, H), 0)^n
```

Where `V` is the view direction (toward camera) and `n` is the shininess exponent (e.g., 32–128 for a shiny surface).

**Combined shading:**

```
I_final = I_ambient + I_diffuse + I_specular
```

### 4.3 Shadow Mapping

**Standard approach:** Render the scene's depth from the light's point of view into a **shadow map**. During the main render pass, compare each pixel's distance from the light to the shadow map value. If the pixel is farther, it's in shadow.

**Performance optimization: Shadow Cache.** A 2026 patent demonstrates that a Shadow Cache approach achieves **210 FPS** with a 2000m sight distance, compared to **160 FPS** for cascaded shadow maps (CSM). For your single-light setup, use a **cube shadow map** (6 faces, 512×512 each) for the point light.

**Shadow bias:** Add a small bias (e.g., 0.001–0.005) to the comparison to prevent self-shadowing artifacts (shadow acne).

### 4.4 Shader vs. Per-Pixel vs. Local Approach

| Approach | Latency | Quality | Recommendation |
|:---|:---|:---|:---|
| **Per-pixel shader** | Lowest | Good | Use for diffuse and specular shading |
| **Local patch** (e.g., D2NT's DAG filter) | Low | Excellent at edges | Use for normal estimation |
| **Full-screen pass** | Medium | Highest | Use for final composition and color grading |

**Recommended pipeline:** D2NT for normals (patch-level), per-pixel shaders for shading and shadow comparison, and a final full-screen pass for color grading and intensity modulation.

---

## Part 5: Multi-Task Models — Research & Exploration

### 5.1 Y-MAP-Net

Y-MAP-Net is a **Y-shaped neural network** that simultaneously predicts **depth, surface normals, human pose (17 COCO joints), semantic segmentation, and multi-label captions** in a single forward pass . It uses a **multi-teacher, single-student training paradigm**, distilling the capabilities of larger foundation models into a lightweight architecture .

**For your challenge, Y-MAP-Net is the most promising multi-task candidate** because:
- It provides **depth + normals + human pose** in one pass, replacing three separate models.
- It supports **TensorFlow, TFLite, JAX, and ONNX** backends .
- It has a **Gradio web UI** for quick testing .
- It is explicitly designed for **resource-constrained robotic platforms** .

**Benchmark plan:** Measure Y-MAP-Net's latency for the full output set vs. running MediaPipe + DepthAnythingV2 + D2NT separately.

### 5.2 M2H (Multi-Mono-Hydra)

M2H predicts **semantic segmentation, depth, edge, and surface normal estimation** from a single monocular image . Its key innovation is a **Window-Based Cross-Task Attention Module** that enables structured feature exchange between tasks .

**Published benchmarks:**
- **+3.4% mIoU improvement** in semantic segmentation on NYUDv2 
- **13% lower RMSE** in depth estimation 
- **30 FPS on an RTX 3080 laptop GPU** 

**For your challenge:** M2H gives you depth and normals in one pass, but does not provide human pose. You would still need MediaPipe for hand tracking. The trade-off is: one heavy model (M2H) + one light model (MediaPipe) vs. two heavy models (DepthAnythingV2 + separate normal estimator) + one light model.

### 5.3 Other Multi-Task Options to Explore

- **UniDepth V2:** Achieves near-real-time performance and was the most consistent and efficient model in a nuScenes benchmark . Worth testing as a depth-only model.
- **Metric3D v2:** Trades efficiency for stronger accuracy . Good for metric depth if you need absolute scale.
- **Distill Any Depth:** Ranked #2 on NYU-Depth V2 . A distillation approach that could give you a smaller, faster depth model.

### 5.4 Decision Matrix

| Pipeline | Models | Est. Total Latency | Complexity |
|:---|:---|:---|:---|
| **Separate (recommended baseline)** | MediaPipe + DepthAnythingV2 + D2NT | ~15 + 8 + 2 = **25 ms** | Medium |
| **Multi-task (Y-MAP-Net)** | Y-MAP-Net (all-in-one) | **~20 ms** (est.) | Low |
| **Multi-task (M2H)** | M2H + MediaPipe | ~18 + 15 = **33 ms** | Medium |
| **Lightweight** | MediaPipe + Flex-Nano + D2NT | ~15 + 5 + 2 = **22 ms** | Medium |

**Recommendation:** Start with the **separate pipeline** as your baseline, then benchmark Y-MAP-Net. If Y-MAP-Net's latency is under 20 ms for all outputs, switch to it. This gives you the best of both worlds: depth, normals, and human pose in one pass.

---

## Remote Demo Platform: Phone/PC via WebRTC

Your idea of using a **browser-based remote rendering platform** for the demo is excellent. It allows the jury to test the system on their own phones without installing anything.

### Architecture

```
Phone Browser ←→ Signaling Server ←→ GPU Server (PC)
     │                                      │
     │  WebRTC video stream (H.264/H.265)   │
     │  ◄───────────────────────────────────┤
     │                                      │
     │  Input data channel (touch/mouse)    │
     ├──────────────────────────────────────►
     │                                      │
     │  Camera feed from phone (if needed)  │
     └──────────────────────────────────────►
```

### How It Works

**Pixel Streaming** is the proven approach. It runs the application on a remote GPU server and streams the rendered output to a web browser in real-time over WebRTC . The user sees and interacts with the application through their browser—no download, no installation, no minimum hardware requirements .

**WebRTC provides sub-100ms latency** on good connections, compared to 200–500ms for traditional streaming solutions . It has adaptive bitrate, NAT traversal via STUN/TURN, and is built into every modern browser .

### Implementation for Your Challenge

1. **GPU Server (PC):** Runs your full pipeline (MediaPipe + depth + normals + lighting + shadows). Renders the final frame at 60 FPS.
2. **Signaling Server:** A lightweight Node.js server that brokers connections between the GPU server and the phone browser .
3. **Phone Client:** A simple HTML/JavaScript page that displays the video stream and sends input (touch, gyroscope, camera feed) back to the GPU server via WebRTC data channel .

**For the camera feed:** If the phone's camera is used as input, stream it from the phone to the GPU server via WebRTC. The server processes it and streams the rendered result back. This gives you a **true phone-based demo** with the phone as both camera and display.

### Latency Budget

| Stage | Latency |
|:---|:---|
| Phone camera capture | ~10 ms |
| WebRTC uplink (phone → server) | ~30–50 ms |
| Server processing (full pipeline) | ~25 ms |
| WebRTC downlink (server → phone) | ~30–50 ms |
| **Total end-to-end** | **~95–135 ms** |

This is acceptable for a demo. The jury will see the light respond to gestures with a slight delay, but the interaction will feel responsive.

### Recommended Tools

- **MediaMTX / Pion / aiortc** for WebRTC signaling and streaming.
- **GStreamer** for H.264 encoding on the GPU server.
- **Tailscale or ngrok** for exposing the local server to the internet without complex networking.

---

## Summary Implementation Roadmap

| Phase | Task | Deliverable | Timeline |
|:---|:---|:---|:---|
| **Week 1** | Benchmark MediaPipe, DepthAnythingV2, Flex-Nano, DepthPro | Benchmark report with latency tables | |
| **Week 2** | Implement D2NT + Sobel comparison + Kalman temporal filtering | Normal map visualization + angular error report | |
| **Week 3** | Implement lighting pipeline (diffuse + specular + shadow map) | Interactive light demo with shadows | |
| **Week 4** | Test Y-MAP-Net and M2H; compare with separate pipeline | Multi-task benchmark report | |
| **Week 5** | Build WebRTC remote demo platform | Phone-accessible live demo | |
| **Week 6** | Optimize TensorRT, INT8 quantization, temporal smoothing | Final optimized pipeline at >30 FPS | |
| **Week 7** | Record pre-selection video, write technical architecture brief, prepare pitch deck | All three deliverables | |

### Key Metrics to Track

- **End-to-end latency** (camera → rendered frame)
- **FPS** (frames per second)
- **Depth AbsRel** (accuracy)
- **Normal angular error** (degrees)
- **Shadow smoothness** (frame-to-frame shadow map variance)
- **Hand tracking jitter** (pixel std dev)

### Anti-Cheat Compliance

- No pre-rendered footage: all outputs must come from the live pipeline.
- No hardcoded depth maps: depth must be computed from the camera feed.
- No unhandled frame drops: maintain ≥30 FPS throughout the demo.
- Live demo must match the pre-selection video exactly.

---

This document gives your team a clear starting point. The next step is to set up your benchmarking environment and run the first round of model tests. Would you like me to detail the TensorRT conversion steps for DepthAnythingV2, or the WebRTC signaling server setup?