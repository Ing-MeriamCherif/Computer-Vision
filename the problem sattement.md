# AI & Vision Challenge Specifications Book

**8th Edition - Challenges**  
**National Robotics Week | 8th edition**

---

## INTRODUCTION

Welcome to the NRW 8th Edition AI & Vision Challenge: The ultimate proving ground for computer vision architects and spatial rendering engineers. To a standard camera, the world is just a flat grid of passive pixels, your mission is to give the machine sight, depth and spatial intelligence. Landing at the top of this leaderboard requires slicing through monocular feed noise, building instant 3D geometry and bending virtual light in real time.

---

## CONTEXT AND VISION

### Hackathon Context

Organized under the 8th Edition of National Robotics Week by IEEE INSAT Student Branch and IEEE RAS INSAT Chapter, this challenge bridges deep-learning spatial perception with real-time computer graphics by teaching a computer to see a camera feed, estimate 3D geometry and interact with dynamic virtual illumination in real time.

### Vision

**Spatial Perception:** Turn flat 2D RGB camera feeds into responsive, mathematically sound 3D environments without dedicated LiDAR hardware.

**Real-Time Edge AI:** Prove that modern monocular depth and segmentation models can run synchronously with graphics pipelines at ultra-low latency.

---

## TARGET ACQUISITION PROTOCOL

To claim the top spot on the leaderboard, your team must hit every target with absolute precision. Use the protocol below as your field guide to master our core mission, clear each technical milestone, and push your real-time rendering pipeline to its absolute limit.

### PRIMARY OBJECTIVE

Develop a high-throughput, real-time computer vision system that takes a standard camera feed, extracts pixel-accurate 3D depth geometry and renders a user-controlled virtual light source that dynamically illuminates the scene, casts realistic dynamic shadows, and reacts seamlessly to real-time hand gestures including moving the light source forward and backward through 3D space (X, Y, Z).

### PROGRESSION

This challenge is structured as a progression: each level earns you points and takes you to the next. You're not limited to the levels we share; it's about what you add next, as creativity is key in the scoring system.

---

## "GAMIFICATION" THE TECHNICAL PROGRESSION LADDER

Teams earn points by progressively unlocking technical milestones:

### Level 01: GEOMETRY ENGINE

**Objective:** Extract continuous 3D surface normal vectors N(x,y) from depth gradients in real time.

**Visual Benchmark:** Live depth-to-normal map visualization running synchronously with the camera feed.

### Level 02: DYNAMIC RELIGHTING

**Objective:** Apply real-time Lambertian diffuse and specular shading models onto the calculated 3D surface geometry.

**Visual Benchmark:** Realistic scene illumination responsive to an interactive virtual point light source.

### Level 03: SPATIAL GESTURE CONTROL

**Objective:** Integrate real-time hand-tracking algorithms to control light placement using physical gestures.

**Visual Benchmark:** Smooth spatial control across X, Y, and Z axes (bringing the virtual light closer or pushing it deeper into the scene depth).

### Level 04: DYNAMIC SHADOWS

**Objective:** Project geometry-aware occlusion shadows cast behind foreground subjects based on relative light positions.

**Visual Benchmark:** Realistic, dynamic shadow vectors that adjust smoothly as the gesture moves the light source.

### Level 05: MULTI-LIGHT & VOLUMETRIC SCATTERING

**Objective:** Support multi-hand gesture tracking to control multiple independent light sources simultaneously.

**Visual Benchmark:** Complex light interactions with overlapping shadow paths and volumetric atmospheric haze rendering.

### Level ∞: THE HYPER-SCALE HORIZON

**Objective:** Apply real-time Lambertian diffuse and specular shading models onto the calculated 3D surface geometry.

**Visual Benchmark:** Realistic scene illumination responsive to an interactive virtual point light source.

---

## SUBMISSION DELIVERABLES CHECKLIST

### DEPLOYMENT PAYLOAD // REQUIRED ARTIFACTS

To qualify for final grading, your team must package and submit all three artifacts before the countdown hits zero.

- **SOURCE CODE REPOSITORY:** A clean, executable codebase containing your complete pipeline, custom modules and dependency manifest (requirements.txt or Dockerfile). Must launch directly via `python main.py`.
- **TECHNICAL ARCHITECTURE BRIEF:** A system report detailing your neural model selection, normal extraction formulas, temporal filtering strategy and frame-rate optimization tricks.
- **PITCH DECK:** A presentation designed for the jury defense covering your architectural pipeline, technical trade-offs, FPS benchmark results and live demo configuration.

---

## EVALUATION PROTOCOL & LIVE ARENA RULES

### THE TWO-PHASE GAUNTLET // AUTOMATED GRADING & STAGE STRESS-TEST

Submissions are evaluated through a two-phase protocol: an automated benchmark filter for pre-selection, followed by a live interactive presentation and stage defense before the jury.

### Phase 01 // Pre-Selection Demo Submission

- **Video & Code Submission:** Teams submit a short, continuous, unedited video demo showing their live system in action, along with their codebase and project documentation.
- **Visual & Fluidity Review:** Evaluators screen all submitted demos based on real-time rendering quality, hand gesture responsiveness, dynamic shading/shadow precision and overall framerate fluidity.
- **Finalist Shortlist:** Only the top-ranked teams that demonstrate seamless real-time interaction and advance furthest along the progression ladder will be selected as finalists.

### Phase 02 // Live Stage Defense & Jury Presentation

- **Live Interactive Demonstration:** Finalist teams hook up their pipeline live on stage, allowing the jury to physically test real-time gesture control (X, Y, Z light manipulation), specular shading and dynamic shadow projection.
- **Technical Presentation:** Teams present their full project documentation to the jury, walking through their vision pipeline, depth/normal estimation methods, latency optimizations and architectural choices.
- **Live Verification & Anti-Cheat:** The live demo must match the capabilities shown in the pre-selection video. Any pipeline using pre-rendered footage, hardcoded depth maps, or experiencing unhandled frame drops during live interaction will be immediately disqualified.

---

## SUBMISSION INSTRUCTIONS

Please submit your final document by email to: **nrw8challenges@gmail.com**

The email subject must follow this format: **[Challenge Name] - [Team Name]**

---
