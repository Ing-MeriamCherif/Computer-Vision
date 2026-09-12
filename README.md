# Person 1 — Hand Vector Torch (Phase 1)

Hand-tracked virtual torch: palm position → 3D light position → direction vector
+ distance-falloff intensity. Owns **detection path only** (capture → hand →
vector). Rendering/shadows belong to Person 4; depth model lands in Phase 2.

## Setup (RTX 4060 machine)

```powershell
# Python 3.10 or 3.11 recommended (3.13 works, see note)
pip install -r requirements.txt
# MediaPipe Tasks bundle (hand detector, ~8 MB, one-time download):
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task','hand_landmarker.task')"
# YOLO backend model auto-downloads on first use (yolo11n-pose.pt)
# For CUDA torch (4060): install torch with CUDA from https://pytorch.org first
```

Python 3.13 note: `mediapipe` 1.x removed `mp.solutions` (legacy API) — the
`tasks` backend is used, which is the maintained path anyway.

## Run

```powershell
$env:MP_HAND_BUNDLE='hand_landmarker.task'
python main.py --mode hand-only                 # live window, q quits
python main.py --mode profile --save-json results/mine.json   # + timing table
# No webcam? synthetic frames:
python main.py --mode profile --no-camera --no-show --max-frames 60
```

Screen: green landmarks + palm circle + green arrow (light direction) +
`I=` intensity + `Z=` depth proxy + warm glow around hand (torch).
Top-left: green `FPS` (full pipeline) + yellow `DET` (**detection-path FPS —
the number to compare**).

## Backend switch (compare on the 4060)

```powershell
$env:HAND_BACKEND='tasks'   # MediaPipe, CPU ~10ms here
$env:HAND_BACKEND='yolo'    # YOLO11n-pose wrist, needs CUDA to win
$env:YOLO_DEVICE='cuda'     # on 4060; 'cpu' fallback elsewhere
```

Measured here (CPU): Tasks 9.7ms vs YOLO-CPU 80ms. Expected on 4060
(published): YOLO11n-pose PyTorch FP32 ≈ 5–8ms, TensorRT ≈ 2ms → DET 60–100+.

## Parameters (every knob, via env or `.env`)

| Param | Default | What to test |
|:---|:---|:---|
| `CAMERA_ID/W/H/FPS` | 0/640/480/30 | 640×480 baseline; 1280×720 for quality check |
| `CAM_MIRROR` | 1 | 1 = motion matches vector (webcam); 0 = raw camera |
| `HAND_BACKEND` | auto | tasks vs yolo — compare DET per machine |
| `MP_HAND_BUNDLE` | hand_landmarker.task | path to Tasks bundle; absent → next backend |
| `YOLO_MODEL/DEVICE/CONF` | yolo11n-pose.pt/auto/0.3 | DEVICE=cuda on 4060 |
| `MAX_HANDS` | 1 | 2 reserved for L05 multi-light |
| `MIN_DETECTION/TRACKING_CONF` | 0.3 | raise to 0.5 if false positives; lower if edges drop |
| `HAND_DETECT_EVERY_N` | 2 | 1 = max accuracy; 3 = max FPS (auto-disabled on fast motion) |
| `HAND_INFER_W/H` | 320/240 | no-op for Tasks (internal resize); keep full-res |
| `FILTER` | oneeuro | oneeuro (smooth rest + snappy moves) vs ema |
| `ONEEURO_MINCUTOFF/BETA/DCUTOFF` | 1.0/0.3/1.0 | MINCUTOFF↓ steadier; BETA↑ snappier on fast moves |
| `EMA_ALPHA` | 0.4 | only for FILTER=ema |
| `MAX_JUMP_PX` | 280 | teleport gate (exempt at high speed) |
| `FAST_PX_S` | 900 | above this: skip off + gate off (fast-motion mode) |
| `LK_MAX_FRAMES` | 8 | optical-flow coast on dropout; 0 disables |
| `HOLD_LAST_S` | 0.3 | freeze last light on total loss |
| `USE_SIZE_DEPTH/PALM_REAL_M/Z_MIN/Z_MAX` | 1/0.085/0.2/3.0 | Z=fx·real/palm_px proxy until real depth; calibrate PALM_REAL_M to your hand |
| `D_REF_M/I0` | 0.5/1.0 | torch falloff: I=I₀/(1+(d/d_ref)²); smaller d_ref = faster dimming |
| `GLOW_RADIUS_PX` | 260 | torch glow size on screen |

## Fast motion notes

Rapid hands fail in this order: motion blur kills detector → displacement
outruns tracker → filter lags. Fixes wired in: wide LK window (31px/4lvl),
fast-motion mode (skip off above `FAST_PX_S`), OneEuro `BETA=0.3`, speed-exempt
outlier gate. Hardware help: 60fps camera mode halves inter-frame displacement.

## Continuous API (for renderer / teammates / phone)

In-process or over HTTP — same packet, Person-4 field names
(`position_camera_m`, `intensity`, `active`, `timestamp_s`):

```python
from pipeline import TorchPipeline
pipe = TorchPipeline().start()          # background capture->hand->vector loop
print(pipe.latest)                      # newest packet, never queued
pipe.on_packet(lambda p: render(p))     # push
for p in pipe.packets(): ...            # pull generator
pipe.stop()
```

```powershell
python api_server.py --port 8000        # live camera
curl http://<PC_IP>:8000/health         # {"ok":true,"backend":"mediapipe-tasks"}
curl http://<PC_IP>:8000/light          # latest packet JSON
curl -N http://<PC_IP>:8000/stream      # SSE, one event per packet
```

## GPU run (4060)

```powershell
copy .env.gpu.example .env
pip install -r requirements.txt   # cuda torch build for your CUDA version
python main.py --mode profile --save-json results/gpu_baseline.json
python -m depth.benchmark         # depth package spread (see AGENTS.md)
python tools/depth_live.py --depth-every 1
```

Merged branches: `talel-phase2` (D2NT normals, depth estimator) +
`feature/depth` (`depth/` async package, `model_comparison/`, tests).
Person-1 pipeline files kept intact; `depth/` is standalone (`python -m ...`).

## Files

`main.py` loop · `pipeline.py` continuous API · `api_server.py` HTTP/SSE ·
`hand_tracker.py` backends · `light_vector.py` math (+Person-4
`Light` contract: +X right/+Y down/+Z fwd, meters) · `filters.py` EMA/1€/gates ·
`validation_render.py` glow+grid · `normals_stub.py` half-res Sobel placeholder ·
`config.py` env config · `results/` profiles (`bench_*.json`).
