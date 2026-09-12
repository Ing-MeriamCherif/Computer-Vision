"""Person 1 — Phase-1 shared config (hardware-agnostic).

All values can be overridden via environment variables or `.env`.
Runs on weak laptop AND on RTX 4060 without code changes.
"""
from __future__ import annotations

import os


def _getenv(key: str, default: str) -> str:
    return os.getenv(key, default)


def _getint(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _getfloat(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


# ---- Camera ----
CAMERA_ID: int = _getint("CAMERA_ID", 0)
CAMERA_WIDTH: int = _getint("CAMERA_WIDTH", 640)   # 480p baseline for hand FPS
CAMERA_HEIGHT: int = _getint("CAMERA_HEIGHT", 480)
CAMERA_FPS: int = _getint("CAMERA_FPS", 30)
# Mirror webcam (default 1): flip frame so moving hand right moves light right.
# Without this the vector looks inverted vs your own motion.
CAM_MIRROR: bool = _getenv("CAM_MIRROR", "1") == "1"

# ---- Device (auto; torch optional in Phase-1) ----
def resolve_device() -> str:
    req = _getenv("DEVICE", "auto").lower()
    if req in ("cuda", "cpu"):
        return req
    try:
        import torch  # optional
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


DEVICE: str = resolve_device()

# ---- Hand tracking ----
MAX_HANDS: int = _getint("MAX_HANDS", 1)  # 2 only for L05 later
# Lowered defaults (0.3): tasks detector misses edge hands at 0.5.
MIN_DETECTION_CONF: float = _getfloat("MIN_DETECTION_CONF", 0.3)
MIN_TRACKING_CONF: float = _getfloat("MIN_TRACKING_CONF", 0.3)
# Backend switch: "auto" | "tasks" | "yolo" | "mock".
# yolo = ultralytics pose (cuda on 4060, cpu fallback here). Compare DET and pick.
HAND_BACKEND: str = _getenv("HAND_BACKEND", "auto").lower()
YOLO_MODEL: str = _getenv("YOLO_MODEL", "yolo11n-pose.pt")
YOLO_DEVICE: str = _getenv("YOLO_DEVICE", "auto").lower()  # auto|cpu|cuda|0
YOLO_CONF: float = _getfloat("YOLO_CONF", 0.3)
HAND_DETECT_EVERY_N: int = _getint("HAND_DETECT_EVERY_N", 2)  # measured: 2x hand saving, no visible lag
# Downscaled inference input for hand detector (landmarks rescaled back).
# Tasks CNN cost scales ~linearly with pixels: 320x240 ~= 4x cheaper than 640x480.
HAND_INFER_W: int = _getint("HAND_INFER_W", 320)
HAND_INFER_H: int = _getint("HAND_INFER_H", 240)
ROI_CROP: bool = _getenv("ROI_CROP", "1") == "1"
ROI_PAD: float = _getfloat("ROI_PAD", 0.2)

# ---- Light vector (cdc 4.1, maybe-to-test math) ----
FIXED_Z_M: float = _getfloat("FIXED_Z_M", 0.5)  # fallback when no size/depth
D_REF_M: float = _getfloat("D_REF_M", 0.5)
I0: float = _getfloat("I0", 1.0)
EMA_ALPHA: float = _getfloat("EMA_ALPHA", 0.4)  # 0.3-0.5 per cdc
PALM_IDX: int = _getint("PALM_IDX", 9)
# Size-based depth proxy: z = fx * PALM_REAL_M / palm_px (palm width 5-17).
# This replaces the fixed 0.5m stub until the real depth model lands.
USE_SIZE_DEPTH: bool = _getenv("USE_SIZE_DEPTH", "1") == "1"
PALM_REAL_M: float = _getfloat("PALM_REAL_M", 0.085)  # adult palm width ~8-9cm
Z_MIN_M: float = _getfloat("Z_MIN_M", 0.2)
Z_MAX_M: float = _getfloat("Z_MAX_M", 3.0)
# ---- Vector resilience ----
# Hold last stable light briefly when hand is lost (edge/out-of-frame).
HOLD_LAST_S: float = _getfloat("HOLD_LAST_S", 0.5)
# FILTER: "oneeuro" (smooth at rest, responsive on fast moves) or "ema".
FILTER: str = _getenv("FILTER", "oneeuro").lower()
ONEEURO_MINCUTOFF: float = _getfloat("ONEEURO_MINCUTOFF", 1.0)
ONEEURO_BETA: float = _getfloat("ONEEURO_BETA", 0.3)  # snappy on fast moves
ONEEURO_DCUTOFF: float = _getfloat("ONEEURO_DCUTOFF", 1.0)
# Outlier gate: single-frame palm jump larger than this = misdetection.
# Genuine fast motion is exempt (see FAST_PX_S): gate only applies at low speed.
MAX_JUMP_PX: float = _getfloat("MAX_JUMP_PX", 280.0)
# Above this palm speed (px/s), frame-skip is disabled: fast hands get a
# detection every frame, otherwise the hand outruns the tracker.
FAST_PX_S: float = _getfloat("FAST_PX_S", 900.0)
# LK optical-flow bridge: track last palm up to N frames when detector drops
# (edge / motion blur). 0 disables.
LK_MAX_FRAMES: int = _getint("LK_MAX_FRAMES", 12)
# Torch glow radius in px @640 wide.
GLOW_RADIUS_PX: float = _getfloat("GLOW_RADIUS_PX", 260.0)

# ---- Camera intrinsics estimate (fx from ~60deg HFOV if unknown) ----
def estimate_intrinsics(w: int, h: int) -> tuple[float, float, float, float]:
    import math
    hfov_deg = float(os.getenv("HFOV_DEG", "60"))
    fx = (w / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    fy = fx  # square pixels assumption
    cx, cy = w / 2.0, h / 2.0
    return fx, fy, cx, cy


# ---- Render / validation ----
SHOW_GRID: bool = _getenv("SHOW_GRID", "1") == "1"

# Person4-compatible light colors (warm default for light 1)
LIGHT_COLOR_WARM = (1.0, 0.85, 0.7)
