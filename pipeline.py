"""Person 1 continuous API: TorchPipeline.

Wraps capture -> hand backend -> size-Z -> filter/gates/LK bridge into a
single step() + a background run() loop publishing latest-packet semantics:

- `latest` always holds the newest LightPacket (Person 4 Rule 4: consumers
  take latest, no queue growth).
- `on_packet(cb)` for push consumers, `packets()` generator for pull loops.
- Packet dict matches Person 4 `Light`/`LightState` field names so the
  renderer can consume it without translation.

No rendering, no depth model here — detection path only.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

import config
from filters import (EMAFilter, OneEuroVec3, adaptive_alpha, edge_factor,
                     jump_rejected)
from hand_tracker import HandResult, create_tracker
from light_vector import compute_intensity, depth_from_palm_size, palm_to_light
from utils import FPSMeter, Intrinsics


@dataclass
class LightPacket:
    frame_id: int
    timestamp_s: float
    found: bool
    held: bool                      # coasting (LK) or frozen (hold window)
    active: bool                    # found or held -> renderer should light
    palm_uv: list[float] | None
    position_camera_m: list[float] | None  # +X right, +Y down, +Z fwd, meters
    intensity: float
    z_m: float | None
    confidence: float
    backend: str
    det_fps: float
    color_rgb: list[float] = field(default_factory=lambda: [1.0, 0.85, 0.7])


class TorchPipeline:
    def __init__(self, width: int | None = None, height: int | None = None,
                 no_camera: bool = False) -> None:
        self.W = width or config.CAMERA_WIDTH
        self.H = height or config.CAMERA_HEIGHT
        fx, fy, cx, cy = config.estimate_intrinsics(self.W, self.H)
        self.intr = Intrinsics(fx, fy, cx, cy, self.W, self.H)
        self.no_camera = no_camera
        self.tracker = create_tracker()
        self.ema = EMAFilter(alpha=config.EMA_ALPHA)
        self.oneeuro = OneEuroVec3(mincutoff=config.ONEEURO_MINCUTOFF,
                                   beta=config.ONEEURO_BETA,
                                   dcutoff=config.ONEEURO_DCUTOFF)
        self.use_oneeuro = config.FILTER == "oneeuro"
        self.det_meter = FPSMeter()
        self.infer_w, self.infer_h = config.HAND_INFER_W, config.HAND_INFER_H
        self.sx, self.sy = self.W / self.infer_w, self.H / self.infer_h

        self._lock = threading.Lock()
        self._latest: LightPacket | None = None
        self._cbs: list = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_id = 0

        # carry-over state (same logic as main.py loop)
        self._last_res: HandResult | None = None
        self._last_light_pos: np.ndarray | None = None
        self._last_seen_t = 0.0
        self._last_raw_uv = None
        self._last_raw_t = 0.0
        self._last_speed = 0.0
        self._last_palm_px = None
        self._prev_gray = None
        self._lk_point = None
        self._lk_frames = 0
        self._last_packet: LightPacket | None = None

    # ---- consumer API ----
    @property
    def latest(self) -> LightPacket | None:
        with self._lock:
            return self._latest

    def on_packet(self, cb) -> None:
        self._cbs.append(cb)

    def packets(self):
        """Pull generator: yields each new packet once (poll latest)."""
        seen = -1
        while not self._stop.is_set():
            p = self.latest
            if p is not None and p.frame_id != seen:
                seen = p.frame_id
                yield p
            else:
                time.sleep(0.002)

    # ---- lifecycle ----
    def start(self) -> "TorchPipeline":
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.tracker.close()
        except Exception:
            pass

    # ---- one step ----
    def open_camera(self):
        cap = cv2.VideoCapture(config.CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        return cap if cap.isOpened() else None

    def _loop(self) -> None:
        cap = None if self.no_camera else self.open_camera()
        try:
            while not self._stop.is_set():
                pkt = self.step(cap)
                with self._lock:
                    self._latest = pkt
                for cb in self._cbs:
                    try:
                        cb(pkt)
                    except Exception:
                        pass
        finally:
            if cap is not None:
                cap.release()

    def step(self, cap=None, frame_bgr=None) -> LightPacket:
        W, H, intr = self.W, self.H, self.intr
        if frame_bgr is not None:
            frame = frame_bgr
        elif self.no_camera or cap is None:
            frame = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
            frame = cv2.GaussianBlur(frame, (21, 21), 0)
        else:
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (W, H))
                if config.CAM_MIRROR:
                    frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        n = max(1, config.HAND_DETECT_EVERY_N)
        if (self._last_res is None or not self._last_res.found
                or self._last_light_pos is None):
            n = 1
        elif self._last_speed > config.FAST_PX_S:
            n = 1
        if self._last_res is not None and (self._frame_id % n) != 0:
            res = self._last_res
        else:
            small = cv2.resize(rgb, (self.infer_w, self.infer_h),
                               interpolation=cv2.INTER_LINEAR)
            res = self.tracker.process(small)
            if res.found and res.landmarks_uv is not None:
                res.landmarks_uv = res.landmarks_uv * np.array([self.sx, self.sy])
                if res.palm_uv is not None:
                    res.palm_uv = (res.palm_uv[0] * self.sx, res.palm_uv[1] * self.sy)
                if res.palm_px is not None:
                    res.palm_px = res.palm_px * self.sx
            self._last_res = res

        if res.found and res.palm_uv is not None:
            now0 = time.time()
            if self._last_raw_uv is not None and now0 > self._last_raw_t:
                d = math.hypot(res.palm_uv[0] - self._last_raw_uv[0],
                               res.palm_uv[1] - self._last_raw_uv[1])
                self._last_speed = 0.5 * (d / max(now0 - self._last_raw_t, 1e-3)) \
                    + 0.5 * self._last_speed
            self._last_raw_t = now0
            if (self._last_speed <= config.FAST_PX_S and jump_rejected(
                    res.palm_uv, self._last_raw_uv, config.MAX_JUMP_PX)):
                res.found = False
            else:
                self._last_raw_uv = res.palm_uv
                self._last_palm_px = res.palm_px

        found_uv = res.palm_uv if (res.found and res.palm_uv) else None
        bridged_conf = 0.3
        if found_uv is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (config.LK_MAX_FRAMES > 0 and self._lk_point is not None
                    and self._prev_gray is not None
                    and self._lk_frames < config.LK_MAX_FRAMES):
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(
                    self._prev_gray, gray, self._lk_point, None,
                    winSize=(31, 31), maxLevel=4,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              20, 0.03))
                if st is not None and bool(st[0, 0]):
                    u, v = float(nxt[0, 0, 0]), float(nxt[0, 0, 1])
                    if 0 <= u < W and 0 <= v < H:
                        self._lk_point = nxt
                        self._lk_frames += 1
                        found_uv = (u, v)
            self._prev_gray = gray
        else:
            self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._lk_point = np.array([[found_uv]], dtype=np.float32)
            self._lk_frames = 0

        held = False
        pos = None
        inten = 0.0
        z_show = None
        conf = res.confidence if res.found else bridged_conf
        if found_uv is not None:
            z_est, _ = depth_from_palm_size(
                intr.fx, res.palm_px if res.found else self._last_palm_px,
                real_m=config.PALM_REAL_M, z_min=config.Z_MIN_M,
                z_max=config.Z_MAX_M, fallback=config.FIXED_Z_M)
            ls = palm_to_light(found_uv[0], found_uv[1], intr, time.time(),
                               fixed_z=z_est, d_ref=config.D_REF_M, i0=config.I0,
                               confidence=conf)
            now = time.time()
            if self.use_oneeuro:
                edge = edge_factor(found_uv[0], found_uv[1], W, H)
                damp = 1.0 - 0.5 * edge * (1.0 - conf)
                raw = (ls.position_camera_m if self._last_light_pos is None
                       else damp * ls.position_camera_m
                       + (1.0 - damp) * self._last_light_pos)
                smooth = self.oneeuro(raw, now)
            else:
                edge = edge_factor(found_uv[0], found_uv[1], W, H)
                alpha = adaptive_alpha(config.EMA_ALPHA, conf, edge)
                smooth = self.ema.update_dynamic(ls.position_camera_m, alpha)
            pos = smooth
            inten = compute_intensity(smooth, d_ref=config.D_REF_M, i0=config.I0)
            z_show = float(smooth[2])
            self._last_light_pos = smooth
            self._last_seen_t = now
            held = not res.found  # LK coast counts as held
        else:
            if (self._last_light_pos is not None
                    and (time.time() - self._last_seen_t) < config.HOLD_LAST_S):
                held = True
                pos = self._last_light_pos
                inten = compute_intensity(pos, d_ref=config.D_REF_M, i0=config.I0)
                z_show = float(pos[2])
            else:
                self.ema.reset()
                self.oneeuro.reset()
                self._last_light_pos = None
                self._last_raw_uv = None
                self._lk_point = None

        det_fps = self.det_meter.tick()
        pkt = LightPacket(
            frame_id=self._frame_id, timestamp_s=time.time(),
            found=bool(res.found), held=held,
            active=pos is not None,
            palm_uv=[float(found_uv[0]), float(found_uv[1])] if found_uv else None,
            position_camera_m=[float(x) for x in pos] if pos is not None else None,
            intensity=float(inten), z_m=z_show, confidence=float(conf),
            backend=self.tracker.backend_name, det_fps=float(det_fps))
        self._frame_id += 1
        self._last_packet = pkt
        return pkt

    def to_dict(self, pkt: LightPacket | None = None) -> dict:
        pkt = pkt if pkt is not None else self.latest
        if pkt is None:
            return {"active": False, "found": False}
        return asdict(pkt)
