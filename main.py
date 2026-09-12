"""Person 1 Phase-1 entry: hand-only baseline + light vector + FPS profile.

Usage:
  python main.py --mode hand-only [--no-camera] [--max-frames 300] [--save-json results/baseline.json]
  python main.py --mode profile   # same loop + prints StageProfiler summary

 contract note (Person 4): LightState here uses +X right, +Y down, +Z fwd meters.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import cv2
import numpy as np

import config
from filters import (EMAFilter, OneEuroVec3, adaptive_alpha, edge_factor,
                     jump_rejected)
from hand_tracker import create_tracker
from light_vector import compute_intensity, depth_from_palm_size, palm_to_light
from normals_stub import depth_to_normals_sobel, dummy_depth, normals_to_rgb
from utils import FPSMeter, Intrinsics, StageProfiler
from validation_render import apply_torch_glow, draw_light_arrow, make_grid


def draw_landmarks(bgr: np.ndarray, lm_uv: np.ndarray | None) -> np.ndarray:
    out = bgr
    if lm_uv is None:
        return out
    for x, y in lm_uv.astype(int):
        cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)
    return out


def open_camera(w: int, h: int, fps: int):
    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        return None
    return cap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hand-only", choices=["hand-only", "profile"])
    ap.add_argument("--no-camera", action="store_true", help="synthetic frames (no webcam needed)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--save-json", default="")
    ap.add_argument("--show", action="store_true", default=True)
    ap.add_argument("--no-show", dest="show", action="store_false")
    args = ap.parse_args()

    W, H = config.CAMERA_WIDTH, config.CAMERA_HEIGHT
    fx, fy, cx, cy = config.estimate_intrinsics(W, H)
    intr = Intrinsics(fx, fy, cx, cy, W, H)

    tracker = create_tracker()
    ema = EMAFilter(alpha=config.EMA_ALPHA)
    oneeuro = OneEuroVec3(mincutoff=config.ONEEURO_MINCUTOFF,
                          beta=config.ONEEURO_BETA,
                          dcutoff=config.ONEEURO_DCUTOFF)
    use_oneeuro = config.FILTER == "oneeuro"
    prof = StageProfiler()
    fps_meter = FPSMeter()      # full pipeline (display + stubs)
    det_meter = FPSMeter()      # detection path only: capture+hand+vector
    print(f"[main] backend={tracker.backend_name} device={config.DEVICE} {W}x{H}")

    cap = None if args.no_camera else open_camera(W, H, config.CAMERA_FPS)
    if args.no_camera:
        print("[main] synthetic mode (no camera)")
    elif cap is None:
        print("[main] camera open failed -> synthetic fallback")
        args.no_camera = True

    frame_id = 0
    last_light = None
    last_seen_t = 0.0
    last_res = None  # reused on skipped frames (HAND_DETECT_EVERY_N)
    last_raw_uv = None  # for outlier gate
    last_raw_t = 0.0
    last_speed = 0.0  # px/s, drives adaptive detect rate + gate exemption
    last_palm_px = None
    prev_gray = None  # LK bridge state
    lk_point = None
    lk_frames = 0
    det_fps = 0.0
    infer_w, infer_h = config.HAND_INFER_W, config.HAND_INFER_H
    sx, sy = W / infer_w, H / infer_h
    try:
        while True:
            t0 = time.perf_counter()
            if args.no_camera:
                frame_bgr = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
                frame_bgr = cv2.GaussianBlur(frame_bgr, (21, 21), 0)
            else:
                ok, frame_bgr = cap.read()
                if not ok:
                    print("[main] camera read failed, stopping")
                    break
                frame_bgr = cv2.resize(frame_bgr, (W, H))
                if config.CAM_MIRROR:
                    frame_bgr = cv2.flip(frame_bgr, 1)  # mirror: motion matches vector
            t_cap = time.perf_counter()

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Downscaled inference + frame-skip: reuse last result on skip frames.
            # Fast hands disable skipping: a fast palm outruns a stale result.
            n = max(1, config.HAND_DETECT_EVERY_N)
            if last_speed > config.FAST_PX_S:
                n = 1
            if last_res is not None and (frame_id % n) != 0:
                res = last_res
            else:
                small = cv2.resize(rgb, (infer_w, infer_h),
                                   interpolation=cv2.INTER_LINEAR)
                res = tracker.process(small)
                if res.found and res.landmarks_uv is not None:
                    res.landmarks_uv = res.landmarks_uv * np.array([sx, sy])
                    if res.palm_uv is not None:
                        res.palm_uv = (res.palm_uv[0] * sx, res.palm_uv[1] * sy)
                    if res.palm_px is not None:
                        res.palm_px = res.palm_px * sx
                last_res = res
            t_hand = time.perf_counter()

            if res.found and res.palm_uv is not None:
                # Palm speed estimate (px/s) from raw detections.
                now0 = time.time()
                if last_raw_uv is not None and now0 > last_raw_t:
                    d = math.hypot(res.palm_uv[0] - last_raw_uv[0],
                                   res.palm_uv[1] - last_raw_uv[1])
                    inst = d / max(now0 - last_raw_t, 1e-3)
                    last_speed = 0.5 * inst + 0.5 * last_speed
                last_raw_t = now0
                # Outlier gate: teleport in one frame at LOW speed = misdetection.
                # At high speed big jumps are genuine fast motion — never reject.
                if (last_speed <= config.FAST_PX_S and jump_rejected(
                        res.palm_uv, last_raw_uv, config.MAX_JUMP_PX)):
                    res.found = False
                else:
                    last_raw_uv = res.palm_uv
                    last_palm_px = res.palm_px
            if res.found and res.palm_uv is not None:
                if config.USE_SIZE_DEPTH:
                    z_est, _ = depth_from_palm_size(
                        fx, res.palm_px, real_m=config.PALM_REAL_M,
                        z_min=config.Z_MIN_M, z_max=config.Z_MAX_M,
                        fallback=config.FIXED_Z_M)
                else:
                    z_est = config.FIXED_Z_M
                ls = palm_to_light(res.palm_uv[0], res.palm_uv[1], intr,
                                   time.time(), fixed_z=z_est,
                                   d_ref=config.D_REF_M, i0=config.I0,
                                   confidence=res.confidence)
                now = time.time()
                if use_oneeuro:
                    # 1-euro adapts itself to speed; edge/low-conf pre-damp input.
                    edge = edge_factor(res.palm_uv[0], res.palm_uv[1], W, H)
                    damp = 1.0 - 0.5 * edge * (1.0 - res.confidence)
                    raw = (ls.position_camera_m if last_light is None
                           else damp * ls.position_camera_m
                           + (1.0 - damp) * last_light.position_camera_m)
                    smooth = oneeuro(raw, now)
                else:
                    edge = edge_factor(res.palm_uv[0], res.palm_uv[1], W, H)
                    alpha = adaptive_alpha(config.EMA_ALPHA, res.confidence, edge)
                    smooth = ema.update_dynamic(ls.position_camera_m, alpha)
                ls.position_camera_m = smooth
                # intensity follows the SMOOTHED position (torch feel)
                ls.intensity = compute_intensity(smooth, d_ref=config.D_REF_M,
                                                 i0=config.I0)
                last_light = ls
                last_seen_t = now
                held = False
                # LK bridge state follows fresh detections
                lk_point = np.array([[res.palm_uv]], dtype=np.float32)
                lk_frames = 0
            else:
                # LK optical-flow bridge: coast through short detector dropouts
                # (edge exit, motion blur) instead of freezing instantly.
                bridged = False
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                if (config.LK_MAX_FRAMES > 0 and lk_point is not None
                        and prev_gray is not None and lk_frames < config.LK_MAX_FRAMES):
                    # Wide window + deep pyramid: fast hands displace many px
                    # per frame; defaults (21px/3lvl) lose them immediately.
                    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, lk_point, None,
                        winSize=(31, 31), maxLevel=4,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                  20, 0.03))
                    if st is not None and bool(st[0, 0]):
                        u, v = float(nxt[0, 0, 0]), float(nxt[0, 0, 1])
                        if 0 <= u < W and 0 <= v < H:
                            lk_point = nxt
                            lk_frames += 1
                            z_est, _ = depth_from_palm_size(
                                fx, last_palm_px, real_m=config.PALM_REAL_M,
                                z_min=config.Z_MIN_M, z_max=config.Z_MAX_M,
                                fallback=config.FIXED_Z_M)
                            ls = palm_to_light(u, v, intr, time.time(),
                                               fixed_z=z_est, d_ref=config.D_REF_M,
                                               i0=config.I0, confidence=0.3)
                            now = time.time()
                            if use_oneeuro:
                                smooth = oneeuro(ls.position_camera_m, now)
                            else:
                                smooth = ema.update_dynamic(
                                    ls.position_camera_m, config.EMA_ALPHA * 0.4)
                            ls.position_camera_m = smooth
                            ls.intensity = compute_intensity(
                                smooth, d_ref=config.D_REF_M, i0=config.I0)
                            last_light = ls
                            bridged = True
                            held = True
                if not bridged:
                    # Hold last stable light briefly, then drop.
                    if last_light is not None and (time.time() - last_seen_t) < config.HOLD_LAST_S:
                        held = True
                    else:
                        ema.reset()
                        oneeuro.reset()
                        last_light = None
                        last_raw_uv = None
                        lk_point = None
                        held = False
            t_vec = time.perf_counter()
            det_fps = det_meter.tick()  # detection path only, no stub/render cost
            prev_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # depth/normals stub at half-res (display only, excluded from DET fps)
            dh, dw = H // 2, W // 2
            depth_small = dummy_depth(dh, dw)
            normals_small = depth_to_normals_sobel(depth_small, fx / 2, fy / 2)
            depth = cv2.resize(depth_small, (W, H), interpolation=cv2.INTER_NEAREST)
            normals = cv2.resize(normals_small, (W, H), interpolation=cv2.INTER_LINEAR)
            t_norm = time.perf_counter()

            intensity = float(last_light.intensity) if last_light else 0.0
            z_show = float(last_light.position_camera_m[2]) if last_light else None
            palm = res.palm_uv if res.found else None
            if palm is None and held and last_light is not None:
                # project held light back to screen so arrow doesn't vanish at edge
                lx, ly, lz = last_light.position_camera_m
                palm = (float(lz * 0 + intr.cx + lx * intr.fx / max(lz, 1e-3)),
                        float(intr.cy + ly * intr.fy / max(lz, 1e-3)))
            rgb_lm = draw_landmarks(frame_bgr.copy(), res.landmarks_uv)
            tinted = apply_torch_glow(frame_bgr, palm, intensity,
                                      radius_px=config.GLOW_RADIUS_PX)
            tinted = draw_light_arrow(tinted, palm, intensity, z_show, held)
            fps = fps_meter.tick()
            grid = make_grid(rgb_lm, depth, normals_to_rgb(normals), tinted, fps, det_fps)
            t_render = time.perf_counter()

            prof.add("capture", (t_cap - t0) * 1000)
            prof.add("hand", (t_hand - t_cap) * 1000)
            prof.add("vector", (t_vec - t_hand) * 1000)
            prof.add("normals_stub", (t_norm - t_vec) * 1000)
            prof.add("render", (t_render - t_norm) * 1000)
            prof.add("total", (t_render - t0) * 1000)

            if args.show:
                cv2.imshow("person1 phase1 hand-only (q=quit)", grid)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_id += 1
            if args.max_frames and frame_id >= args.max_frames:
                break
    finally:
        tracker.close()
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # opencv-headless: no GUI backend

    summ = prof.summary()
    print(f"\n[profile] frames={frame_id} backend={tracker.backend_name}")
    for stage, s in summ.items():
        print(f"  {stage:12s} avg={s['avg_ms']:6.2f} min={s['min_ms']:6.2f} "
              f"max={s['max_ms']:6.2f} p95={s['p95_ms']:6.2f} n={s['n']}")
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({"backend": tracker.backend_name, "device": config.DEVICE,
                       "wh": [W, H], "stages": summ}, f, indent=2)
        print(f"[profile] saved {args.save_json}")


if __name__ == "__main__":
    main()
