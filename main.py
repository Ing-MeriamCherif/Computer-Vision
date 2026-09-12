"""Person 1 Phase-1 entry: hand-only baseline + light vector + FPS profile.

Usage:
  python main.py --mode hand-only [--no-camera] [--max-frames 300] [--save-json results/baseline.json]
  python main.py --mode profile   # same loop + prints StageProfiler summary

 contract note (Person 4): LightState here uses +X right, +Y down, +Z fwd meters.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np

import config
from filters import EMAFilter
from hand_tracker import create_tracker
from light_vector import depth_from_palm_size, palm_to_light
from normals_stub import depth_to_normals_sobel, dummy_depth, normals_to_rgb
from utils import FPSMeter, Intrinsics, StageProfiler
from validation_render import apply_warm_tint, draw_light_arrow, make_grid


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
            t_cap = time.perf_counter()

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Downscaled inference + frame-skip: reuse last result on skip frames.
            n = max(1, config.HAND_DETECT_EVERY_N)
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
                smooth = ema.update(ls.position_camera_m)
                ls.position_camera_m = smooth
                last_light = ls
                last_seen_t = time.time()
                held = False
            else:
                # Hold last stable light briefly (edge/out-of-frame), then drop.
                if last_light is not None and (time.time() - last_seen_t) < config.HOLD_LAST_S:
                    held = True
                else:
                    ema.reset()
                    last_light = None
                    held = False
            t_vec = time.perf_counter()
            det_fps = det_meter.tick()  # detection path only, no stub/render cost

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
            tinted = apply_warm_tint(frame_bgr, 0.4 + intensity)
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
