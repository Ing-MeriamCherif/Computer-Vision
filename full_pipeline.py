"""Single linked pipeline: capture -> hand torch -> depth -> normals -> renderer.

GPU-inference design (works on CPU for correctness check):
- depth runs in a background DepthWorker (latest-frame only, never blocks);
- hand runs in the main loop via TorchPipeline (frame injected, no 2nd capture);
- normals run on the latest depth (age-gated);
- renderer consumes a validated RenderPacket every frame (latest-only).

  python full_pipeline.py [--no-show --frames 60 --snapshot-every 20]
  GPU:  FULL_DEPTH_DEVICE=cuda FULL_DEPTH_FP16=1 FULL_RENDER_PROFILE=balanced
  CPU check: defaults below (relative depth mapped to meters, SAFE renderer).

Keys (window): 1=RGB 4=Lambertian 6=Final 7=Shadow, q=quit.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from contracts.render_types import DepthFrame, Light, LightState, NormalFrame, RenderPacket
from depth.async_depth import DepthWorker, LatestDepthBuffer, LatestFrameBuffer
from depth.config import DEPTH_CONFIG
from depth.model import DepthModel
from normal_translator import recipe
from pipeline import TorchPipeline
from renderer.config import (ShadowConfig, ShadowQualityProfile,
                             StageQualityProfile, VolumetricConfig,
                             stage_quality_settings)
from renderer.renderer import DebugMode, Renderer
from utils import FPSMeter, StageProfiler


def cfg(key: str, default: str) -> str:
    return os.getenv(key, default)


def depth_to_meters(state) -> np.ndarray:
    """Contract-safe meters: metric passthrough, else relative mapped to volume."""
    d = state.depth_map.astype(np.float64)
    if state.scale_mode == "metric":
        out = d
    else:
        d -= d.min()
        mx = d.max()
        out = 0.5 + 4.5 * (1.0 - d / (mx + 1e-9)) if mx > 1e-9 else np.full_like(d, 2.0)
    return np.clip(out, 0.05, 50.0).astype(np.float32)


def edge_mask(depth_m: np.ndarray, tau: float = 0.08) -> np.ndarray:
    gx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
    return ((np.abs(gx) + np.abs(gy)) > tau)


def build_packet(frame_bgr, torch_pkt, depth_m, fx, fy, cx, cy,
                 frame_id: int) -> RenderPacket:
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ts = time.time()
    depth = DepthFrame(depth_m=depth_m.astype(np.float32),
                       valid_mask=np.ones((h, w), bool),
                       fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
                       timestamp_s=ts)
    normals = recipe(depth_m, fx, fy, cx, cy, config.NORMALS_METHOD,
                     config.NORMALS_SCALE, config.NORMALS_BILATERAL_D)
    normals = (normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-9))
    normals = normals.astype(np.float32)
    lights = []
    if torch_pkt is not None and torch_pkt.active and torch_pkt.position_camera_m:
        lights.append(Light(
            position_camera_m=np.array(torch_pkt.position_camera_m, dtype=np.float32),
            color_rgb=np.array([1.0, 0.85, 0.7], dtype=np.float32),
            intensity=float(max(torch_pkt.intensity, 0.0)),
            active=True, confidence=float(max(0.0, min(1.0, torch_pkt.confidence)))))
    return RenderPacket(rgb=rgb, depth=depth,
                        normals=NormalFrame(normals_camera=normals,
                                            valid_mask=np.ones((h, w), bool),
                                            timestamp_s=ts,
                                            edge_mask=edge_mask(depth_m)),
                        lights=LightState(lights=lights, timestamp_s=ts),
                        frame_id=frame_id, timestamp_s=ts)


def make_context(w: int, h: int, show: bool):
    import glfw
    import moderngl
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    if not show:
        glfw.window_hint(glfw.VISIBLE, False)
    win = glfw.create_window(w, h, "full pipeline (q=quit)", None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError("glfw window failed")
    glfw.make_context_current(win)
    return glfw, win, moderngl.create_context(require=330)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--snapshot-every", type=int, default=20)
    ap.add_argument("--mode", default=os.getenv("FULL_RENDER_MODE", "final"))
    args = ap.parse_args()

    W, H = config.CAMERA_WIDTH, config.CAMERA_HEIGHT
    fx, fy, cx, cy = config.estimate_intrinsics(W, H)
    prof = StageProfiler()
    fpsm = FPSMeter()
    modes = {"rgb": DebugMode.RGB, "lambertian": DebugMode.LAMBERTIAN,
             "final": DebugMode.FINAL, "shadow": DebugMode.SHADOW_MASK}
    mode = modes.get(args.mode.lower(), DebugMode.FINAL)

    # hand torch (frame injected; mock backend guarantees an active light
    # for the CPU correctness check — real hand via HAND_BACKEND=auto)
    torch_pipe = TorchPipeline(width=W, height=H)
    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    assert cap.isOpened(), "camera failed"

    # depth worker (async, latest-only; sync warmup once for first packet)
    use_metric = cfg("FULL_DEPTH_METRIC", "0") == "1"
    device = cfg("FULL_DEPTH_DEVICE",
                 "cuda" if DEPTH_CONFIG.device == "cuda" else "cpu")
    model = DepthModel(backend=DEPTH_CONFIG.backend, device=device,
                       input_size=DEPTH_CONFIG.input_size,
                       fp16=cfg("FULL_DEPTH_FP16", "1") == "1",
                       metric=use_metric)
    print(f"[full] depth {DEPTH_CONFIG.backend} device={device} "
          f"metric={use_metric} (sync warmup...)")
    ok, f0 = cap.read()
    f0 = cv2.resize(f0, (W, H))
    state0 = model.infer(f0)
    frame_buf, depth_buf = LatestFrameBuffer(), LatestDepthBuffer()
    depth_buf.put(state0)
    worker = DepthWorker(model, frame_buf, depth_buf)
    worker.start()

    # renderer (SAFE profile on CPU check; balanced/high on GPU)
    profile = cfg("FULL_RENDER_PROFILE", "safe").lower()
    settings = stage_quality_settings(StageQualityProfile(profile))
    glfw, win, ctx = make_context(W, H, show=not args.no_show)
    pkt0 = build_packet(f0, torch_pipe.step(frame_bgr=f0),
                        depth_to_meters(state0), fx, fy, cx, cy, 0)
    renderer = Renderer(ctx, pkt0,
                        shadow_config=settings.shadow,
                        secondary_shadow_mode=settings.secondary_shadow_mode,
                        volumetric_config=settings.volumetric)
    try:
        from renderer.config import ShadowQualityProfile as SQP
        renderer.shadow_configs = (
            ShadowConfig.for_profile(SQP.SAFE), ShadowConfig.for_profile(SQP.SAFE))
    except Exception:
        pass

    os.makedirs("results", exist_ok=True)
    i = 0
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (W, H))
            if config.CAM_MIRROR:
                frame = cv2.flip(frame, 1)
            t_cap = time.perf_counter()

            tpkt = torch_pipe.step(frame_bgr=frame)
            t_hand = time.perf_counter()

            frame_buf.put(frame)
            st = depth_buf.get()
            age_ms = (time.time() - st.timestamp) * 1000 if st else 1e9
            depth_m = depth_to_meters(st)
            t_depth = time.perf_counter()

            pkt = build_packet(frame, tpkt, depth_m, fx, fy, cx, cy, i)
            t_norm = time.perf_counter()

            renderer.render(pkt, mode)
            if not args.no_show:
                glfw.swap_buffers(win)
                glfw.poll_events()
                if glfw.window_should_close(win):
                    break
            else:
                glfw.poll_events()
            t_render = time.perf_counter()

            if args.snapshot_every and i % args.snapshot_every == 0:
                raw = ctx.detect_framebuffer().read(components=3)
                img = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
                cv2.imwrite(f"results/full_snap_{i:04d}.png",
                            cv2.cvtColor(np.flipud(img), cv2.COLOR_RGB2BGR))
            fps = fpsm.tick()
            if i % 10 == 0:
                print(f"f{i} det={tpkt.det_fps:.0f} render-fps={fps:.1f} "
                      f"active={tpkt.active} I={tpkt.intensity:.2f} "
                      f"depth_age={age_ms:.0f}ms", flush=True)
            prof.add("capture", (t_cap - t0) * 1000)
            prof.add("hand", (t_hand - t_cap) * 1000)
            prof.add("depth_wait", (t_depth - t_hand) * 1000)
            prof.add("normals_packet", (t_norm - t_depth) * 1000)
            prof.add("render", (t_render - t_norm) * 1000)
            prof.add("total", (t_render - t0) * 1000)
            i += 1
            if args.frames and i >= args.frames:
                break
            if not args.no_show and glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS:
                break
    finally:
        worker.stop()
        torch_pipe.stop()
        cap.release()
        glfw.terminate()
    print(f"\n[full] frames={i}")
    for stage, s in prof.summary().items():
        print(f"  {stage:14s} avg={s['avg_ms']:7.2f} p95={s['p95_ms']:7.2f}")
    print("snapshots in results/full_snap_*.png")


if __name__ == "__main__":
    main()
