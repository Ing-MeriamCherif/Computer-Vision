"""10-minute stress test for the async depth pipeline.

Runs headless (no GUI) and logs stats every second.
Checks for: crashes, memory growth, deadlocks, stale depth.

Usage:
    python -m depth.stress_test --camera-index 0 --duration 600
    python -m depth.stress_test --camera-index 0 --duration 60   # quick test
"""

from __future__ import annotations

import argparse
import gc
import threading
import time

import numpy as np

from depth.async_depth import DepthWorker, LatestDepthBuffer, LatestFrameBuffer
from depth.config import DEPTH_CONFIG
from depth.model import DepthModel


def camera_capture_loop(cap, frame_buffer: LatestFrameBuffer, running: threading.Event):
    while running.is_set():
        ok, frame = cap.read()
        if ok and frame is not None:
            frame_buffer.put(frame)
        else:
            time.sleep(0.005)


def open_camera(index: int):
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Camera {index} did not open.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(30):
        cap.read()
    return cap


def get_vram_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Async depth stress test")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--input-size", type=int, default=DEPTH_CONFIG.input_size)
    parser.add_argument("--device", default=DEPTH_CONFIG.device)
    parser.add_argument("--metric", action="store_true", default=DEPTH_CONFIG.metric)
    parser.add_argument("--duration", type=int, default=600,
                        help="Test duration in seconds (default: 600 = 10 min)")
    parser.add_argument("--max-depth-age-ms", type=float, default=200,
                        help="Reject depth older than this (ms)")
    args = parser.parse_args()

    MAX_AGE = args.max_depth_age_ms / 1000.0

    cap = open_camera(args.camera_index)
    model = DepthModel(backend=DEPTH_CONFIG.backend, device=args.device,
                       input_size=args.input_size, fp16=DEPTH_CONFIG.fp16,
                       metric=args.metric)

    print("Loading model...", end=" ", flush=True)
    model.warmup(iterations=3)
    print("ready.")

    frame_buffer = LatestFrameBuffer()
    depth_buffer = LatestDepthBuffer()
    worker = DepthWorker(model, frame_buffer, depth_buffer)
    worker.start()

    running = threading.Event()
    running.set()
    cam_thread = threading.Thread(target=camera_capture_loop,
                                  args=(cap, frame_buffer, running), daemon=True)
    cam_thread.start()

    # Counters
    rendered = 0
    stale_rejected = 0
    camera_fails = 0
    vram_start = get_vram_mb()
    vram_peak = vram_start
    start_time = time.time()
    last_report = start_time

    print(f"Stress test: {args.duration}s | camera {args.camera_index} | "
          f"{args.input_size}x{args.input_size} | metric={args.metric}")
    print(f"Max depth age: {args.max_depth_age_ms:.0f}ms")
    print("-" * 80)

    try:
        while True:
            now = time.time()
            elapsed = now - start_time

            if elapsed >= args.duration:
                break

            state = depth_buffer.get()
            frame = frame_buffer.get()

            if frame is None or state is None:
                camera_fails += 1
                time.sleep(0.001)
                continue

            # Freshness check
            age = now - state.timestamp
            if age > MAX_AGE:
                stale_rejected += 1
                time.sleep(0.001)
                continue

            rendered += 1

            # Track VRAM
            vram_now = get_vram_mb()
            if vram_now > vram_peak:
                vram_peak = vram_now

            # Report every second
            if now - last_report >= 1.0:
                fps = rendered / (now - start_time)
                w_stats = worker.stats_dict()
                vram_delta = vram_now - vram_start
                print(f"[{elapsed:6.0f}s] "
                      f"rendered={rendered:5d}  "
                      f"fps={fps:5.1f}  "
                      f"worker_proc={w_stats['processed']:5d}  "
                      f"stale={stale_rejected:4d}  "
                      f"cam_fail={camera_fails:4d}  "
                      f"vram={vram_now:.0f}MB(+{vram_delta:.0f})  "
                      f"age={age*1000:.0f}ms  "
                      f"inf={state.inference_ms:.0f}ms  "
                      f"worker={'OK' if w_stats['alive'] else 'DEAD'}")
                last_report = now

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        running.clear()
        worker.stop()
        cam_thread.join(timeout=2.0)
        cap.release()

    # Final report
    total_time = time.time() - start_time
    fps_avg = rendered / total_time if total_time > 0 else 0
    w_stats = worker.stats_dict()

    print("=" * 80)
    print(f"STRESS TEST COMPLETE — {total_time:.0f}s")
    print(f"  Camera open:       {'YES' if cap.isOpened() else 'NO (released)'}")
    print(f"  Worker alive:      {'YES' if w_stats['alive'] else 'NO (stopped)'}")
    print(f"  Frames rendered:   {rendered}")
    print(f"  Average FPS:       {fps_avg:.1f}")
    print(f"  Worker processed:  {w_stats['processed']}")
    print(f"  Stale rejected:    {stale_rejected}")
    print(f"  Camera failures:   {camera_fails}")
    print(f"  VRAM start:        {vram_start:.0f} MB")
    print(f"  VRAM peak:         {vram_peak:.0f} MB")
    print(f"  VRAM delta:        {vram_peak - vram_start:.0f} MB")
    print(f"  Memory leak:       {'YES (VRAM grew)' if vram_peak - vram_start > 50 else 'NO'}")
    print("=" * 80)

    # Pass/fail
    issues = []
    if w_stats["alive"]:
        issues.append("Worker still alive after stop()")
    if rendered == 0:
        issues.append("No frames rendered")
    if vram_peak - vram_start > 100:
        issues.append(f"Possible memory leak: +{vram_peak - vram_start:.0f}MB")
    if stale_rejected > rendered * 0.5:
        issues.append(f"High stale rate: {stale_rejected}/{rendered + stale_rejected}")

    if issues:
        print("\nISSUES:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("\nALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
