"""Live camera benchmark — sweep resolutions with a real-time preview.

Usage:
    python -m depth.camera_benchmark_live --camera-index 0 --resolutions 336,420,518

Opens the camera once, then for each resolution:
  1. Loads the model at that size (with a short pause for model setup).
  2. Runs N frames with a live camera | depth preview and real-time stats.
  3. Frees the model and moves on.

At the end, prints a comparison table and saves a CSV.
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from depth.benchmark import get_vram_mb
from depth.model import DepthModel


def open_camera(index: int, width: int = 640, height: int = 480):
    """Open a camera with MJPG codec and discard warm-up frames."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Camera {index} did not open.")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    for _ in range(30):
        cap.read()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    print(f"Camera {index}: {w}x{h} @ {fps_cam:.0f}fps fourcc={fourcc_str}")
    return cap


def run_resolution(
    cap: cv2.VideoCapture,
    backend: str,
    device: str,
    fp16: bool,
    res: int,
    num_frames: int,
) -> dict:
    """Benchmark one resolution with a live preview."""
    model = DepthModel(backend=backend, device=device, input_size=res, fp16=fp16)
    print(f"  Loading model @ {res}x{res}...", end=" ", flush=True)
    model.warmup(iterations=3)
    print("ready.")

    latencies = []
    ema_ms = None
    paused = False

    for i in range(num_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Camera stopped.")
            break

        t0 = time.perf_counter()
        state = model.infer(frame)
        ms = (time.perf_counter() - t0) * 1000.0
        ema_ms = ms if ema_ms is None else 0.9 * ema_ms + 0.1 * ms
        latencies.append(ms)

        # Live preview
        depth_u8 = (np.clip(state.depth_map, 0.0, 1.0) * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        if colored.shape[:2] != frame.shape[:2]:
            colored = cv2.resize(colored, (frame.shape[1], frame.shape[0]))
        view = np.hstack([frame, colored])

        # Overlay
        progress = f"{i + 1}/{num_frames}"
        info = f"{res}x{res}  {ema_ms:.1f}ms  {1000.0 / ema_ms:.1f}FPS  [{progress}]  q=skip"
        cv2.putText(view, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("camera_benchmark_live", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print(f"  Skipped at frame {i + 1}/{num_frames}.")
            break

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    vram = get_vram_mb()

    latencies = np.array(latencies) if latencies else np.array([0.0])
    stats = {
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "fps": 1000.0 / float(np.median(latencies)) if np.median(latencies) > 0 else 0.0,
        "vram_mb": vram,
        "num_frames": len(latencies),
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Live camera benchmark")
    parser.add_argument("--backend", default="depth_anything_v2_small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--resolutions", default="336,420,518")
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output", default="camera_benchmark_live_results.csv")
    args = parser.parse_args()

    resolutions = [int(r.strip()) for r in args.resolutions.split(",")]
    cap = open_camera(args.camera_index)

    print(f"\nBenchmarking {len(resolutions)} resolutions × {args.num_frames} frames "
          f"on camera {args.camera_index}\n")

    results = []
    for idx, res in enumerate(resolutions):
        print(f"[{idx + 1}/{len(resolutions)}] Resolution {res}x{res}")
        stats = run_resolution(
            cap, args.backend, args.device, args.fp16, res, args.num_frames,
        )
        stats["backend"] = args.backend
        stats["resolution"] = res
        stats["device"] = args.device
        stats["fp16"] = args.fp16
        stats["source"] = f"camera{args.camera_index}"
        results.append(stats)

        print(f"  median={stats['median_ms']:.1f}ms  p95={stats['p95_ms']:.1f}ms  "
              f"fps={stats['fps']:.1f}  vram={stats['vram_mb']:.0f}MB\n")

    # Summary table
    print("=" * 60)
    print(f"{'Resolution':<12} {'Median':>8} {'P95':>8} {'FPS':>6} {'VRAM':>6}")
    print("-" * 60)
    for r in results:
        print(f"{r['resolution']:<12} {r['median_ms']:>7.1f}ms {r['p95_ms']:>7.1f}ms "
              f"{r['fps']:>5.1f} {r['vram_mb']:>5.0f}MB")
    print("=" * 60)

    # CSV
    fieldnames = ["backend", "resolution", "device", "fp16", "source",
                  "median_ms", "p95_ms", "fps", "vram_mb", "num_frames"]
    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
