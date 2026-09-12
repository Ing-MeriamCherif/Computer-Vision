"""Benchmark depth backends on live camera frames.

Usage:
    python -m depth.camera_benchmark --camera-index 0 --resolutions 336,420,518
    python -m depth.camera_benchmark --camera-index 0 --show

Frames are captured up front so the timed loop measures inference, not USB
capture. Camera frames are BGR, matching the backend's expected input.
With --show, a live camera | depth preview window opens during benchmarking
(needs a display; visualization runs outside the timed region).
Outputs a CSV with: backend, resolution, device, fp16, source, median_ms,
p95_ms, fps, vram_mb, num_frames.
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np
import torch

from depth.benchmark import get_vram_mb
from depth.model import DepthModel


def capture_camera_frames(
    index: int,
    num_frames: int,
    discard: int = 30,
) -> list:
    """Grab real BGR frames from a camera.

    Raises RuntimeError with a clear message if the camera is unavailable.
    """
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {index}. "
            "Check it exists (e.g. `ls /dev/video*`) and is not used by another app."
        )

    # Force MJPG codec — YUYV is slow and can produce black frames on USB cams.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        # Discard frames so camera warms up and auto-exposure settles.
        for _ in range(discard):
            cap.read()
        frames = []
        while len(frames) < num_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Camera {index} stopped delivering frames "
                    f"after {len(frames)}/{num_frames}."
                )
            frames.append(frame)
    finally:
        cap.release()
    return frames


def show_depth_preview(frame, depth_map, window="camera_benchmark [camera | depth]") -> bool:
    """Render side-by-side camera + depth colormap. Returns False if no display."""
    import cv2

    try:
        depth_u8 = (np.clip(depth_map, 0.0, 1.0) * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        if colored.shape[:2] != frame.shape[:2]:
            colored = cv2.resize(colored, (frame.shape[1], frame.shape[0]))
        cv2.imshow(window, np.hstack([frame, colored]))
        cv2.waitKey(1)
    except cv2.error:
        return False
    return True


def benchmark_camera(
    model: DepthModel,
    frames: list,
    num_frames: int = 50,
    warmup_frames: int = 5,
    show: bool = False,
) -> dict:
    """Run inference loop over captured frames and collect timing stats."""
    get_frame = lambda i: frames[i % len(frames)]  # noqa: E731

    # Warmup
    for i in range(warmup_frames):
        model.infer(get_frame(i))
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Reset VRAM tracker
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Benchmark (visualization stays outside the timed region)
    show_ok = show
    latencies = []
    for i in range(num_frames):
        frame = get_frame(i)
        t0 = time.perf_counter()
        state = model.infer(frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if show_ok and (i % 5 == 0 or i == num_frames - 1):
            show_ok = show_depth_preview(frame, state.depth_map)
    if show and not show_ok:
        print("  (no display available — continuing headless)")

    latencies = np.array(latencies)
    vram = get_vram_mb()

    return {
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "fps": 1000.0 / float(np.median(latencies)),
        "vram_mb": vram,
        "num_frames": num_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark depth backends on camera frames")
    parser.add_argument("--backend", default="depth_anything_v2_small", help="Backend name")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 inference")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--resolutions", default="336,420,518", help="Comma-separated input sizes")
    parser.add_argument("--num-frames", type=int, default=50, help="Frames per resolution")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for cv2.VideoCapture")
    parser.add_argument("--camera-discard", type=int, default=30,
                        help="Frames to discard so auto-exposure settles")
    parser.add_argument("--show", action="store_true",
                        help="Open a live camera | depth preview window (needs a display)")
    parser.add_argument("--output", default="camera_benchmark_results.csv", help="Output CSV path")
    args = parser.parse_args()

    resolutions = [int(r.strip()) for r in args.resolutions.split(",")]
    results = []

    print(f"Capturing {args.num_frames} frames from camera {args.camera_index}...",
          end=" ", flush=True)
    frames = capture_camera_frames(args.camera_index, args.num_frames,
                                   discard=args.camera_discard)
    h, w = frames[0].shape[:2]
    print(f"done ({w}x{h}).")

    print(f"Benchmarking backend={args.backend}, device={args.device}, fp16={args.fp16}")
    print(f"Resolutions: {resolutions}, Frames per test: {args.num_frames}\n")

    for res in resolutions:
        print(f"Testing resolution {res}x{res}...", end=" ", flush=True)

        model = DepthModel(
            backend=args.backend,
            device=args.device,
            input_size=res,
            fp16=args.fp16,
        )
        model.warmup(iterations=3)

        stats = benchmark_camera(model, frames, num_frames=args.num_frames,
                                   show=args.show)
        stats["backend"] = args.backend
        stats["resolution"] = res
        stats["device"] = args.device
        stats["fp16"] = args.fp16
        stats["source"] = f"camera{args.camera_index}"
        results.append(stats)

        print(f"median={stats['median_ms']:.1f}ms  p95={stats['p95_ms']:.1f}ms  "
              f"fps={stats['fps']:.1f}  vram={stats['vram_mb']:.0f}MB")

        # Free GPU/CPU memory so resolutions don't contaminate each other.
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write CSV
    fieldnames = ["backend", "resolution", "device", "fp16", "source",
                  "median_ms", "p95_ms", "fps", "vram_mb", "num_frames"]
    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")

    if args.show:
        import cv2

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
