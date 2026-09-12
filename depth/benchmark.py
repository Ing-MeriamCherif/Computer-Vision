"""Benchmark depth backends across resolutions.

Usage:
    python -m depth.benchmark --backend depth_anything_v2_small --device cuda
    python -m depth.benchmark --backend depth_anything_v2_small --resolutions 336,420,518

Outputs a CSV with: backend, resolution, median_ms, p95_ms, fps, vram_mb.
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np
import torch

from depth.model import DepthModel


DEFAULT_RESOLUTIONS = [336, 420, 518]


def get_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def benchmark_single(
    model: DepthModel,
    num_frames: int = 50,
    warmup_frames: int = 5,
) -> dict:
    """Run inference loop and collect timing stats."""
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup_frames):
        model.infer(dummy)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Reset VRAM tracker
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    latencies = []
    for _ in range(num_frames):
        t0 = time.perf_counter()
        model.infer(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

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
    parser = argparse.ArgumentParser(description="Benchmark depth backends")
    parser.add_argument("--backend", default="depth_anything_v2_small", help="Backend name")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 inference")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--resolutions", default="336,420,518", help="Comma-separated input sizes")
    parser.add_argument("--num-frames", type=int, default=50, help="Frames per resolution")
    parser.add_argument("--output", default="benchmark_results.csv", help="Output CSV path")
    args = parser.parse_args()

    resolutions = [int(r.strip()) for r in args.resolutions.split(",")]
    results = []

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

        stats = benchmark_single(model, num_frames=args.num_frames)
        stats["backend"] = args.backend
        stats["resolution"] = res
        stats["device"] = args.device
        stats["fp16"] = args.fp16
        results.append(stats)

        print(f"median={stats['median_ms']:.1f}ms  p95={stats['p95_ms']:.1f}ms  "
              f"fps={stats['fps']:.1f}  vram={stats['vram_mb']:.0f}MB")

        # Free GPU/CPU memory so resolutions don't contaminate each other.
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write CSV
    fieldnames = ["backend", "resolution", "device", "fp16",
                  "median_ms", "p95_ms", "fps", "vram_mb", "num_frames"]
    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
