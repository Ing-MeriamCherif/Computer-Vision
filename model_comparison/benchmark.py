"""Compare depth models: Depth Anything V2 vs YOLO26.

Runs dummy-frame benchmarks (no camera needed) and writes a CSV.

Usage:
    python -m model_comparison.benchmark --device cuda --num-frames 50
    python -m model_comparison.benchmark --device cuda --models yolo26n,yolo26s
    python -m model_comparison.benchmark --device cuda --models all
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


def benchmark_model(
    backend: str,
    device: str,
    input_size: int,
    fp16: bool,
    num_frames: int = 50,
    warmup: int = 5,
) -> dict:
    """Benchmark a single depth backend on dummy frames."""
    model = DepthModel(backend=backend, device=device, input_size=input_size, fp16=fp16)

    print(f"  Loading {backend}...", end=" ", flush=True)
    model.warmup(iterations=warmup)
    print("ready.")

    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    latencies = []
    for _ in range(num_frames):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.infer(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)
    vram = get_vram_mb()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "fps": 1000.0 / float(np.median(latencies)),
        "vram_mb": vram,
        "num_frames": num_frames,
    }


MODELS = {
    "da_v2_small": "depth_anything_v2_small",
    "da_v2_base": "depth_anything_v2_base",
    "da_v2_large": "depth_anything_v2_large",
    "yolo26n": "yolo26n_depth",
    "yolo26s": "yolo26s_depth",
    "yolo26m": "yolo26m_depth",
    "yolo26l": "yolo26l_depth",
    "yolo26x": "yolo26x_depth",
}


def main():
    parser = argparse.ArgumentParser(description="Compare depth models")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=420)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--models", default="all",
                        help="Comma-separated: da_v2_small,da_v2_base,yolo26n,yolo26s,yolo26m,all")
    parser.add_argument("--output", default="model_comparison/results.csv")
    args = parser.parse_args()

    if args.models == "all":
        models = MODELS
    else:
        names = [n.strip() for n in args.models.split(",")]
        models = {k: v for k, v in MODELS.items() if k in names}
        if not models:
            raise SystemExit(f"Unknown models: {names}. Available: {list(MODELS.keys())}")

    results = []
    for name, backend in models.items():
        print(f"\n[{name}]")
        try:
            stats = benchmark_model(backend, args.device, args.input_size,
                                    args.fp16, args.num_frames)
            stats["model"] = name
            stats["backend"] = backend
            results.append(stats)
            print(f"  median={stats['median_ms']:.1f}ms  p95={stats['p95_ms']:.1f}ms  "
                  f"fps={stats['fps']:.1f}  vram={stats['vram_mb']:.0f}MB")
        except Exception as e:
            print(f"  FAILED: {e}")

    if not results:
        raise SystemExit("No models succeeded.")

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'Median':>8} {'P95':>8} {'FPS':>6} {'VRAM':>6}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["median_ms"]):
        print(f"{r['model']:<20} {r['median_ms']:>7.1f}ms {r['p95_ms']:>7.1f}ms "
              f"{r['fps']:>5.1f} {r['vram_mb']:>5.0f}MB")
    print("=" * 70)

    # Speedup vs DA V2 Small
    da_small = next((r for r in results if r["model"] == "da_v2_small"), None)
    if da_small:
        print(f"\nSpeedup vs Depth Anything V2 Small ({da_small['median_ms']:.1f}ms):")
        for r in sorted(results, key=lambda x: x["median_ms"]):
            speedup = da_small["median_ms"] / r["median_ms"] if r["median_ms"] > 0 else 0
            print(f"  {r['model']:<20} {speedup:.1f}x")

    # CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "backend", "median_ms", "p95_ms", "fps", "vram_mb", "num_frames"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
