"""Compare Depth Anything V2 models — Small vs Base vs Large.

Runs dummy-frame benchmarks (no camera needed) and writes a CSV.

Usage:
    python -m model_comparison.benchmark --device cuda --num-frames 50
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


MODELS = {
    "depth_anything_v2_small": "depth-anything/Depth-Anything-V2-Small-hf",
    "depth_anything_v2_base": "depth-anything/Depth-Anything-V2-Base-hf",
    "depth_anything_v2_large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def benchmark_model(
    model_id: str,
    device: str,
    input_size: int,
    fp16: bool,
    num_frames: int = 50,
    warmup: int = 5,
) -> dict:
    """Benchmark a single HF depth model."""
    from transformers import pipeline

    dtype = torch.float16 if fp16 and device == "cuda" else torch.float32
    dev = 0 if device == "cuda" else -1

    print(f"  Loading {model_id}...", end=" ", flush=True)
    pipe = pipeline(
        task="depth-estimation",
        model=model_id,
        device=dev,
        torch_dtype=dtype,
    )
    print("ready.")

    from PIL import Image

    dummy = Image.fromarray(
        np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
    )

    # Warmup
    for _ in range(warmup):
        pipe(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    latencies = []
    for _ in range(num_frames):
        t0 = time.perf_counter()
        pipe(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)
    vram = get_vram_mb()

    del pipe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "fps": 1000.0 / float(np.median(latencies)),
        "vram_mb": vram,
        "num_frames": num_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare depth models")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=420)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--models", default="all",
                        help="Comma-separated model names or 'all' (small,base,large)")
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
    for name, model_id in models.items():
        print(f"[{name}]")
        stats = benchmark_model(model_id, args.device, args.input_size,
                                args.fp16, args.num_frames)
        stats["model"] = name
        stats["model_id"] = model_id
        results.append(stats)
        print(f"  median={stats['median_ms']:.1f}ms  p95={stats['p95_ms']:.1f}ms  "
              f"fps={stats['fps']:.1f}  vram={stats['vram_mb']:.0f}MB\n")

    # Summary
    print("=" * 65)
    print(f"{'Model':<30} {'Median':>8} {'P95':>8} {'FPS':>6} {'VRAM':>6}")
    print("-" * 65)
    for r in results:
        print(f"{r['model']:<30} {r['median_ms']:>7.1f}ms {r['p95_ms']:>7.1f}ms "
              f"{r['fps']:>5.1f} {r['vram_mb']:>5.0f}MB")
    print("=" * 65)

    # CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "model_id", "median_ms", "p95_ms", "fps", "vram_mb", "num_frames"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
