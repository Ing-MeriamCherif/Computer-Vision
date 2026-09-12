#!/usr/bin/env python3
"""Compare optional OpenCV flow scales on a deterministic translated texture."""

from __future__ import annotations

import argparse
import json
import time
import numpy as np

from geometry import OpenCVFlowProvider


def run(width: int = 160, height: int = 96, dx: float = 2.0, dy: float = 0.0, iterations: int = 3) -> dict:
    yy, xx = np.indices((height, width), dtype=np.float32)
    texture = (127 + 60 * np.sin(xx * .19) * np.cos(yy * .13) + 20 * np.sin((xx + yy) * .07)).clip(0, 255).astype(np.uint8)
    current = np.roll(texture, (int(round(dy)), int(round(dx))), axis=(0, 1))
    result = {}
    for method in ("dis", "farneback"):
        for scale in (1.0, 0.5, 0.25):
            key = f"{method}_{scale:g}"
            try:
                provider = OpenCVFlowProvider(method=method, flow_scale=scale)
                samples, errors = [], []
                for _ in range(iterations):
                    start = time.perf_counter()
                    motion = provider.compute(texture, current, 0, 1, 1 / 30)
                    samples.append((time.perf_counter() - start) * 1000)
                    valid = motion.valid_mask
                    expected = np.array([dx, dy], np.float32)
                    endpoint = np.linalg.norm(motion.forward_flow - expected, axis=-1)
                    errors.append(endpoint[valid])
                values = np.concatenate(errors) if errors else np.empty(0)
                result[key] = {"status": "available", "runtime_ms": float(np.mean(samples)), "epe_mean": float(values.mean()), "epe_median": float(np.median(values)), "epe_p95": float(np.percentile(values, 95))}
            except RuntimeError as exc:
                result[key] = {"status": "skipped", "reason": str(exc)}
    return {"resolution": [width, height], "translation": [dx, dy], "iterations": iterations, "results": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    print(json.dumps(run(iterations=args.iterations), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
