"""Phase-2 D2NT test: Sobel vs d2nt_basic/v2/v3 on KNOWN geometry.

Scene (GT normals analytic): tilted background plane + fronto-parallel box.
Metrics per method: mean/median angular error overall + flat interior +
3px edge band, plus median latency (20 runs). Saves JSON + normal viz.
Decision rule (contexte.md): adopt D2NT iff edge error drops clearly and
latency fits budget (<5ms target, <10ms max).

  python tools/bench_normals.py [--wh 640 480] [--noise 0.0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from normal_translator import translate


def build_scene(w: int, h: int, fx: float, fy: float, cx: float, cy: float,
                noise: float = 0.0):
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    # background: tilted plane Z = 3 + 0.4x + 0.2y -> n = (-0.4,-0.2,1)/|..|
    depth = 3.0 + 0.4 * x + 0.2 * y
    gt = np.zeros((h, w, 3))
    nb = np.array([-0.4, -0.2, 1.0])
    gt[:] = nb / np.linalg.norm(nb)
    # foreground box, front face Z=1.5 -> n=(0,0,1)
    x0, x1, y0, y1 = int(w * .35), int(w * .65), int(h * .3), int(h * .7)
    depth[y0:y1, x0:x1] = 1.5
    gt[y0:y1, x0:x1] = (0, 0, 1)
    if noise > 0:
        depth += np.random.randn(h, w).astype(np.float64) * noise
    # edge band: pixels within 3px of the box border
    edge = np.zeros((h, w), bool)
    edge[y0 - 3:y1 + 3, x0 - 3:x1 + 3] = True
    edge[y0 + 3:y1 - 3, x0 + 3:x1 - 3] = False
    return depth, gt, edge


def ang_err(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    d = np.clip(np.sum(est * gt, axis=-1), -1, 1)
    return np.degrees(np.arccos(d))


def bench(method: str, depth, fx, fy, cx, cy, runs: int = 20):
    translate(depth, fx, fy, cx, cy, method)  # warmup
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        n = translate(depth, fx, fy, cx, cy, method)
        ts.append((time.perf_counter() - t0) * 1000)
    return n, float(np.median(ts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wh", nargs=2, type=int, default=[640, 480])
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--out", default="results/normals_bench.json")
    args = ap.parse_args()
    w, h = args.wh
    fx = fy = (w / 2) / np.tan(np.radians(30))
    cx, cy = w / 2, h / 2
    depth, gt, edge = build_scene(w, h, fx, fy, cx, cy, args.noise)
    flat = ~edge

    report = {"res": [w, h], "noise_m": args.noise, "methods": {}}
    for m in ["sobel", "d2nt_basic", "d2nt_v2", "d2nt_v3"]:
        try:
            n, med = bench(m, depth, fx, fy, cx, cy)
        except Exception as e:
            report["methods"][m] = {"error": str(e)}
            print(f"{m:10s} FAILED: {e}")
            continue
        e_all, e_flat, e_edge = ang_err(n, gt), ang_err(n[flat], gt[flat]), ang_err(n[edge], gt[edge])
        r = {"lat_ms": round(med, 2),
             "mean_all": round(float(e_all.mean()), 2),
             "mean_flat": round(float(e_flat.mean()), 2),
             "mean_edge": round(float(e_edge.mean()), 2),
             "med_edge": round(float(np.median(e_edge)), 2)}
        report["methods"][m] = r
        print(f"{m:10s} {med:6.2f}ms  mean all/flat/edge = "
              f"{r['mean_all']:.2f}/{r['mean_flat']:.2f}/{r['mean_edge']:.2f} deg")
        vis = ((n + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(f"results/normals_{m}.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
