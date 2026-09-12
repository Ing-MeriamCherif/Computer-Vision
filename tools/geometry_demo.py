#!/usr/bin/env python3
"""Run a camera-space reconstruction smoke demo without a neural depth model."""

from __future__ import annotations

import argparse
import json

from geometry import CameraModel, backproject_depth
from geometry.debug import export_ply, fit_plane, fronto_parallel_plane, step_depth, tilted_plane


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--shape", choices=("plane", "tilted", "step"), default="tilted")
    parser.add_argument("--ply", help="optional output PLY path")
    args = parser.parse_args()
    camera = CameraModel(args.width, args.height, args.width * 0.9, args.height * 0.9, (args.width - 1) / 2, (args.height - 1) / 2)
    depth = {"plane": fronto_parallel_plane, "tilted": tilted_plane, "step": step_depth}[args.shape](camera)
    points, valid = backproject_depth(depth, camera, "relative")
    result = {"shape": args.shape, "resolution": [args.width, args.height], "valid_points": int(valid.sum())}
    if args.shape != "step":
        fit = fit_plane(points, valid)
        result.update(rms_plane_residual=fit.rms_residual, max_plane_residual=fit.max_abs_residual)
    if args.ply:
        export_ply(args.ply, points, valid)
        result["ply"] = args.ply
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
