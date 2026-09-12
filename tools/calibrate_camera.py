#!/usr/bin/env python3
"""Calibrate a camera from checkerboard images and save JSON intrinsics."""

from __future__ import annotations

import argparse
from pathlib import Path

from geometry.calibration import calibrate_checkerboard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="checkerboard image paths")
    parser.add_argument("--board-width", type=int, default=9)
    parser.add_argument("--board-height", type=int, default=6)
    parser.add_argument("--square-size", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate_checkerboard(args.images, (args.board_width, args.board_height), args.square_size)
    result.camera.save_json(args.output)
    print(f"views={result.views_used} reprojection_error_px={result.reprojection_error_px:.4f} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
