"""Print the depth matrix from a live camera frame.

Usage:
    python -m depth.depth_matrix --camera-index 0 --input-size 420

Captures one frame, runs depth inference, and prints a grid of
depth values (0-1 range) downsampled to fit the terminal.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from depth.model import DepthModel


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Camera {index} did not open.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(30):
        cap.read()
    return cap


def main():
    parser = argparse.ArgumentParser(description="Print depth matrix")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--input-size", type=int, default=420)
    parser.add_argument("--grid", type=int, default=20,
                        help="Grid rows to display (downsampled from full H)")
    args = parser.parse_args()

    cap = open_camera(args.camera_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit("Failed to capture frame.")

    model = DepthModel(backend="depth_anything_v2_small", device="cuda",
                       input_size=args.input_size, fp16=True)
    model.warmup(iterations=3)
    state = model.infer(frame)

    depth = state.depth_map
    h, w = depth.shape

    # Downsample to grid rows
    rows = args.grid
    cols = int(w / h * rows * 2)  # keep aspect ratio
    step_h = max(1, h // rows)
    step_w = max(1, w // cols)
    small = depth[::step_h, ::step_w]

    print(f"\nDepth matrix: {h}x{w} -> displayed {small.shape[0]}x{small.shape[1]}")
    print(f"Values: 0.0 (closest) ... 1.0 (farthest)\n")

    # Header
    print("       ", end="")
    for c in range(small.shape[1]):
        print(f"{c:5}", end="")
    print()

    for r in range(small.shape[0]):
        print(f"row{r:2} ", end="")
        for c in range(small.shape[1]):
            v = small[r, c]
            print(f" {v:.2f}", end="")
        print()

    # Also print full matrix stats
    print(f"\nFull matrix stats:")
    print(f"  shape: {depth.shape}")
    print(f"  min:   {depth.min():.4f}")
    print(f"  max:   {depth.max():.4f}")
    print(f"  mean:  {depth.mean():.4f}")
    print(f"  std:   {depth.std():.4f}")

    # Show as 8-bit image too
    depth_u8 = (np.clip(depth, 0.0, 1.0) * 255).astype(np.uint8)
    print(f"\nAs 0-255 (multiplied by 255):")
    print(f"  min:   {depth_u8.min()}")
    print(f"  max:   {depth_u8.max()}")
    print(f"  mean:  {depth_u8.mean():.1f}")


if __name__ == "__main__":
    main()
