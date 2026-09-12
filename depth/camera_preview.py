"""Live camera preview to find a working camera index.

Usage:
    python -m depth.camera_preview --camera-index 0

Press q in the window to quit.
"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Live camera preview")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    import cv2

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(
            f"Camera {args.camera_index} did not open. "
            "Run `ls /dev/video*` and try each device number as --camera-index."
        )

    # Force MJPG codec — YUYV is slow and can produce black frames on USB cams.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Discard first frames so camera warms up.
    for _ in range(30):
        cap.read()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {args.camera_index}: {w}x{h} — press q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera stopped delivering frames.")
                break
            cv2.imshow(f"camera {args.camera_index} (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
