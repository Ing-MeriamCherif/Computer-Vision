#!/usr/bin/env python3
"""Capture the host webcam and validate every interface pipeline mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from tools.webcam_geometry_app import WebcamGeometrySession


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--artifacts", default="artifacts/webcam-e2e")
    args = parser.parse_args()
    output = Path(args.artifacts)
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open webcam {args.device}")
    ok, bgr = capture.read()
    capture.release()
    if not ok or bgr is None:
        raise RuntimeError(f"cannot read a frame from {args.device}")
    cv2.imwrite(str(output / "webcam.jpg"), bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    session = WebcamGeometrySession()
    results = {}
    for mode in ("CUDA current geometry", "Phase 1-4 temporal", "Phase 5 persistent"):
        depth, normals, confidence, persistent, stats = session.process(rgb, mode, True)
        slug = mode.lower().replace(" ", "-")
        cv2.imwrite(str(output / f"{slug}-depth.jpg"), cv2.cvtColor(depth, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output / f"{slug}-normals.jpg"), cv2.cvtColor(normals, cv2.COLOR_RGB2BGR))
        if persistent is not None:
            cv2.imwrite(str(output / f"{slug}-persistent.jpg"), cv2.cvtColor(persistent, cv2.COLOR_RGB2BGR))
        results[mode] = stats
    passed = all(item["status"] == "ok" and item["renderer_contract_valid"] for item in results.values())
    payload = {"webcam": args.device, "capture_shape": list(bgr.shape), "passed": passed, "modes": results}
    (output / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
