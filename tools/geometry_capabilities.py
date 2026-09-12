#!/usr/bin/env python3
"""Report optional CPU/GPU capabilities without installing dependencies."""

from __future__ import annotations

import importlib.util
import json

import numpy as np


def main() -> int:
    result = {"python": __import__("sys").version.split()[0], "numpy": np.__version__}
    try:
        import cv2
        result["opencv"] = cv2.__version__
        result["dis_available"] = bool(hasattr(cv2, "DISOpticalFlow_create"))
        result["farneback_available"] = hasattr(cv2, "calcOpticalFlowFarneback")
        result["opencv_cuda_devices"] = int(cv2.cuda.getCudaEnabledDeviceCount()) if hasattr(cv2, "cuda") else 0
    except ImportError:
        result.update(opencv=None, dis_available=False, farneback_available=False, opencv_cuda_devices=0)
    result["cupy_installed"] = importlib.util.find_spec("cupy") is not None
    result["pytorch_cuda"] = False
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch
            result["pytorch_cuda"] = bool(torch.cuda.is_available())
        except Exception:
            result["pytorch_cuda"] = False
    result["nvidia_optical_flow"] = False
    result["cpu_fallback"] = True
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
