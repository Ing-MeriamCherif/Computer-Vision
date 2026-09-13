"""Opt-in adapter for the colleague ``feature/depth`` implementation.

The upstream package is intentionally preserved under ``integrations/``. This
adapter loads its original ``DepthModel`` entry point and maps its
``DepthState`` into our shared state contract, including source frame IDs.
Set ``NRW_DEPTH_SOURCE=colleague`` to use it; the tested local-checkpoint
provider remains the default for offline and low-latency runs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from .depth_provider import DepthInferenceDiagnostics
from .state import DepthState


class ColleagueDepthProvider:
    backend_name = "colleague-depth-anything-v2-small"

    def __init__(self, *, device: str = "auto", input_size: int = 518, fp16: bool = True, module_root: str | Path | None = None) -> None:
        root = Path(module_root) if module_root is not None else Path(__file__).resolve().parents[1] / "integrations" / "colleague_depth"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        # Mariem's module keeps the historical top-level ``depth`` package
        # name. Remove a previously loaded variant so A/B runs in one process
        # resolve the requested source deterministically.
        for module_name in tuple(sys.modules):
            if module_name == "depth" or module_name.startswith("depth."):
                del sys.modules[module_name]
        from depth.model import DepthModel  # the preserved upstream module

        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self._model = DepthModel(device=device, input_size=input_size, fp16=fp16)
        self.last_diagnostics: DepthInferenceDiagnostics | None = None
        self._session_scale: float | None = None

    def load(self) -> None:
        """Satisfy the common provider lifecycle contract.

        The preserved colleague model constructs its backend eagerly, so load
        is intentionally an idempotent no-op rather than a second warmup.
        """
        return None

    @property
    def device(self):
        """Resolved torch device for live HUD/diagnostics."""
        return getattr(getattr(self._model, "_backend", None), "device", None)

    def warmup(self, iterations: int = 1) -> None:
        self._model.warmup(iterations=max(0, int(iterations)))

    def compute(self, rgb_frame: np.ndarray, source_frame_id: int | str, timestamp: float) -> DepthState:
        import cv2

        frame = np.asarray(rgb_frame, dtype=np.uint8)[..., :3]
        t0 = time.perf_counter()
        upstream = self._model.infer(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        elapsed = (time.perf_counter() - t0) * 1000.0

        depth_arr = np.asarray(upstream.depth_map, dtype=np.float32)
        valid = upstream.valid_mask if upstream.valid_mask is not None else np.isfinite(depth_arr)
        finite = valid & np.isfinite(depth_arr)
        target_scale = 2.0 / max(float(np.median(depth_arr[finite])), 1e-6)
        self._session_scale = target_scale if self._session_scale is None else 0.10 * target_scale + 0.90 * self._session_scale
        depth_arr = (depth_arr * self._session_scale).astype(np.float32)
        min_d = float(np.min(depth_arr[valid])) if valid.any() else float(np.min(depth_arr))
        max_d = float(np.max(depth_arr[valid])) if valid.any() else float(np.max(depth_arr))
        peak_vram = None
        try:
            import torch
            if torch.cuda.is_available():
                peak_vram = float(torch.cuda.max_memory_allocated() / 1048576.0)
        except Exception:
            pass

        self.last_diagnostics = DepthInferenceDiagnostics(
            str(getattr(self._model, "device", "unknown")),
            elapsed,
            min_d,
            max_d,
            peak_vram,
            float(self._session_scale),
        )

        # The upstream branch exposes an all-valid mask, not neural
        # confidence. Derive conservative geometry/depth reliability from
        # local gradients rather than labeling every network pixel as 1.0.
        import cv2
        gx = cv2.Sobel(depth_arr, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth_arr, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.hypot(gx, gy)
        ref = max(float(np.percentile(gradient[finite], 90)), 1e-6) if finite.any() else 1.0
        reliability = np.exp(-np.clip(gradient / ref, 0.0, 4.0)).astype(np.float32)

        return DepthState(
            depth_arr,
            time.monotonic() if timestamp is None else float(timestamp),
            source_frame_id,
            upstream.scale_mode,
            valid_mask=valid,
            confidence=np.where(finite, reliability, 0.0).astype(np.float32),
        )


class MariemDepthProvider(ColleagueDepthProvider):
    """Adapter for the latest depth module on Mariem's ``main`` branch.

    The module is vendored under ``depth_module/`` with its original model and
    backend entry points preserved.  It is selectable independently so the
    older colleague integration remains available for A/B comparison.
    """

    backend_name = "mariem-main-depth-anything-v2-small"

    def __init__(self, *, device: str = "auto", input_size: int = 420, fp16: bool = True) -> None:
        root = Path(__file__).resolve().parents[1] / "depth_module"
        super().__init__(device=device, input_size=input_size, fp16=fp16, module_root=root)
