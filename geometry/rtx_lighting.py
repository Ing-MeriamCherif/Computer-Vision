"""Optional NVIDIA OptiX backend boundary for Mode 7.

The native module is intentionally optional: importing this module must remain
safe on non-RTX machines.  The renderer reports a precise reason when OptiX is
not installed or the current GPU is not an RTX device.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import re
from typing import Any, Protocol


class RelightBackend(Protocol):
    backend_name: str

    def render(self, rgb: Any, geometry: Any, lights: Any, *, ambient: float = 0.4) -> tuple[Any, dict[str, Any]]: ...

    def close(self) -> None: ...

    def reset_history(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RTXCapability:
    available: bool
    reason: str
    gpu_name: str | None = None


def detect_rtx_optix() -> RTXCapability:
    """Detect an actual RTX-class NVIDIA device and the optional native module."""
    try:
        import torch

        if not torch.cuda.is_available():
            return RTXCapability(False, "CUDA is unavailable")
        gpu_name = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001
        return RTXCapability(False, f"CUDA query failed: {type(exc).__name__}: {exc}")

    # CUDA SM 7.5 is shared by GTX and RTX; the model name is therefore needed
    # to avoid falsely advertising RT cores on a GTX 16-series card.
    if not re.search(r"\bRTX\b", gpu_name, re.IGNORECASE):
        return RTXCapability(False, f"GPU has no RTX hardware: {gpu_name}", gpu_name)
    module_name = os.environ.get("P123_OPTIX_MODULE", "_p123_optix")
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        return RTXCapability(False, f"OptiX native module unavailable ({module_name}): {exc}", gpu_name)
    return RTXCapability(True, "OptiX native module available", gpu_name)


def create_optix_renderer(quality: str = "balanced") -> RelightBackend:
    """Construct the optional native renderer or raise a diagnostic error."""
    capability = detect_rtx_optix()
    if not capability.available:
        raise RuntimeError(capability.reason)
    module_name = os.environ.get("P123_OPTIX_MODULE", "_p123_optix")
    module = importlib.import_module(module_name)
    factory = getattr(module, "create_renderer", None)
    if not callable(factory):
        raise RuntimeError(f"{module_name} does not export create_renderer")
    renderer = factory(str(quality))
    renderer.backend_name = "RTX_OPTIX"
    return renderer


__all__ = ["RelightBackend", "RTXCapability", "detect_rtx_optix", "create_optix_renderer"]
