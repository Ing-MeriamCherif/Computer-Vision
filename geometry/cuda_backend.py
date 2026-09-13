"""Optional PyTorch/CUDA geometry backend with an explicit CPU fallback.

This module is additive: importing the base ``geometry`` package never requires
PyTorch. Large tensor operations stay on the selected device until the final
renderer-compatible ``GeometryState`` snapshot is requested.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .backproject import DepthScaleMode
from .camera import CameraModel
from .state import GeometryState


@dataclass(frozen=True, slots=True)
class CudaGeometryDiagnostics:
    device: str
    cuda_available: bool
    backprojection_ms: float
    normals_ms: float
    transfer_ms: float
    peak_vram_mb: float | None


class TorchGeometryBackend:
    """Device-agnostic Torch backend; ``device='auto'`` prefers CUDA."""

    def __init__(self, device: str = "auto", *, discontinuity_threshold: float = 0.15) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for TorchGeometryBackend") from exc
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        self.device = torch.device(device)
        self.discontinuity_threshold = float(discontinuity_threshold)
        self.last_diagnostics: CudaGeometryDiagnostics | None = None
        self._grid_cache: dict[tuple[int, int, float, float, float, float, str], tuple] = {}
        self._call_count = 0

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def _sync(self) -> None:
        if self.is_cuda:
            self.torch.cuda.synchronize(self.device)

    def _rays(self, camera: CameraModel):
        key = (camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy, str(self.device))
        rays = self._grid_cache.get(key)
        if rays is None:
            torch = self.torch
            yy, xx = torch.meshgrid(
                torch.arange(camera.height, dtype=torch.float32, device=self.device),
                torch.arange(camera.width, dtype=torch.float32, device=self.device),
                indexing="ij",
            )
            rays = ((xx - camera.cx) / camera.fx, (yy - camera.cy) / camera.fy)
            if len(self._grid_cache) >= 8:
                self._grid_cache.pop(next(iter(self._grid_cache)))
            self._grid_cache[key] = rays
        return rays

    def process_depth(self, depth: np.ndarray, camera: CameraModel, *, frame_id: int | str = 0, timestamp: float = 0.0, scale_mode: DepthScaleMode | str = DepthScaleMode.RELATIVE, valid_mask: np.ndarray | None = None, input_confidence: np.ndarray | None = None) -> GeometryState:
        torch = self.torch
        z = torch.as_tensor(np.asarray(depth, dtype=np.float32), device=self.device)
        if z.shape != (camera.height, camera.width):
            raise ValueError("depth shape must match camera resolution")
        valid = torch.isfinite(z) & (z > 1e-6)
        if valid_mask is not None:
            mask = torch.as_tensor(np.asarray(valid_mask, dtype=bool), device=self.device)
            if mask.shape != z.shape:
                raise ValueError("valid_mask must match depth")
            valid &= mask
        start = time.perf_counter()
        ray_x, ray_y = self._rays(camera)
        points = torch.stack((ray_x * z, ray_y * z, z), dim=-1)
        points = torch.where(valid[..., None], points, torch.full_like(points, float("nan")))
        backprojection_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        def estimate(radius: int):
            left, right = torch.roll(points, radius, 1), torch.roll(points, -radius, 1)
            up, down = torch.roll(points, radius, 0), torch.roll(points, -radius, 0)
            left_v, right_v = torch.roll(valid, radius, 1), torch.roll(valid, -radius, 1)
            up_v, down_v = torch.roll(valid, radius, 0), torch.roll(valid, -radius, 0)
            left_v[:, :radius] = False; right_v[:, -radius:] = False; up_v[:radius, :] = False; down_v[-radius:, :] = False
            def compatible(other_z, other_valid):
                relative = torch.abs(z - other_z) / torch.clamp(torch.minimum(torch.abs(z), torch.abs(other_z)), min=1e-6)
                return other_valid & torch.isfinite(relative) & (relative <= self.discontinuity_threshold * (1.0 + 0.15 * radius))
            nvalid = valid & compatible(left[..., 2], left_v) & compatible(right[..., 2], right_v) & compatible(up[..., 2], up_v) & compatible(down[..., 2], down_v)
            n = torch.linalg.cross(down - up, right - left, dim=-1)
            lengths = torch.linalg.vector_norm(n, dim=-1)
            nvalid &= torch.isfinite(lengths) & (lengths > 1e-8)
            n = n / torch.where(nvalid, lengths, torch.ones_like(lengths))[..., None]
            return n, nvalid

        normals1, valid1 = estimate(1)
        normals2, valid2 = estimate(2)
        # Multi-scale edge-aware selection: use the larger footprint on smooth
        # regions to suppress monocular-depth speckle, retaining radius 1 at
        # genuine depth edges and radius 2 where the immediate footprint is
        # incomplete.
        neighbor = (torch.roll(z, 1, 0) + torch.roll(z, -1, 0) + torch.roll(z, 1, 1) + torch.roll(z, -1, 1)) * 0.25
        edge_strength = torch.abs(z - neighbor) / torch.clamp(torch.abs(z), min=1e-6)
        use2 = valid2 & ((~valid1) | (edge_strength < 0.08))
        normals = torch.where(use2[..., None], normals2, normals1)
        normal_valid = valid1 | valid2
        flip = torch.sum(normals * points, dim=-1) > 0
        normals = torch.where(flip[..., None], -normals, normals)
        normals = torch.where(normal_valid[..., None], normals, torch.full_like(normals, float("nan")))
        confidence = normal_valid.to(torch.float32)
        if input_confidence is not None:
            supplied = torch.as_tensor(np.asarray(input_confidence, dtype=np.float32), device=self.device)
            if supplied.shape != z.shape:
                raise ValueError("input_confidence must match depth")
            confidence *= torch.nan_to_num(supplied, nan=0.0).clamp(0, 1)
        normals_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        arrays = [z, points, valid, normals, confidence, normal_valid]
        depth_np, points_np, valid_np, normals_np, confidence_np, normal_valid_np = [item.detach().cpu().numpy() for item in arrays]
        transfer_ms = (time.perf_counter() - start) * 1000.0
        self._call_count += 1
        peak = float(torch.cuda.max_memory_allocated(self.device) / 1048576.0) if self.is_cuda and self._call_count % 30 == 0 else None
        self.last_diagnostics = CudaGeometryDiagnostics(str(self.device), torch.cuda.is_available(), backprojection_ms, normals_ms, transfer_ms, peak)
        confidence_np = np.where(valid_np, confidence_np, 0.0).astype(np.float32)
        return GeometryState(
            timestamp, frame_id, depth_np.astype(np.float32), points_np.astype(np.float32), valid_np,
            camera, scale_mode, normals=normals_np.astype(np.float32), confidence=confidence_np,
            normal_valid_mask=normal_valid_np, normal_confidence=confidence_np,
            selected_radius=np.where(valid1.detach().cpu().numpy(), 1, np.where(valid2.detach().cpu().numpy(), 2, 0)).astype(np.int16), spatial_confidence=confidence_np,
        )


def torch_cuda_status() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch_not_installed"}
    status: dict[str, object] = {"available": bool(torch.cuda.is_available()), "torch": torch.__version__, "cuda_runtime": torch.version.cuda}
    if torch.cuda.is_available():
        status.update({"device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)), "compiled_arches": torch.cuda.get_arch_list()})
    return status
