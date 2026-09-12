"""Camera-space geometry primitives for the NRW AI & Vision challenge.

The geometry package uses one coordinate convention everywhere:
``+X`` is image-right, ``+Y`` is image-down, and ``+Z`` is forward from the
camera into the scene. Pixel coordinates use the same top-left origin.
"""

from .backproject import DepthScaleMode, backproject_depth, depth_valid_mask
from .camera import CameraModel
from .state import DepthState, GeometryState
from .transforms import ImageTransform
from .normals import (
    AngularMetrics,
    NormalConfig,
    NormalMode,
    NormalResult,
    angular_metrics,
    estimate_normals,
    geometry_from_depth_state,
    normals_to_rgb,
)
from .motion import MotionState, OpticalFlowProvider, OpenCVFlowProvider, flow_consistency
from .temporal import TemporalConfig, TemporalGeometryEngine, photometric_error
from .warp import warp_depth_backward, warp_field_backward, warp_normals_backward
from .alignment import DepthAlignmentResult, align_history_depth, align_inverse_depth

__all__ = [
    "CameraModel",
    "DepthScaleMode",
    "DepthState",
    "GeometryState",
    "ImageTransform",
    "backproject_depth",
    "depth_valid_mask",
    "AngularMetrics",
    "NormalConfig",
    "NormalMode",
    "NormalResult",
    "angular_metrics",
    "estimate_normals",
    "geometry_from_depth_state",
    "normals_to_rgb",
    "MotionState",
    "OpticalFlowProvider",
    "OpenCVFlowProvider",
    "flow_consistency",
    "TemporalConfig",
    "TemporalGeometryEngine",
    "photometric_error",
    "warp_depth_backward",
    "warp_field_backward",
    "warp_normals_backward",
    "DepthAlignmentResult",
    "align_history_depth",
    "align_inverse_depth",
]
