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
from .validation import GeometryValidationReport, validate_renderer_geometry
from .diagnostics import GeometryDiagnostics, StageTimer, geometry_state_nbytes, motion_state_nbytes, summarize_timings
from .pose import (
    CameraPoseState,
    PoseConfig,
    PoseCorrespondences,
    PoseEstimateResult,
    PoseEstimator,
    StaticGeometryResult,
    classify_static_geometry,
    compose_world_pose,
    compute_rigid_flow_residual,
    compute_static_confidence,
    extract_pose_correspondences,
    pose_error,
)
from .persistent import (
    AdvancedGeometryEngine,
    PersistentGeometryConfig,
    PersistentGeometryMapper,
    PersistentGeometryState,
    Surfel,
    SurfelMap,
    compute_dynamic_contamination,
    compute_reprojection_metrics,
    persistent_hole_fill,
)
from .cuda_backend import CudaGeometryDiagnostics, TorchGeometryBackend, torch_cuda_status
from .depth_provider import DepthAnythingProvider, DepthInferenceDiagnostics

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
    "GeometryValidationReport",
    "validate_renderer_geometry",
    "GeometryDiagnostics",
    "StageTimer",
    "geometry_state_nbytes",
    "motion_state_nbytes",
    "summarize_timings",
    "CameraPoseState",
    "PoseConfig",
    "PoseCorrespondences",
    "PoseEstimateResult",
    "PoseEstimator",
    "StaticGeometryResult",
    "classify_static_geometry",
    "compose_world_pose",
    "compute_rigid_flow_residual",
    "compute_static_confidence",
    "extract_pose_correspondences",
    "pose_error",
    "PersistentGeometryState",
    "PersistentGeometryConfig",
    "Surfel",
    "SurfelMap",
    "PersistentGeometryMapper",
    "AdvancedGeometryEngine",
    "persistent_hole_fill",
    "compute_dynamic_contamination",
    "compute_reprojection_metrics",
    "CudaGeometryDiagnostics",
    "TorchGeometryBackend",
    "torch_cuda_status",
    "DepthAnythingProvider",
    "DepthInferenceDiagnostics",
]
