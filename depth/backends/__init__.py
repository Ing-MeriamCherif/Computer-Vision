from depth.backends.depth_anything_v2_small import (
    DepthAnythingV2Small,
    DepthAnythingV2Base,
    DepthAnythingV2Large,
)
from depth.backends.yolo_depth import (
    YOLO26nDepth,
    YOLO26sDepth,
    YOLO26mDepth,
    YOLO26lDepth,
    YOLO26xDepth,
)

BACKENDS = {
    "depth_anything_v2_small": DepthAnythingV2Small,
    "depth_anything_v2_base": DepthAnythingV2Base,
    "depth_anything_v2_large": DepthAnythingV2Large,
    "yolo26n_depth": YOLO26nDepth,
    "yolo26s_depth": YOLO26sDepth,
    "yolo26m_depth": YOLO26mDepth,
    "yolo26l_depth": YOLO26lDepth,
    "yolo26x_depth": YOLO26xDepth,
}


def get_backend(name: str):
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(BACKENDS.keys())}")
    return BACKENDS[name]
