from depth.backends.depth_anything_v2_small import (
    DepthAnythingV2Small,
    DepthAnythingV2Base,
    DepthAnythingV2Large,
)

BACKENDS = {
    "depth_anything_v2_small": DepthAnythingV2Small,
    "depth_anything_v2_base": DepthAnythingV2Base,
    "depth_anything_v2_large": DepthAnythingV2Large,
}


def get_backend(name: str):
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(BACKENDS.keys())}")
    return BACKENDS[name]
