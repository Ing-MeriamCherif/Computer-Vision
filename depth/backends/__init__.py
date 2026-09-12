from depth.backends.depth_anything_v2_small import DepthAnythingV2Small

BACKENDS = {
    "depth_anything_v2_small": DepthAnythingV2Small,
}


def get_backend(name: str):
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(BACKENDS.keys())}")
    return BACKENDS[name]
