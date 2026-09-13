"""Build the exact 336x448 FP32 Depth Anything TensorRT engine."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/depth-anything-v2-small")
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument("--width", type=int, default=448)
    args = parser.parse_args()

    import onnx
    import tensorrt as trt
    import torch
    from transformers import AutoModelForDepthEstimation

    class Wrapper(torch.nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            return self.model(pixel_values=pixel_values).predicted_depth

    root = Path(args.model)
    onnx_path = root / f"depth_{args.height}x{args.width}.onnx"
    engine_path = root / f"depth_{args.height}x{args.width}_fp32.engine"
    model = Wrapper(AutoModelForDepthEstimation.from_pretrained(root, local_files_only=True).eval()).cuda()
    sample = torch.zeros((1, 3, args.height, args.width), dtype=torch.float32, device="cuda")
    with torch.inference_mode():
        torch.onnx.export(
            model, (sample,), str(onnx_path), input_names=["pixel_values"],
            output_names=["predicted_depth"], opset_version=18,
            dynamo=False, do_constant_folding=True,
        )
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    trt_parser = trt.OnnxParser(network, logger)
    if not trt_parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(trt_parser.get_error(i)) for i in range(trt_parser.num_errors))
        raise RuntimeError(errors)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.write_bytes(bytes(serialized))
    print(f"Built exact FP32 engine: {engine_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
