#!/usr/bin/env python3
"""Local webcam interface for CUDA depth, Phase 1-4, and Phase 5 geometry."""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import asdict

import numpy as np

from geometry import (
    CameraModel, CameraPoseState, DepthAnythingProvider, OpenCVFlowProvider,
    PersistentGeometryMapper, PoseEstimator, SurfelMap, TemporalConfig,
    TemporalGeometryEngine, TorchGeometryBackend, normals_to_rgb,
    torch_cuda_status, validate_renderer_geometry,
)


def _heatmap(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    import cv2
    array = np.asarray(values, dtype=np.float32)
    mask = np.isfinite(array) if valid is None else np.asarray(valid, dtype=bool) & np.isfinite(array)
    image = np.zeros(array.shape, np.uint8)
    if mask.any():
        low, high = np.percentile(array[mask], (2, 98))
        image[mask] = np.clip((array[mask] - low) / max(float(high - low), 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(image, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


class WebcamGeometrySession:
    def __init__(self, model_path: str = "models/depth-anything-v2-small") -> None:
        self.model_path = model_path
        self.lock = threading.Lock()
        self.depth_provider = DepthAnythingProvider(model_path)
        self.gpu_backend: TorchGeometryBackend | None = None
        self.reset()

    def reset(self) -> str:
        self.frame_id = 0
        self.camera = None
        self.temporal = None
        self.mapper = None
        self.pose_estimator = None
        self.flow = OpenCVFlowProvider(method="farneback", flow_scale=0.5)
        self.previous_rgb = None
        self.previous_geometry = None
        return "Session reset; next frame anchors a new camera/world origin."

    def _configure(self, width: int, height: int) -> None:
        if self.camera is not None and (self.camera.width, self.camera.height) == (width, height):
            return
        self.camera = CameraModel(width, height, width * 0.9, height * 0.9, (width - 1) / 2, (height - 1) / 2)
        self.temporal = TemporalGeometryEngine(self.camera, TemporalConfig(diagnostics_level="timing", alignment_max_samples=5000))
        scene_voxel = 2.0 * 0.02
        self.mapper = PersistentGeometryMapper(self.camera, SurfelMap(voxel_size=scene_voxel, max_surfels=50_000), mapping_stride=max(2, width // 160))
        self.pose_estimator = PoseEstimator(self.camera, max_samples=2000, min_inliers=20)
        self.previous_rgb = None
        self.previous_geometry = None
        self.frame_id = 0

    def _motion(self, rgb: np.ndarray, timestamp: float):
        if self.previous_rgb is None:
            return None
        return self.flow.compute(self.previous_rgb, rgb, self.frame_id - 1, self.frame_id, timestamp)

    def _pose(self, motion, geometry, timestamp: float):
        if self.frame_id == 0:
            return CameraPoseState(timestamp, 0, np.eye(4, dtype=np.float32), True, 1.0)
        if motion is None or self.previous_geometry is None:
            return None
        previous = self.previous_geometry
        valid = previous.valid_mask & motion.valid_mask & (motion.flow_confidence > 0.25)
        if previous.confidence is not None:
            valid &= previous.confidence > 0.2
        yy, xx = np.nonzero(valid)
        if len(xx) < self.pose_estimator.min_correspondences:
            return None
        object_points = previous.positions_3d[yy, xx]
        image_points = np.stack((xx, yy), axis=-1).astype(np.float32) + motion.forward_flow[yy, xx]
        return self.pose_estimator.estimate(object_points, image_points)

    def process(self, frame: np.ndarray | None, mode: str = "Phase 1-4 temporal", use_cuda_geometry: bool = True):
        if frame is None:
            return None, None, None, None, {"status": "waiting_for_camera"}
        with self.lock:
            started = time.perf_counter()
            rgb = np.asarray(frame, dtype=np.uint8)[..., :3]
            height, width = rgb.shape[:2]
            self._configure(width, height)
            timestamp = self.frame_id / 30.0
            depth_state = self.depth_provider.compute(rgb, self.frame_id, timestamp)
            motion = self._motion(rgb, timestamp)
            prior = self.previous_geometry
            state = self.temporal.update(rgb, self.camera, self.frame_id, timestamp, depth_state, motion)
            cuda_state = None
            if use_cuda_geometry:
                if self.gpu_backend is None:
                    self.gpu_backend = TorchGeometryBackend("auto")
                cuda_state = self.gpu_backend.process_depth(depth_state.depth, self.camera, frame_id=self.frame_id, timestamp=timestamp, input_confidence=depth_state.confidence)
            display_state = cuda_state if mode == "CUDA current geometry" and cuda_state is not None else state
            persistent = None
            pose = None
            if mode == "Phase 5 persistent":
                self.previous_geometry = prior
                pose = self._pose(motion, state, timestamp)
                self.previous_geometry = prior
                if pose is not None:
                    persistent = self.mapper.update(state, pose)
            self.previous_rgb = rgb.copy()
            self.previous_geometry = state
            report = validate_renderer_geometry(display_state, validate_projection=True)
            depth_image = _heatmap(display_state.depth, display_state.valid_mask)
            normals_image = normals_to_rgb(display_state.normals, display_state.normal_valid_mask)
            confidence = np.asarray(display_state.confidence if display_state.confidence is not None else np.zeros((height, width)), dtype=np.float32)
            confidence_image = np.repeat((np.clip(confidence, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=-1)
            persistent_image = _heatmap(persistent.projected_depth, persistent.projected_valid) if persistent is not None else np.zeros_like(depth_image)
            stats = {
                "status": "ok" if report.valid else "validation_failed",
                "frame_id": self.frame_id,
                "mode": mode,
                "resolution": [width, height],
                "cuda": torch_cuda_status(),
                "depth": asdict(self.depth_provider.last_diagnostics),
                "cuda_geometry": asdict(self.gpu_backend.last_diagnostics) if cuda_state is not None else None,
                "temporal": asdict(self.temporal.last_diagnostics) if self.temporal.last_diagnostics is not None else None,
                "pose": None if pose is None else {"valid": bool(pose.valid), "confidence": float(pose.confidence), "inliers": int(pose.inlier_count), "reprojection_error": float(pose.reprojection_error), "reason": getattr(pose, "reason", "none")},
                "persistent": None if persistent is None else {"surfel_count": persistent.surfel_count, "coverage_percent": float(persistent.projected_valid.mean() * 100), **persistent.map_stats},
                "renderer_contract_valid": report.valid,
                "renderer_contract_errors": list(report.errors),
                "total_ms": (time.perf_counter() - started) * 1000.0,
            }
            self.frame_id += 1
            return depth_image, normals_image, confidence_image, persistent_image, stats


def build_demo(model_path: str = "models/depth-anything-v2-small"):
    import gradio as gr
    session = WebcamGeometrySession(model_path)
    with gr.Blocks(title="NRW Geometry Lab") as demo:
        gr.Markdown("# NRW Geometry Lab\nLive CUDA depth, stable Phase 1–4 geometry, and optional Phase 5 world-memory diagnostics.")
        with gr.Row():
            camera = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Live webcam")
            upload = gr.Image(sources=["upload"], type="numpy", label="Offline/test image")
            with gr.Column():
                mode = gr.Radio(["CUDA current geometry", "Phase 1-4 temporal", "Phase 5 persistent"], value="Phase 1-4 temporal", label="Pipeline mode")
                use_cuda = gr.Checkbox(value=True, label="CUDA geometry processing")
                process_once = gr.Button("Process current frame", variant="primary")
                reset = gr.Button("Reset temporal + world state")
                reset_status = gr.Markdown()
        with gr.Row():
            depth = gr.Image(label="Relative depth", streaming=True)
            normals = gr.Image(label="Camera-facing normals", streaming=True)
        with gr.Row():
            confidence = gr.Image(label="Renderer confidence", streaming=True)
            persistent = gr.Image(label="Persistent reprojection", streaming=True)
        diagnostics = gr.JSON(label="Runtime diagnostics")
        camera.stream(session.process, [camera, mode, use_cuda], [depth, normals, confidence, persistent, diagnostics], stream_every=0.35, concurrency_limit=1, api_name="process_frame")
        process_once.click(session.process, [upload, mode, use_cuda], [depth, normals, confidence, persistent, diagnostics], concurrency_limit=1, api_name="process_once")
        reset.click(session.reset, outputs=reset_status)
    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model", default="models/depth-anything-v2-small")
    args = parser.parse_args()
    build_demo(args.model).queue(default_concurrency_limit=1).launch(server_name=args.host, server_port=args.port, share=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
