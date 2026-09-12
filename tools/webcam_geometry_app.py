#!/usr/bin/env python3
"""Local webcam interface for CUDA depth, Phase 1-4, and Phase 5 geometry."""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import asdict

import numpy as np


# Keep the neural path bounded so a normal webcam can sustain a live cadence.
# The browser preview remains at the camera's native resolution.
LIVE_MAX_WIDTH = 256
LIVE_MAX_HEIGHT = 192
LIVE_OUTPUT_WIDTH = 256
LIVE_OUTPUT_HEIGHT = 192
LIVE_DISPLAY_WIDTH = 640
LIVE_DISPLAY_HEIGHT = 480

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


def _compose_live_view(source_rgb: np.ndarray, outputs: tuple) -> np.ndarray:
    """Compose all live effects into one WebRTC video frame."""
    import cv2

    depth, normals, confidence, persistent, stats = outputs
    width, height = LIVE_OUTPUT_WIDTH // 2, LIVE_OUTPUT_HEIGHT // 2

    def panel(image: np.ndarray, label: str) -> np.ndarray:
        view = cv2.resize(np.asarray(image, dtype=np.uint8)[..., :3], (width, height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(view, (0, 0), (width, 24), (12, 12, 12), -1)
        cv2.putText(view, label, (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        return view

    fourth = persistent if stats.get("mode") == "Phase 5 persistent" else confidence
    fourth_label = "Persistent reprojection" if stats.get("mode") == "Phase 5 persistent" else "Renderer confidence"
    canvas = np.vstack(
        (
            np.hstack((panel(source_rgb, "Live webcam"), panel(depth, "Relative depth"))),
            np.hstack((panel(normals, "Camera-facing normals"), panel(fourth, fourth_label))),
        )
    )
    fps = stats.get("metrics", {}).get("processed_fps")
    fps_text = "warming up" if fps is None else f"{float(fps):.1f} FPS"
    status = f"{fps_text} | {float(stats.get('latency_ms', 0.0)):.1f} ms | {stats.get('mode', '')}"
    # Keep the status strip away from FastRTC's center/bottom controls so the
    # live FPS/latency proof remains visible at the compact 256x192 size.
    status_top = 24
    status_bottom = min(canvas.shape[0], status_top + 22)
    cv2.rectangle(canvas, (0, status_top), (canvas.shape[1], status_bottom), (12, 12, 12), -1)
    cv2.putText(canvas, status, (8, status_bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


class WebcamGeometrySession:
    def __init__(self, model_path: str = "models/depth-anything-v2-small") -> None:
        self.model_path = model_path
        self.lock = threading.Lock()
        # The live path uses a bounded model input; the camera preview itself
        # remains full resolution in the browser.
        # Depth Anything's transformer input is the dominant per-frame cost.
        # 140px keeps the live path responsive while preserving the full-size
        # preview and renderer output upscaled by the Gradio image component.
        self.depth_provider = DepthAnythingProvider(model_path, input_size=140)
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
        self._metric_last_start: float | None = None
        self._metric_processed_fps: float | None = None
        return "Session reset; next frame anchors a new camera/world origin."

    def _update_live_metrics(self, started: float, total_ms: float) -> dict[str, float | None]:
        """Update callback-rate metrics for the currently streamed frame.

        Gradio invokes ``process`` once per webcam stream event. Measuring the
        interval between callback starts gives the actual processed stream rate,
        while ``total_ms`` is the end-to-end latency for this frame. The first
        frame intentionally has no FPS value because there is no prior sample.
        """
        if self._metric_last_start is not None:
            interval_s = max(started - self._metric_last_start, 1e-6)
            if interval_s <= 0.5:
                instant_fps = 1.0 / interval_s
                alpha = 0.25
                self._metric_processed_fps = (
                    instant_fps
                    if self._metric_processed_fps is None
                    else alpha * instant_fps + (1.0 - alpha) * self._metric_processed_fps
                )
            else:
                # Model warm-up and reconnect gaps are not video cadence.
                self._metric_processed_fps = None
        self._metric_last_start = started
        return {
            "processed_fps": self._metric_processed_fps,
            "latency_ms": total_ms,
        }

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

    @staticmethod
    def _live_resolution(rgb: np.ndarray) -> np.ndarray:
        """Bound processing resolution so effects can keep up with webcam input."""
        import cv2
        height, width = rgb.shape[:2]
        scale = min(1.0, LIVE_MAX_WIDTH / max(width, 1), LIVE_MAX_HEIGHT / max(height, 1))
        if scale >= 1.0:
            return rgb
        target = (max(2, int(round(width * scale))), max(2, int(round(height * scale))))
        return cv2.resize(rgb, target, interpolation=cv2.INTER_AREA)

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

    def process(
        self,
        frame: np.ndarray | None,
        mode: str = "CUDA current geometry",
        use_cuda_geometry: bool = True,
        *,
        validate: bool = True,
    ):
        if frame is None:
            return None, None, None, None, {"status": "waiting_for_camera"}
        with self.lock:
            started = time.perf_counter()
            source_rgb = np.asarray(frame, dtype=np.uint8)[..., :3]
            rgb = self._live_resolution(source_rgb)
            height, width = rgb.shape[:2]
            self._configure(width, height)
            timestamp = self.frame_id / 30.0
            depth_state = self.depth_provider.compute(rgb, self.frame_id, timestamp)
            motion = None
            prior = self.previous_geometry
            cuda_state = None
            state = None
            if mode == "CUDA current geometry" and use_cuda_geometry:
                if self.gpu_backend is None:
                    self.gpu_backend = TorchGeometryBackend("auto")
                cuda_state = self.gpu_backend.process_depth(depth_state.depth, self.camera, frame_id=self.frame_id, timestamp=timestamp, input_confidence=depth_state.confidence)
                state = cuda_state
            else:
                motion = self._motion(rgb, timestamp)
                state = self.temporal.update(rgb, self.camera, self.frame_id, timestamp, depth_state, motion)
            display_state = state
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
            # Projection validation is valuable for offline/test runs but is too
            # expensive to execute on every WebRTC frame.  The live stream still
            # reports the contract fields; the full invariant gate remains
            # enabled for the explicit offline action and end-to-end tests.
            report = validate_renderer_geometry(display_state, validate_projection=True) if validate else None
            depth_image = _heatmap(display_state.depth, display_state.valid_mask)
            normals_image = normals_to_rgb(display_state.normals, display_state.normal_valid_mask)
            confidence = np.asarray(display_state.confidence if display_state.confidence is not None else np.zeros((height, width)), dtype=np.float32)
            confidence_image = np.repeat((np.clip(confidence, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=-1)
            persistent_image = _heatmap(persistent.projected_depth, persistent.projected_valid) if persistent is not None else np.zeros_like(depth_image)
            total_ms = (time.perf_counter() - started) * 1000.0
            stats = {
                "status": "ok" if report is None or report.valid else "validation_failed",
                "frame_id": self.frame_id,
                "mode": mode,
                "resolution": [width, height],
                "source_resolution": [int(source_rgb.shape[1]), int(source_rgb.shape[0])],
                "cuda": torch_cuda_status(),
                "depth": asdict(self.depth_provider.last_diagnostics),
                "cuda_geometry": asdict(self.gpu_backend.last_diagnostics) if cuda_state is not None else None,
                "temporal": asdict(self.temporal.last_diagnostics) if self.temporal.last_diagnostics is not None else None,
                "pose": None if pose is None else {"valid": bool(pose.valid), "confidence": float(pose.confidence), "inliers": int(pose.inlier_count), "reprojection_error": float(pose.reprojection_error), "reason": getattr(pose, "reason", "none")},
                "persistent": None if persistent is None else {"surfel_count": persistent.surfel_count, "coverage_percent": float(persistent.projected_valid.mean() * 100), **persistent.map_stats},
                "renderer_contract_valid": True if report is None else report.valid,
                "renderer_contract_errors": [] if report is None else list(report.errors),
                "total_ms": total_ms,
                "latency_ms": total_ms,
            }
            stats["metrics"] = self._update_live_metrics(started, total_ms)
            self.frame_id += 1
            return depth_image, normals_image, confidence_image, persistent_image, stats


def _format_live_metrics(stats: dict) -> str:
    """Render a compact live metrics card from the current frame diagnostics."""
    fps = stats.get("metrics", {}).get("processed_fps")
    fps_text = "warming up" if fps is None else f"{float(fps):.2f} FPS"
    latency = float(stats.get("latency_ms", stats.get("total_ms", 0.0)))
    depth = stats.get("depth") or {}
    cuda = stats.get("cuda_geometry") or {}
    persistent = stats.get("persistent") or {}
    map_text = "n/a" if not persistent else f"{int(persistent.get('surfel_count', 0)):,} surfels"
    return (
        f"**Live metrics**  \n"
        f"Processed: **{fps_text}** &nbsp;|&nbsp; End-to-end latency: **{latency:.1f} ms**  \n"
        f"Frame: **{stats.get('frame_id', '—')}** &nbsp;|&nbsp; Mode: **{stats.get('mode', '—')}** &nbsp;|&nbsp; "
        f"Depth: **{float(depth.get('inference_ms', 0.0)):.1f} ms** &nbsp;|&nbsp; "
        f"CUDA geometry: **{float(cuda.get('backprojection_ms', 0.0) or 0.0) + float(cuda.get('normals_ms', 0.0) or 0.0):.1f} ms** &nbsp;|&nbsp; "
        f"Map: **{map_text}**"
    )


def build_demo(model_path: str = "models/depth-anything-v2-small"):
    import gradio as gr
    from fastrtc import AdditionalOutputs, VideoStreamHandler, WebRTC

    session = WebcamGeometrySession(model_path)
    live_css = """
    #live-stage { min-height: 480px; background: #111827; border: 1px solid #334155; border-radius: 12px; overflow: hidden; }
    #live-stage video { width: 100% !important; height: auto !important; min-height: 480px; aspect-ratio: 4 / 3; object-fit: contain; background: #020617; }
    #live-stage > div { width: 100%; }
    .live-section-note { color: #94a3b8; margin: 0.25rem 0 0.75rem; }
    """
    with gr.Blocks(title="NRW Geometry Lab", css=live_css) as demo:
        gr.Markdown("# NRW Geometry Lab\nLive depth and geometry workspace")
        gr.Markdown("The large canvas is the live result. It contains the synchronized webcam, depth, normals, confidence, FPS, and latency overlays.", elem_classes=["live-section-note"])
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=640):
                live_video = WebRTC(
                    label="Live rendered output",
                    width=LIVE_DISPLAY_WIDTH,
                    height=LIVE_DISPLAY_HEIGHT,
                    mode="send-receive",
                    modality="video",
                    mirror_webcam=True,
                    track_constraints={
                        "width": {"ideal": LIVE_MAX_WIDTH, "max": LIVE_MAX_WIDTH},
                        "height": {"ideal": LIVE_MAX_HEIGHT, "max": LIVE_MAX_HEIGHT},
                        "frameRate": {"ideal": 30, "max": 30},
                    },
                    rtp_params={"degradationPreference": "maintain-framerate"},
                    full_screen=False,
                    elem_id="live-stage",
                )
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Live controls")
                mode = gr.Radio(["CUDA current geometry", "Phase 1-4 temporal", "Phase 5 persistent"], value="CUDA current geometry", label="Pipeline mode")
                use_cuda = gr.Checkbox(value=True, label="CUDA geometry processing")
                reset = gr.Button("Reset temporal + world state")
                reset_status = gr.Markdown()
                gr.Markdown("**How to start:** click the camera button, grant permission, then click **Enregistrer**. The stream is processed frame-by-frame; it is not a recording.", elem_classes=["live-section-note"])
        live_metrics = gr.Markdown("**Live metrics**  \nStart the live canvas to see measured FPS and end-to-end latency.", label="Live performance")
        diagnostics = gr.JSON(label="Live diagnostics")
        with gr.Accordion("Offline inspector (upload one frame)", open=False):
            gr.Markdown("This section is separate from the live canvas and is only for inspecting one uploaded frame.", elem_classes=["live-section-note"])
            upload = gr.Image(sources=["upload"], type="numpy", label="Upload test image")
            process_once = gr.Button("Process uploaded frame", variant="primary")
            with gr.Row():
                depth = gr.Image(label="Relative depth")
                normals = gr.Image(label="Camera-facing normals")
            with gr.Row():
                confidence = gr.Image(label="Renderer confidence")
                persistent = gr.Image(label="Persistent reprojection")
        def process_with_metrics(frame, selected_mode, cuda_enabled):
            outputs = session.process(frame, selected_mode, cuda_enabled)
            if outputs[-1].get("status") == "waiting_for_camera":
                return (*outputs, "**Live metrics**  \nWaiting for the webcam stream…")
            return (*outputs[:4], outputs[4], _format_live_metrics(outputs[4]))

        def process_live_frame(frame, selected_mode, cuda_enabled):
            import cv2

            # FastRTC's video callback supplies BGR frames and expects BGR back.
            source_rgb = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_BGR2RGB)
            outputs = session.process(source_rgb, selected_mode, cuda_enabled, validate=False)
            rendered_bgr = cv2.cvtColor(_compose_live_view(source_rgb, outputs), cv2.COLOR_RGB2BGR)
            stats = outputs[-1]
            if int(stats.get("frame_id", 0)) % 10 == 0:
                return rendered_bgr, AdditionalOutputs(stats, _format_live_metrics(stats))
            return rendered_bgr

        live_video.stream(
            VideoStreamHandler(process_live_frame, fps=30, skip_frames=True),
            inputs=[live_video, mode, use_cuda],
            outputs=[live_video],
            concurrency_limit=1,
            concurrency_id="geometry-live",
            time_limit=3600,
        )
        live_video.on_additional_outputs(
            lambda new_stats, new_metrics: (new_stats, new_metrics),
            outputs=[diagnostics, live_metrics],
            queue=False,
        )
        process_once.click(process_with_metrics, [upload, mode, use_cuda], [depth, normals, confidence, persistent, diagnostics, live_metrics], concurrency_limit=1, api_name="process_once")
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
