"""Native live competition application and decoupled pipeline runtime.

Integrates camera capture, monocular depth inference, multi-hand tracking,
dynamic relighting (diffuse + specular + dynamic shadows + volumetric scattering),
and optional persistent geometry into a native GLFW+OpenGL window running at
display refresh rates (30-60+ FPS).
"""

from __future__ import annotations

from enum import Enum, IntEnum
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .async_pipeline import DepthWorker, LatestDepthBuffer, LatestFrameBuffer
from .backproject import DepthScaleMode
from .camera import CameraModel
from .camera_worker import CameraCaptureWorker, LatestFrameSlot
from .cuda_backend import TorchGeometryBackend, torch_cuda_status
from .depth_provider import DepthAnythingProvider
from .colleague_depth import ColleagueDepthProvider
from .hand_control import GestureState, HandControlEngine, TrackedHand, create_hand_tracker
from .lighting import (
    LightState,
    project_light_orb,
    render_light_orbs,
    render_volumetric_scattering,
    sample_depth,
    shade_geometry,
)
from .motion import OpenCVFlowProvider
from .native_window import NativeOpenGLWindow
from .normals import normals_to_rgb
from .persistent import PersistentGeometryConfig, PersistentGeometryMapper, PersistentGeometryState, SurfelMap
from .persistent_worker import PersistentMapWorker
from .pose import CameraPoseState, PoseEstimator
from .state import DepthState, GeometryState


class AppMode(IntEnum):
    """Supported interactive challenge levels and visualization modes."""

    FEED = 1          # Clean full-res camera RGB + hand skeleton
    DEPTH = 2         # Monocular relative/metric depth colormap
    NORMALS = 3       # Camera-facing surface normals (RGB)
    DIFFUSE = 4       # Lambertian relighting (ambient + diffuse)
    SPECULAR = 5      # Blinn-Phong specular highlights + roughness
    SHADOWS = 6       # Dynamic screen-space raymarched self-shadows (L4)
    GESTURE = 7       # Single-hand 3D gesture light control (L3)
    MULTILIGHT = 8    # Two-hand dual colored lights + volumetric scattering (L5)
    INFINITY = 9      # Persistent surfel map and camera pose tracking (Phase 5)


class QualityProfile(str, Enum):
    LOW = "LOW"
    BALANCED = "BALANCED"
    HIGH = "HIGH"


@dataclass(frozen=True, slots=True)
class QualityProfileConfig:
    name: str
    depth_size: tuple[int, int]         # (height, width) for depth neural net
    shadow_steps: int                   # Screen-space shadow sample count
    volumetric_steps: int               # Ray-marched volumetric steps
    volumetric_downsample: int          # Spatial downsampling factor for haze
    persistent_hz: float                # Persistent mapping update rate limit


PROFILE_CONFIGS: dict[QualityProfile, QualityProfileConfig] = {
    QualityProfile.LOW: QualityProfileConfig(
        name="LOW",
        depth_size=(144, 192),
        shadow_steps=2,
        volumetric_steps=3,
        volumetric_downsample=4,
        persistent_hz=2.0,
    ),
    QualityProfile.BALANCED: QualityProfileConfig(
        name="BALANCED",
        depth_size=(192, 256),
        shadow_steps=4,
        volumetric_steps=5,
        volumetric_downsample=4,
        persistent_hz=4.0,
    ),
    QualityProfile.HIGH: QualityProfileConfig(
        name="HIGH",
        depth_size=(240, 320),
        shadow_steps=8,
        volumetric_steps=8,
        volumetric_downsample=4,
        persistent_hz=8.0,
    ),
}


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm knuckles
)


def _heatmap(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Colormap a depth or confidence array using the TURBO palette."""
    array = np.asarray(values, dtype=np.float32)
    mask = np.isfinite(array) if valid is None else np.asarray(valid, dtype=bool) & np.isfinite(array)
    image = np.zeros(array.shape, dtype=np.uint8)
    if mask.any():
        low, high = np.percentile(array[mask], (2, 98))
        norm = np.clip((array[mask] - low) / max(float(high - low), 1e-6) * 255.0, 0, 255)
        image[mask] = norm.astype(np.uint8)
    colored_bgr = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def _draw_transparent_box(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int] = (15, 15, 15),
    alpha: float = 0.72,
) -> None:
    """Draw a semi-transparent dark rectangle for crisp HUD readability."""
    h, w = img.shape[:2]
    xa, ya = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    xb, yb = max(0, min(x2, w)), max(0, min(y2, h))
    if xb <= xa or yb <= ya:
        return
    sub = img[ya:yb, xa:xb]
    rect = np.full(sub.shape, color, dtype=np.uint8)
    img[ya:yb, xa:xb] = cv2.addWeighted(sub, 1.0 - alpha, rect, alpha, 0.0)


def draw_hand_skeleton(
    img: np.ndarray,
    hand: TrackedHand,
    color_rgb: tuple[int, int, int] = (0, 255, 200),
    draw_vector: bool = True,
    label: str | None = None,
) -> None:
    """Draw hand landmarks, bones, palm center, and virtual light beam."""
    h, w = img.shape[:2]
    # Draw bones
    if hand.landmarks_uv is not None:
        pts = hand.landmarks_uv.astype(np.int32)
        for i, j in HAND_CONNECTIONS:
            if i < len(pts) and j < len(pts):
                p1 = (int(np.clip(pts[i][0], 0, w - 1)), int(np.clip(pts[i][1], 0, h - 1)))
                p2 = (int(np.clip(pts[j][0], 0, w - 1)), int(np.clip(pts[j][1], 0, h - 1)))
                cv2.line(img, p1, p2, color_rgb, 2, cv2.LINE_AA)
        for pt in pts:
            u, v = int(np.clip(pt[0], 0, w - 1)), int(np.clip(pt[1], 0, h - 1))
            cv2.circle(img, (u, v), 3, (255, 255, 255), -1, cv2.LINE_AA)

    # Draw palm center
    pu, pv = int(round(hand.palm_uv[0])), int(round(hand.palm_uv[1]))
    pu = int(np.clip(pu, 0, w - 1))
    pv = int(np.clip(pv, 0, h - 1))
    cv2.circle(img, (pu, pv), 8, color_rgb, 2, cv2.LINE_AA)
    cv2.circle(img, (pu, pv), 14, color_rgb, 1, cv2.LINE_AA)

    if draw_vector:
        center = (w // 2, h // 2)
        cv2.arrowedLine(img, center, (pu, pv), color_rgb, 2, cv2.LINE_AA, tipLength=0.08)

    # Label
    text = label or f"Hand {hand.hand_id} ({hand.handedness or '?'})"
    cv2.putText(img, text, (pu + 16, pv + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


class HandTrackingWorker:
    """Decoupled hand tracking thread running at camera/model cadence."""

    def __init__(self, engine: HandControlEngine) -> None:
        self.engine = engine
        self._lock = threading.Lock()
        self._frame_slot = LatestFrameSlot()
        self._latest_depth_map: np.ndarray | None = None
        self._latest_gesture_state: GestureState | None = None
        self._running = False
        self._thread: threading.Thread | None = None

        self.last_latency_ms: float = 0.0
        self.tracking_fps: float = 0.0
        self._fps_samples: list[float] = []

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="HandTrackingWorker")
        self._thread.start()

    def submit_frame(
        self,
        frame_rgb: np.ndarray,
        frame_id: int | str,
        timestamp: float,
        depth_map: np.ndarray | None = None,
    ) -> None:
        if depth_map is not None:
            with self._lock:
                self._latest_depth_map = depth_map
        self._frame_slot.put(frame_rgb, int(frame_id) if isinstance(frame_id, int) else 0, timestamp)

    def get_latest(self) -> GestureState | None:
        with self._lock:
            return self._latest_gesture_state

    def _loop(self) -> None:
        while self._running:
            item = self._frame_slot.get(timeout=0.1)
            if item is None:
                continue
            frame_rgb, frame_id, ts = item
            with self._lock:
                d_map = self._latest_depth_map
            t0 = time.perf_counter()
            try:
                state = self.engine.update(frame_rgb, ts, frame_id, depth_map=d_map)
                dt = (time.perf_counter() - t0) * 1000.0
                with self._lock:
                    self._latest_gesture_state = state
                    self.last_latency_ms = dt
                    if dt > 1e-4:
                        self._fps_samples.append(1000.0 / dt)
                        if len(self._fps_samples) > 20:
                            self._fps_samples.pop(0)
                        self.tracking_fps = float(np.median(self._fps_samples))
            except Exception:
                time.sleep(0.01)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            self.engine.close()
        except Exception:
            pass


class SyntheticCameraProvider:
    """Generates synthetic animated frames when physical camera is unavailable."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = float(fps)
        self.sequence_id = 0

    def read(self) -> tuple[bool, np.ndarray]:
        t = time.monotonic()
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background gradient
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        img[..., 0] = (xx * 120 + 20).astype(np.uint8)
        img[..., 1] = (yy * 100 + 30).astype(np.uint8)
        img[..., 2] = ((1.0 - xx) * 150 + 20).astype(np.uint8)
        # Animated sphere
        cx = int(self.width / 2 + np.sin(t * 2.0) * (self.width / 4))
        cy = int(self.height / 2 + np.cos(t * 2.0) * (self.height / 6))
        cv2.circle(img, (cx, cy), 60, (230, 230, 240), -1, cv2.LINE_AA)
        self.sequence_id += 1
        return True, img


class NativeLiveApp:
    """The complete native live computer-vision application."""

    def __init__(
        self,
        camera_device: int | str = 0,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        depth_model_path: str = "models/depth-anything-v2-small",
        quality_profile: str | QualityProfile = QualityProfile.BALANCED,
        initial_mode: int | AppMode = AppMode.MULTILIGHT,
        headless: bool = False,
        fullscreen: bool = False,
        use_synthetic_camera: bool = False,
    ) -> None:
        self.requested_device = camera_device
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.depth_model_path = depth_model_path
        self.headless = bool(headless)
        self.use_fullscreen = bool(fullscreen)
        self.use_synthetic = bool(use_synthetic_camera)

        self.current_mode = AppMode(int(initial_mode))
        self.quality_profile = QualityProfile(quality_profile)
        self.profile_config = PROFILE_CONFIGS[self.quality_profile]

        # Feature flags
        self.show_hud = True
        self.show_skeleton = True
        self.shadows_enabled = True
        self.volumetrics_enabled = True

        # Pipeline components
        self.camera_worker: CameraCaptureWorker | None = None
        self.synthetic_camera: SyntheticCameraProvider | None = None
        self.depth_worker: DepthWorker | None = None
        self.depth_frame_buffer: LatestFrameBuffer | None = None
        self.depth_buffer: LatestDepthBuffer | None = None
        self.hand_worker: HandTrackingWorker | None = None
        self.persistent_worker: PersistentMapWorker | None = None
        self.pose_estimator: PoseEstimator | None = None
        self.geometry_backend: TorchGeometryBackend | None = None
        self.window: NativeOpenGLWindow | None = None

        # Camera model
        self.camera = CameraModel(
            width=camera_width,
            height=camera_height,
            fx=float(camera_width) * 0.82,
            fy=float(camera_width) * 0.82,
            cx=float(camera_width) / 2.0,
            cy=float(camera_height) / 2.0,
        )

        # State cache
        self.previous_geometry: GeometryState | None = None
        self.latest_rendered_frame: np.ndarray | None = None
        self.frames_rendered: int = 0
        self.start_time: float = 0.0

        # Telemetry metrics
        self.render_fps: float = 0.0
        self.render_latency_ms: float = 0.0
        self._render_times: list[float] = []
        self._last_render_ts: float | None = None
        self._is_running: bool = False

    def start(self) -> None:
        """Initialize all workers and start the native runtime."""
        self.start_time = time.monotonic()
        self._is_running = True

        # 1. Start Camera Worker (or Synthetic Fallback)
        if not self.use_synthetic:
            try:
                self.camera_worker = CameraCaptureWorker(
                    device=self.requested_device,
                    width=self.camera_width,
                    height=self.camera_height,
                    fps=self.camera_fps,
                )
                self.camera_worker.start()
                # Update camera dimensions to negotiated values
                self.camera_width = self.camera_worker.actual_width
                self.camera_height = self.camera_worker.actual_height
                self.camera = CameraModel(
                    width=self.camera_width,
                    height=self.camera_height,
                    fx=float(self.camera_width) * 0.82,
                    fy=float(self.camera_width) * 0.82,
                    cx=float(self.camera_width) / 2.0,
                    cy=float(self.camera_height) / 2.0,
                )
            except Exception as exc:
                if self.headless:
                    self.synthetic_camera = SyntheticCameraProvider(self.camera_width, self.camera_height, self.camera_fps)
                else:
                    raise exc
        else:
            self.synthetic_camera = SyntheticCameraProvider(self.camera_width, self.camera_height, self.camera_fps)

        # 2. Start Depth Worker
        self.depth_frame_buffer = LatestFrameBuffer()
        self.depth_buffer = LatestDepthBuffer()
        depth_source = os.getenv("NRW_DEPTH_SOURCE", "local").lower()
        if depth_source == "colleague":
            provider = ColleagueDepthProvider(device="auto", input_size=self.profile_config.depth_size[0], fp16=True)
        else:
            provider = DepthAnythingProvider(
                model_path=self.depth_model_path,
                device="auto",
                use_fp16=True,
                input_size=self.profile_config.depth_size,
            )
        self.depth_worker = DepthWorker(provider.compute, self.depth_frame_buffer, self.depth_buffer)
        self.depth_worker.start()

        # 3. Start Hand Tracking Worker (max_hands=2 for L5)
        hand_engine = HandControlEngine(max_hands=2, filter_mode="one_euro")
        self.hand_worker = HandTrackingWorker(hand_engine)
        self.hand_worker.start()

        # 4. Start Persistent Mapping Worker (sidecar at profile Hz)
        self.persistent_worker = PersistentMapWorker(
            self.camera,
            map_update_hz=self.profile_config.persistent_hz,
        )
        self.persistent_worker.start()
        self.pose_estimator = PoseEstimator(self.camera, max_samples=400, reprojection_error=3.0)

        # 5. Initialize CUDA/CPU Geometry Backend
        self.geometry_backend = TorchGeometryBackend(device="auto")

        # 6. Initialize Native OpenGL Window
        if not self.headless:
            self.window = NativeOpenGLWindow(
                width=self.camera_width,
                height=self.camera_height,
                title="NRW Computer Vision — Native Live Challenge",
                fullscreen=self.use_fullscreen,
            )
            if self.window._is_open:
                self.window.key_handlers.append(self._on_key_event)

    def _on_key_event(self, key: int, action: int, mods: int) -> None:
        """Handle keyboard commands deterministically."""
        # GLFW key constants
        KEY_1, KEY_9 = 49, 57
        KEY_KP_1, KEY_KP_9 = 321, 329
        KEY_D = 68
        KEY_P = 80
        KEY_V = 86
        KEY_S = 83
        KEY_H = 72
        KEY_R = 82
        KEY_C = 67
        KEY_Q = 81
        KEY_ESC = 256

        # Modes 1-9
        if KEY_1 <= key <= KEY_9:
            self.current_mode = AppMode(key - KEY_1 + 1)
        elif KEY_KP_1 <= key <= KEY_KP_9:
            self.current_mode = AppMode(key - KEY_KP_1 + 1)
        elif key == KEY_D:
            self.show_hud = not self.show_hud
        elif key == KEY_P:
            self.cycle_quality_profile()
        elif key == KEY_V:
            self.volumetrics_enabled = not self.volumetrics_enabled
        elif key == KEY_S:
            self.shadows_enabled = not self.shadows_enabled
        elif key == KEY_H:
            self.show_skeleton = not self.show_skeleton
        elif key == KEY_R:
            self.reset_persistent_map()
        elif key == KEY_C:
            self.recalibrate_camera()
        elif key in (KEY_Q, KEY_ESC):
            self._is_running = False

    def cycle_quality_profile(self) -> None:
        """Cycle quality profile LOW -> BALANCED -> HIGH -> LOW."""
        profiles = [QualityProfile.LOW, QualityProfile.BALANCED, QualityProfile.HIGH]
        idx = (profiles.index(self.quality_profile) + 1) % len(profiles)
        self.quality_profile = profiles[idx]
        self.profile_config = PROFILE_CONFIGS[self.quality_profile]
        if self.persistent_worker is not None:
            self.persistent_worker.map_update_hz = self.profile_config.persistent_hz

    def reset_persistent_map(self) -> None:
        """Reset the Phase 5 persistent surfel map and camera pose."""
        if self.persistent_worker is not None:
            self.persistent_worker.reset()
        self.previous_geometry = None

    def recalibrate_camera(self) -> None:
        """Reset camera model to nominal calibrated intrinsics."""
        self.camera = CameraModel(
            width=self.camera_width,
            height=self.camera_height,
            fx=float(self.camera_width) * 0.82,
            fy=float(self.camera_width) * 0.82,
            cx=float(self.camera_width) / 2.0,
            cy=float(self.camera_height) / 2.0,
        )

    def step(self) -> np.ndarray | None:
        """Execute one complete live render step at native display cadence."""
        if not self._is_running:
            return None

        # Poll window events
        if self.window is not None:
            self.window.poll_events()
            if self.window.should_close():
                self._is_running = False
                return None

        t_step_start = time.perf_counter()
        now = time.monotonic()

        # Update render FPS
        if self._last_render_ts is not None:
            delta = now - self._last_render_ts
            if delta > 1e-4:
                self._render_times.append(1.0 / delta)
                if len(self._render_times) > 30:
                    self._render_times.pop(0)
                self.render_fps = float(np.median(self._render_times))
        self._last_render_ts = now

        # 1. Fetch newest physical camera frame (non-blocking or brief wait)
        frame_rgb: np.ndarray
        frame_id: int
        frame_ts: float

        if self.camera_worker is not None:
            packet = self.camera_worker.slot.get_latest()
            if packet is None:
                packet = self.camera_worker.slot.get(timeout=0.2)
            if packet is None:
                return None
            frame_rgb, frame_id, frame_ts = packet
        elif self.synthetic_camera is not None:
            ok, frame_rgb = self.synthetic_camera.read()
            if not ok:
                return None
            frame_id = self.synthetic_camera.sequence_id
            frame_ts = now
        else:
            return None

        # 2. Submit frame to AI workers
        # Depth neural net downsampled according to Quality Profile
        target_h, target_w = self.profile_config.depth_size
        small_frame = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if self.depth_frame_buffer is not None:
            self.depth_frame_buffer.put(small_frame, frame_id, frame_ts)

        # Hand tracking worker consumes full RGB frame
        if self.hand_worker is not None:
            latest_d = self.depth_buffer.get() if self.depth_buffer else None
            self.hand_worker.submit_frame(
                frame_rgb,
                frame_id,
                frame_ts,
                depth_map=latest_d.depth if latest_d else None,
            )

        # 3. Retrieve newest depth state (never blocks on inference)
        depth_state: DepthState | None = self.depth_buffer.get() if self.depth_buffer else None
        depth_full: np.ndarray
        valid_full: np.ndarray

        if depth_state is not None:
            if depth_state.depth.shape != (self.camera.height, self.camera.width):
                depth_full = cv2.resize(
                    depth_state.depth,
                    (self.camera.width, self.camera.height),
                    interpolation=cv2.INTER_LINEAR,
                )
                valid_full = cv2.resize(
                    depth_state.valid_mask.astype(np.uint8),
                    (self.camera.width, self.camera.height),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            else:
                depth_full = depth_state.depth
                valid_full = depth_state.valid_mask if depth_state.valid_mask is not None else np.ones(depth_full.shape, bool)
            depth_age_frames = max(0, int(frame_id) - int(depth_state.source_frame_id)) if isinstance(depth_state.source_frame_id, int) else 0
            depth_age_ms = (frame_ts - depth_state.timestamp) * 1000.0
        else:
            # Cold-start fallback before first depth inference finishes
            depth_full = np.full((self.camera.height, self.camera.width), 1.5, dtype=np.float32)
            valid_full = np.ones(depth_full.shape, dtype=bool)
            depth_age_frames = 0
            depth_age_ms = 0.0

        # 4. Compute full-resolution 3D points and surface normals (1-2 ms on CUDA)
        geometry: GeometryState
        if self.geometry_backend is not None:
            geometry = self.geometry_backend.process_depth(
                depth_full,
                self.camera,
                valid_mask=valid_full,
                frame_id=frame_id,
                timestamp=frame_ts,
            )
        else:
            # Fallback
            geometry = GeometryState(
                frame_ts,
                frame_id,
                depth_full,
                np.zeros((*depth_full.shape, 3), np.float32),
                valid_full,
                self.camera,
                DepthScaleMode.RELATIVE,
            )

        # 5. Retrieve newest hand gesture state
        gesture_state = self.hand_worker.get_latest() if self.hand_worker else None
        hands: tuple[TrackedHand, ...] = gesture_state.hands if gesture_state else ()

        # 6. Construct lights according to Mode
        lights: list[LightState] = []
        if self.current_mode in (AppMode.DIFFUSE, AppMode.SPECULAR, AppMode.SHADOWS, AppMode.GESTURE):
            # Single-hand or default interactive light
            if hands:
                hand = hands[0]
                z_val = hand.depth_z if hand.depth_z and hand.depth_z > 0.1 else 0.7
                pu, pv = hand.palm_uv
                pos = self.camera.unproject(pu, pv, z_val).astype(np.float32)
                # Modulate intensity by confidence / openness
                intensity = 0.78 if hand.palm_width_px and hand.palm_width_px > 50 else 0.64
                lights.append(LightState(
                    position_camera=pos,
                    intensity=intensity,
                    color_rgb=np.array([1.0, 0.95, 0.88], dtype=np.float32),
                    confidence=hand.confidence,
                    source_hand=hand.hand_id,
                    light_id=0,
                ))
            else:
                # Orbiting virtual light when hand is not in frame
                t_orbit = now * 1.5
                orb_x = np.sin(t_orbit) * 0.35
                orb_y = -0.25 + np.cos(t_orbit * 0.7) * 0.15
                orb_z = 0.65
                lights.append(LightState(
                    position_camera=np.array([orb_x, orb_y, orb_z], dtype=np.float32),
                    intensity=0.72,
                    color_rgb=np.array([1.0, 0.95, 0.88], dtype=np.float32),
                    confidence=1.0,
                    source_hand=-1,
                    light_id=0,
                ))
        elif self.current_mode == AppMode.MULTILIGHT:
            # Level 5: Two simultaneous colored lights (Hand 0: Cyan, Hand 1: Amber)
            if len(hands) >= 2:
                # True two-hand control
                h0, h1 = hands[0], hands[1]
                z0 = h0.depth_z if h0.depth_z and h0.depth_z > 0.1 else 0.7
                z1 = h1.depth_z if h1.depth_z and h1.depth_z > 0.1 else 0.7
                pos0 = self.camera.unproject(h0.palm_uv[0], h0.palm_uv[1], z0).astype(np.float32)
                pos1 = self.camera.unproject(h1.palm_uv[0], h1.palm_uv[1], z1).astype(np.float32)
                lights.append(LightState(
                    position_camera=pos0,
                    intensity=0.72,
                    color_rgb=np.array([0.2, 0.75, 1.0], dtype=np.float32),  # Cyan
                    confidence=h0.confidence,
                    source_hand=h0.hand_id,
                    light_id=0,
                ))
                lights.append(LightState(
                    position_camera=pos1,
                    intensity=0.72,
                    color_rgb=np.array([1.0, 0.55, 0.15], dtype=np.float32),  # Amber
                    confidence=h1.confidence,
                    source_hand=h1.hand_id,
                    light_id=1,
                ))
            elif len(hands) == 1:
                # 1 hand in view: mirror second light so dual penumbra & beams are visible
                h0 = hands[0]
                z0 = h0.depth_z if h0.depth_z and h0.depth_z > 0.1 else 0.7
                pos0 = self.camera.unproject(h0.palm_uv[0], h0.palm_uv[1], z0).astype(np.float32)
                pos1 = np.array([-pos0[0], pos0[1], pos0[2]], dtype=np.float32)
                lights.append(LightState(
                    position_camera=pos0,
                    intensity=0.72,
                    color_rgb=np.array([0.2, 0.75, 1.0], dtype=np.float32),  # Cyan
                    confidence=h0.confidence,
                    source_hand=h0.hand_id,
                    light_id=0,
                ))
                lights.append(LightState(
                    position_camera=pos1,
                    intensity=0.62,
                    color_rgb=np.array([1.0, 0.55, 0.15], dtype=np.float32),  # Amber
                    confidence=0.85,
                    source_hand=-1,
                    light_id=1,
                ))
            else:
                # Dual virtual lights
                lights.append(LightState(
                    position_camera=np.array([-0.3, -0.2, 0.65], dtype=np.float32),
                    intensity=0.68,
                    color_rgb=np.array([0.2, 0.75, 1.0], dtype=np.float32),
                    confidence=1.0,
                    source_hand=-1,
                    light_id=0,
                ))
                lights.append(LightState(
                    position_camera=np.array([0.3, -0.2, 0.65], dtype=np.float32),
                    intensity=0.68,
                    color_rgb=np.array([1.0, 0.55, 0.15], dtype=np.float32),
                    confidence=1.0,
                    source_hand=-1,
                    light_id=1,
                ))

        # 7. Render effect for selected mode
        rendered: np.ndarray
        lighting_stats: dict[str, float] = {}

        if self.current_mode == AppMode.FEED:
            rendered = frame_rgb.copy()
        elif self.current_mode == AppMode.DEPTH:
            rendered = _heatmap(depth_full, valid_full)
        elif self.current_mode == AppMode.NORMALS:
            rendered = normals_to_rgb(geometry.normals, geometry.valid_mask)
        elif self.current_mode == AppMode.DIFFUSE:
            rendered, lighting_stats = shade_geometry(
                frame_rgb,
                geometry,
                lights,
                ambient=0.40,
                specular_strength=0.0,
                shadows=False,
                volumetrics=False,
            )
        elif self.current_mode == AppMode.SPECULAR:
            rendered, lighting_stats = shade_geometry(
                frame_rgb,
                geometry,
                lights,
                ambient=0.38,
                specular_strength=0.16,
                shininess=40.0,
                shadows=False,
                volumetrics=False,
            )
        elif self.current_mode in (AppMode.SHADOWS, AppMode.GESTURE):
            rendered, lighting_stats = shade_geometry(
                frame_rgb,
                geometry,
                lights,
                ambient=0.38,
                specular_strength=0.16,
                shininess=40.0,
                shadows=self.shadows_enabled,
                volumetrics=False,
            )
        elif self.current_mode == AppMode.MULTILIGHT:
            rendered, lighting_stats = shade_geometry(
                frame_rgb,
                geometry,
                lights,
                ambient=0.36,
                specular_strength=0.16,
                shininess=44.0,
                shadows=self.shadows_enabled,
                volumetrics=self.volumetrics_enabled,
            )
        elif self.current_mode == AppMode.INFINITY:
            # Mode 9: Persistent Infinity Geometry
            if self.previous_geometry is not None and self.pose_estimator is not None and self.persistent_worker is not None:
                pose_res = self.pose_estimator.estimate_pose(self.previous_geometry, geometry)
                if pose_res.valid:
                    self.persistent_worker.submit(geometry, pose_res, frame_id=frame_id, timestamp=frame_ts)
            self.previous_geometry = geometry

            # Query persistent map snapshot
            p_snapshot = self.persistent_worker.get_latest_snapshot() if self.persistent_worker else None
            if p_snapshot is not None and p_snapshot.surfel_count > 50:
                p_depth = p_snapshot.projected_depth
                p_conf = p_snapshot.projected_confidence
                has_p = np.isfinite(p_depth) & (p_conf > 0.1)
                rendered = frame_rgb.copy()
                # Tint persistent reconstructed surfels green/cyan
                rendered[has_p] = np.clip(
                    rendered[has_p].astype(np.float32) * 0.4 + np.array([30, 220, 150], dtype=np.float32) * 0.6,
                    0, 255,
                ).astype(np.uint8)
            else:
                rendered = frame_rgb.copy()

        # 8. Draw Hand Skeleton Overlay
        if self.show_skeleton and hands:
            for hand in hands:
                hand_col = (0, 240, 255) if hand.hand_id == 0 else (255, 140, 30)
                draw_hand_skeleton(rendered, hand, color_rgb=hand_col, draw_vector=True)

        # Emitters are projected from the same camera-space states used above
        # for shading, shadows, and volumetrics.
        if lights:
            rendered = render_light_orbs(
                rendered,
                self.camera,
                lights,
                depth=geometry.depth,
                valid=geometry.valid_mask,
            )

        # 9. Draw Professional HUD Overlay
        if self.show_hud:
            self._draw_hud_overlay(
                rendered,
                frame_id=frame_id,
                depth_age_frames=depth_age_frames,
                depth_age_ms=depth_age_ms,
                lights=lights,
                hands=hands,
            )

        # 10. Present frame via OpenGL Window
        if self.window is not None:
            self.window.render_frame(rendered)

        self.latest_rendered_frame = rendered
        self.frames_rendered += 1
        self.render_latency_ms = (time.perf_counter() - t_step_start) * 1000.0
        return rendered

    def _draw_hud_overlay(
        self,
        canvas: np.ndarray,
        frame_id: int | str,
        depth_age_frames: int,
        depth_age_ms: float,
        lights: list[LightState],
        hands: tuple[TrackedHand, ...],
    ) -> None:
        """Draw semi-transparent HUD telemetry panels, mode banner, and keybindings."""
        h, w = canvas.shape[:2]

        # Top Header Bar
        _draw_transparent_box(canvas, 0, 0, w, 32, color=(12, 12, 14), alpha=0.82)
        title = "NRW COMPUTER VISION  |  LIVE RUNTIME"
        cv2.putText(canvas, title, (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        cuda_info = torch_cuda_status()
        dev_name = str(cuda_info.get("device", "CPU"))[:24] if cuda_info.get("available") else "CPU"
        profile_str = f"[{self.quality_profile.value}] | GPU: {dev_name}"
        cv2.putText(canvas, profile_str, (w - 290, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1, cv2.LINE_AA)

        # Mode Badge
        mode_titles = {
            AppMode.FEED: "1: Clean Camera Stream (L0)",
            AppMode.DEPTH: "2: Metric Monocular Depth (L1)",
            AppMode.NORMALS: "3: Camera-Facing Normals (L1)",
            AppMode.DIFFUSE: "4: Relit Diffuse (L2)",
            AppMode.SPECULAR: "5: Specular Highlights & Roughness (L2)",
            AppMode.SHADOWS: "6: Dynamic Screen-Space Shadows (L4)",
            AppMode.GESTURE: "7: 3D Gesture Light Control (L3)",
            AppMode.MULTILIGHT: "8: Multi-Light & Volumetric Scattering (L5)",
            AppMode.INFINITY: "9: Persistent Infinity Geometry (Phase 5)",
        }
        mode_str = f"MODE: {mode_titles.get(self.current_mode, str(self.current_mode))}"
        _draw_transparent_box(canvas, 10, 40, 440, 68, color=(18, 22, 28), alpha=0.75)
        cv2.putText(canvas, mode_str, (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 220, 50), 1, cv2.LINE_AA)

        # Telemetry Box (Left)
        cam_fps = self.camera_worker.actual_fps if self.camera_worker else self.camera_fps
        drops = self.camera_worker.dropped_frames if self.camera_worker else 0
        depth_fps = 1000.0 / max(self.depth_worker.last_inference_ms, 1e-3) if self.depth_worker else 0.0
        depth_lat = self.depth_worker.last_inference_ms if self.depth_worker else 0.0
        hand_fps = self.hand_worker.tracking_fps if self.hand_worker else 0.0
        hand_lat = self.hand_worker.last_latency_ms if self.hand_worker else 0.0
        surfel_count = self.persistent_worker.surfel_count if self.persistent_worker else 0

        _draw_transparent_box(canvas, 10, 76, 310, 212, color=(12, 14, 18), alpha=0.70)
        lines = [
            f"Render FPS : {self.render_fps:5.1f} ({self.render_latency_ms:4.1f} ms)",
            f"Camera FPS : {cam_fps:5.1f} | Drops: {drops}",
            f"Depth FPS  : {depth_fps:5.1f} ({depth_lat:4.1f} ms)",
            f"Depth Age  : {depth_age_frames} frames ({depth_age_ms:4.1f} ms)",
            f"Hands FPS  : {hand_fps:5.1f} ({hand_lat:4.1f} ms) | Count: {len(hands)}",
            f"Surfel Map : {surfel_count} surfels",
            f"Shadows    : {'ON' if self.shadows_enabled else 'OFF'} | Volumetrics: {'ON' if self.volumetrics_enabled else 'OFF'}",
        ]
        y_pos = 96
        for line in lines:
            cv2.putText(canvas, line, (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (225, 230, 235), 1, cv2.LINE_AA)
            y_pos += 16

        # Active Lights Panel (Right)
        if lights:
            _draw_transparent_box(canvas, w - 310, 112, w - 10, 112 + len(lights) * 54 + 8, color=(12, 14, 18), alpha=0.70)
            ly = 130
            for i, lt in enumerate(lights):
                pos = lt.position_camera if lt.position_camera is not None else np.zeros(3)
                col_hex = (int(lt.color_rgb[0] * 255), int(lt.color_rgb[1] * 255), int(lt.color_rgb[2] * 255))
                uv = project_light_orb(self.camera, lt)
                uv_text = f"UV: [{uv[0]:.0f}, {uv[1]:.0f}]" if uv is not None else "UV: off-screen"
                cv2.circle(canvas, (w - 295, ly - 4), 6, col_hex, -1, cv2.LINE_AA)
                cv2.putText(canvas, f"Light {i} ({'H' + str(lt.source_hand) if lt.source_hand >= 0 else 'Virtual'})", (w - 282, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, f"XYZ: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] m | {uv_text}", (w - 282, ly + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 190, 200), 1, cv2.LINE_AA)
                cv2.putText(canvas, f"I: {lt.intensity:.2f} | Conf: {lt.confidence:.2f}", (w - 282, ly + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 170, 180), 1, cv2.LINE_AA)
                ly += 54

        # Bottom Shortcut Strip
        _draw_transparent_box(canvas, 0, h - 26, w, h, color=(12, 12, 14), alpha=0.82)
        help_str = "[1-9] Modes  [F] Fullscreen  [D] HUD  [P] Profile  [V] Volumetrics  [S] Shadows  [H] Skeleton  [R] Reset  [Q] Quit"
        cv2.putText(canvas, help_str, (12, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (210, 215, 220), 1, cv2.LINE_AA)

    def run(self, max_frames: int | None = None, timeout_sec: float | None = None) -> None:
        """Main execution loop until window closed, max frames reached, or timeout."""
        start_time = time.monotonic()
        while self._is_running:
            frame = self.step()
            if frame is None and not self._is_running:
                break
            if max_frames is not None and self.frames_rendered >= max_frames:
                break
            if timeout_sec is not None and (time.monotonic() - start_time) >= timeout_sec:
                break
            # Rate limit slightly if headless
            if self.headless:
                time.sleep(0.005)

    def stop(self) -> None:
        """Cleanly stop all threads and release resources."""
        self._is_running = False
        if self.depth_worker is not None:
            self.depth_worker.stop()
            self.depth_worker = None
        if self.hand_worker is not None:
            self.hand_worker.stop()
            self.hand_worker = None
        if self.persistent_worker is not None:
            self.persistent_worker.stop()
            self.persistent_worker = None
        if self.camera_worker is not None:
            self.camera_worker.stop()
            self.camera_worker = None
        if self.window is not None:
            self.window.close()
            self.window = None
