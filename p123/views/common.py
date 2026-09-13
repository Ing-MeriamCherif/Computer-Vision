"""Shared display-only helpers; no inference or geometry work runs here."""

from __future__ import annotations

import cv2
import numpy as np


def fit(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def overlay(image: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28 + 17 * len(lines)), (12, 14, 18), -1)
    cv2.putText(out, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, 43 + i * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 220, 225), 1, cv2.LINE_AA)
    return out


def buttons(image: np.ndarray, active_mode: int) -> np.ndarray:
    out = image.copy()
    labels = ("1 RGB", "2 DEPTH", "3 NORMALS", "4 TEMP", "5 HANDS", "6 XYZ")
    h, w = out.shape[:2]
    button_w = max(1, w // len(labels))
    for idx, label in enumerate(labels, start=1):
        x0 = (idx - 1) * button_w
        x1 = w if idx == len(labels) else idx * button_w
        color = (38, 105, 150) if idx == active_mode else (28, 32, 40)
        cv2.rectangle(out, (x0, h - 32), (x1 - 2, h - 2), color, -1)
        cv2.putText(out, label, (x0 + 8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 245, 245), 1, cv2.LINE_AA)
    return out


def finish(image, title: str, waiting: str | None, snapshot, mode: int, display_size: tuple[int, int] | None) -> np.ndarray:
    metrics = snapshot.metrics
    xyz_age = max((item.age_ms for item in snapshot.xyz), default=None)
    xyz_state = "fresh" if snapshot.xyz and (xyz_age or 9999) <= 220 else "degraded" if snapshot.xyz else "none"
    lines = [
        f"capture {snapshot.rgb_capture_id} | CAM {metrics.capture_hz or 0:.1f} Hz | overwritten {metrics.overwritten_before_consumption}",
        f"depth {metrics.depth_hz or 0:.1f} Hz age p95 {metrics.depth_age_p95_ms or 0:.0f} ms | normals {metrics.normal_hz or 0:.1f} Hz/{metrics.normal_age_p95_ms or 0:.0f}ms | temp {metrics.temporal_hz or 0:.1f} Hz",
        f"hands {metrics.hand_hz or 0:.1f} Hz/{0 if snapshot.hand_state is None else len(snapshot.hand_state.hands)} | XYZ {xyz_state}{'' if xyz_age is None else f' {xyz_age:.0f}ms'}",
    ]
    if snapshot.geometry_state is not None:
        lines.append(f"depth source {snapshot.geometry_state.source_frame_id} | processing {snapshot.geometry_state.processing_frame_id}")
    if waiting:
        lines.append(waiting)
    out = overlay(image, title, lines)
    if display_size is not None and (out.shape[1], out.shape[0]) != display_size:
        out = fit(out, display_size)
    return buttons(out, mode)
