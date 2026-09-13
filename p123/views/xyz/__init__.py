"""Mode 6: Talel camera-space XYZ hand contract overlay with Material 3 styling."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

HAND_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (9, 10), (10, 11), (11, 12),           # Middle
    (13, 14), (14, 15), (15, 16),          # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm knuckles
)


def render(snapshot: Any) -> tuple[np.ndarray | None, str, str | None]:
    """Render Talel metric camera coordinates (X right, Y down, Z forward in meters)."""
    if snapshot.rgb_frame is None:
        return None, "MODE 6 — XYZ CONTRACT", "waiting for camera"

    image = snapshot.rgb_frame.copy()
    h, w = image.shape[:2]

    # Coordinate Axis Legend Card (Bottom-Left)
    card_w = 210
    card_h = 92
    card_x = 12
    card_y = max(40, h - card_h - 12)

    cv2.rectangle(image, (card_x, card_y), (card_x + card_w, card_y + card_h), (22, 26, 32), -1)
    cv2.rectangle(image, (card_x, card_y), (card_x + card_w, card_y + card_h), (58, 70, 88), 1)

    cv2.putText(image, "METRIC CAMERA AXES (m)", (card_x + 10, card_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (138, 180, 248), 1, cv2.LINE_AA)

    origin = (card_x + 36, card_y + 48)
    cv2.circle(image, origin, 3, (242, 244, 246), -1, cv2.LINE_AA)

    # +X Axis (Right) - Red
    cv2.arrowedLine(image, origin, (origin[0] + 48, origin[1]), (242, 139, 130), 2, cv2.LINE_AA, tipLength=0.22)
    cv2.putText(image, "+X right", (origin[0] + 52, origin[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (242, 139, 130), 1, cv2.LINE_AA)

    # +Y Axis (Down) - Green
    cv2.arrowedLine(image, origin, (origin[0], origin[1] + 32), (129, 201, 149), 2, cv2.LINE_AA, tipLength=0.22)
    cv2.putText(image, "+Y down", (origin[0] + 4, origin[1] + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (129, 201, 149), 1, cv2.LINE_AA)

    # +Z Axis (Depth) - Blue
    cv2.putText(image, "⊙ +Z forward (depth)", (card_x + 92, card_y + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (138, 180, 248), 1, cv2.LINE_AA)

    xyz_by_id = {item.hand_id: item for item in (snapshot.xyz or ())}
    visible_hands = snapshot.hand_state.hands if snapshot.hand_state is not None else ()

    if not visible_hands:
        badge_text = "Waiting for detected hand target..."
        tw = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]
        bx = max(10, (w - tw) // 2 - 12)
        by = 16
        cv2.rectangle(image, (bx, by), (bx + tw + 24, by + 26), (22, 26, 32), -1)
        cv2.rectangle(image, (bx, by), (bx + tw + 24, by + 26), (58, 70, 88), 1)
        cv2.circle(image, (bx + 10, by + 13), 3, (253, 214, 99), -1, cv2.LINE_AA)
        cv2.putText(image, badge_text, (bx + 20, by + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (242, 244, 246), 1, cv2.LINE_AA)

    for tracked in visible_hands:
        u, v = round(tracked.palm_uv[0]), round(tracked.palm_uv[1])

        # Skeleton
        if tracked.landmarks_uv is not None and len(tracked.landmarks_uv) >= 21:
            pts = np.asarray(tracked.landmarks_uv, dtype=np.int32)
            for idx_a, idx_b in HAND_BONES:
                if idx_a < len(pts) and idx_b < len(pts):
                    p1 = (int(pts[idx_a][0]), int(pts[idx_a][1]))
                    p2 = (int(pts[idx_b][0]), int(pts[idx_b][1]))
                    cv2.line(image, p1, p2, (120, 217, 236), 1, cv2.LINE_AA)
            for idx, (px, py) in enumerate(pts):
                is_tip = idx in (4, 8, 12, 16, 20)
                cv2.circle(image, (int(px), int(py)), 3 if is_tip else 2, (138, 180, 248) if is_tip else (120, 217, 236), -1, cv2.LINE_AA)

            x0, y0 = np.min(pts, axis=0)
            x1, y1 = np.max(pts, axis=0)
            cv2.rectangle(image, (int(x0) - 8, int(y0) - 8), (int(x1) + 8, int(y1) + 8), (129, 201, 149), 1, cv2.LINE_AA)

        # Palm crosshair
        cv2.circle(image, (u, v), 10, (129, 201, 149), 2, cv2.LINE_AA)
        cv2.line(image, (u - 14, v), (u + 14, v), (129, 201, 149), 1, cv2.LINE_AA)
        cv2.line(image, (u, v - 14), (u, v + 14), (129, 201, 149), 1, cv2.LINE_AA)

        # Coordinate Chip
        xyz = xyz_by_id.get(tracked.hand_id)
        if xyz is not None and xyz.xyz_camera is not None:
            cx, cy_pos, cz = xyz.xyz_camera
            is_estimated = getattr(xyz, "estimated", False)
            is_fresh = (xyz.age_ms <= 220.0 and not is_estimated)
            status_tag = f"Fresh {xyz.age_ms:.0f}ms" if is_fresh else (f"Degraded {xyz.age_ms:.0f}ms" if not is_estimated else "Estimated")
            status_col = (129, 201, 149) if is_fresh else ((253, 214, 99) if not is_estimated else (138, 180, 248))
            coord_line = f"H{tracked.hand_id} XYZ: ({cx:+.2f}, {cy_pos:+.2f}, {cz:.2f})m"
        else:
            coord_line = f"H{tracked.hand_id} XYZ: Depth Pending"
            status_tag = "Pending"
            status_col = (253, 214, 99)

        full_label = f"{coord_line}  [{status_tag}]"
        tsize = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
        tw, _th = tsize
        tx = max(8, min(w - tw - 16, u - tw // 2))
        ty = max(26, v - 22)

        cv2.rectangle(image, (tx - 6, ty - 16), (tx + tw + 6, ty + 6), (22, 26, 32), -1)
        cv2.rectangle(image, (tx - 6, ty - 16), (tx + tw + 6, ty + 6), (58, 70, 88), 1)
        cv2.circle(image, (tx - 1, ty - 5), 3, status_col, -1, cv2.LINE_AA)
        cv2.putText(image, full_label, (tx + 6, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (242, 244, 246), 1, cv2.LINE_AA)

    return image, "MODE 6 — XYZ CONTRACT (Camera Metric XYZ)", None
