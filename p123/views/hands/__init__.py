"""Mode 5: Talel hand tracking overlay with Material 3 styling."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Standard 21-landmark hand skeletal connections
HAND_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (9, 10), (10, 11), (11, 12),           # Middle
    (13, 14), (14, 15), (15, 16),          # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm knuckles
)


def render(snapshot: Any) -> tuple[np.ndarray | None, str, str | None]:
    """Render Talel hand tracker with anti-aliased skeletal topology and modern chips."""
    if snapshot.rgb_frame is None:
        return None, "MODE 5 — HAND TRACKING", "waiting for camera"

    image = snapshot.rgb_frame.copy()
    visible_hands = snapshot.hand_state.hands if snapshot.hand_state is not None else ()

    if not visible_hands:
        # Subtle non-intrusive searching badge
        _sh, sw = image.shape[:2]
        badge_text = "Searching for hands... Present palm to camera"
        tw = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]
        bx = max(10, (sw - tw) // 2 - 12)
        by = 16
        cv2.rectangle(image, (bx, by), (bx + tw + 24, by + 26), (22, 26, 32), -1)
        cv2.rectangle(image, (bx, by), (bx + tw + 24, by + 26), (58, 70, 88), 1)
        cv2.circle(image, (bx + 10, by + 13), 3, (253, 214, 99), -1, cv2.LINE_AA)
        cv2.putText(image, badge_text, (bx + 20, by + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (242, 244, 246), 1, cv2.LINE_AA)

    for hand in visible_hands:
        u, v = round(hand.palm_uv[0]), round(hand.palm_uv[1])

        # Render 21 landmark skeleton if available
        if hand.landmarks_uv is not None and len(hand.landmarks_uv) >= 21:
            pts = np.asarray(hand.landmarks_uv, dtype=np.int32)
            # Bone segments
            for idx_a, idx_b in HAND_BONES:
                if idx_a < len(pts) and idx_b < len(pts):
                    p1 = (int(pts[idx_a][0]), int(pts[idx_a][1]))
                    p2 = (int(pts[idx_b][0]), int(pts[idx_b][1]))
                    cv2.line(image, p1, p2, (120, 217, 236), 1, cv2.LINE_AA)

            # Landmark joints
            for idx, (px, py) in enumerate(pts):
                is_tip = idx in (4, 8, 12, 16, 20)
                pt_color = (138, 180, 248) if is_tip else (120, 217, 236)
                radius = 3 if is_tip else 2
                cv2.circle(image, (int(px), int(py)), radius, pt_color, -1, cv2.LINE_AA)

            # Bounding box
            x0, y0 = np.min(pts, axis=0)
            x1, y1 = np.max(pts, axis=0)
            cv2.rectangle(image, (int(x0) - 8, int(y0) - 8), (int(x1) + 8, int(y1) + 8), (92, 138, 210), 1, cv2.LINE_AA)

        # Palm center rings
        cv2.circle(image, (u, v), 10, (129, 201, 149), 2, cv2.LINE_AA)
        cv2.circle(image, (u, v), 4, (129, 201, 149), -1, cv2.LINE_AA)

        # Floating Status Chip above hand
        status_text = "Fresh" if not hand.stale else "Coasting"
        status_color = (129, 201, 149) if not hand.stale else (253, 214, 99)
        label = f"H{hand.hand_id} {hand.confidence * 100:.0f}% [{status_text}]"
        tsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
        tw, _th = tsize
        tx = max(8, min(image.shape[1] - tw - 16, u - tw // 2))
        ty = max(24, v - 20)

        # Chip background
        cv2.rectangle(image, (tx - 6, ty - 16), (tx + tw + 6, ty + 6), (22, 26, 32), -1)
        cv2.rectangle(image, (tx - 6, ty - 16), (tx + tw + 6, ty + 6), (58, 70, 88), 1)
        cv2.circle(image, (tx - 1, ty - 5), 3, status_color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (tx + 6, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (242, 244, 246), 1, cv2.LINE_AA)

    return image, "MODE 5 — HAND TRACKING (Talel Tracker)", None
