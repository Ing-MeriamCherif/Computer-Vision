"""Mode 5: Talel hand tracking overlay."""

import cv2

def render(snapshot):
    if snapshot.rgb_frame is None:
        return None, "MODE 5 — HANDS", "waiting for camera"
    image = snapshot.rgb_frame.copy()
    if snapshot.hand_state is not None:
        for hand in snapshot.hand_state.hands:
            u, v = map(int, hand.palm_uv)
            cv2.circle(image, (u, v), 12, (0, 255, 210), 2)
            cv2.putText(image, f"H{hand.hand_id} {hand.confidence:.2f}", (u + 14, v), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return image, "MODE 5 — HANDS (Talel tracker)", None
