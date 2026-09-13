"""Mode 6: Talel camera-space XYZ hand overlay."""

import cv2
import numpy as np

def render(snapshot):
    if snapshot.rgb_frame is None:
        return None, "MODE 6 — XYZ CONTRACT", "waiting for camera"
    image = snapshot.rgb_frame.copy()
    legend_y = max(120, image.shape[0] - 132)
    origin = (38, legend_y + 34)
    cv2.rectangle(image, (8, legend_y), (205, min(image.shape[0] - 38, legend_y + 104)), (12, 18, 22), -1)
    cv2.circle(image, origin, 3, (240, 240, 240), -1)
    cv2.arrowedLine(image, origin, (108, origin[1]), (255, 70, 70), 2, cv2.LINE_AA, tipLength=0.18)
    cv2.arrowedLine(image, origin, (origin[0], origin[1] + 54), (70, 255, 70), 2, cv2.LINE_AA, tipLength=0.18)
    cv2.putText(image, "+X right", (112, origin[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(image, "+Y down", (origin[0] + 5, origin[1] + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 255, 90), 1, cv2.LINE_AA)
    cv2.putText(image, "+Z forward", (12, legend_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 150, 255), 1, cv2.LINE_AA)
    xyz_by_id = {item.hand_id: item for item in snapshot.xyz}
    visible_hands = () if snapshot.hand_state is None else snapshot.hand_state.hands
    if not visible_hands:
        cv2.putText(image, "Waiting for a detected hand...", (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 120), 1, cv2.LINE_AA)
    for tracked in visible_hands:
        u, v = map(int, tracked.palm_uv)
        if tracked.landmarks_uv is not None and len(tracked.landmarks_uv):
            points = np.asarray(tracked.landmarks_uv, dtype=np.int32)
            x0, y0 = np.min(points, axis=0).tolist(); x1, y1 = np.max(points, axis=0).tolist()
            cv2.rectangle(image, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), (0, 220, 190), 2)
            for px, py in points:
                cv2.circle(image, (int(px), int(py)), 2, (0, 190, 255), -1)
        cv2.circle(image, (u, v), 10, (0, 255, 210), 2)
        xyz = xyz_by_id.get(tracked.hand_id)
        label = f"H{tracked.hand_id} depth pending" if xyz is None or xyz.xyz_camera is None else f"H{xyz.hand_id} XYZ=({xyz.xyz_camera[0]:.2f},{xyz.xyz_camera[1]:.2f},{xyz.xyz_camera[2]:.2f}) c={xyz.confidence:.2f}"
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        tx = min(max(4, u - text_width // 2), max(4, image.shape[1] - text_width - 4)); ty = max(22, v - 18)
        cv2.rectangle(image, (tx - 3, ty - 16), (tx + text_width + 3, ty + 4), (12, 18, 22), -1)
        cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 210), 1, cv2.LINE_AA)
    return image, "MODE 6 — XYZ CONTRACT (Talel camera coordinates)", None
