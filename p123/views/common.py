"""Google Material 3 design system and display rendering for P123 Live Diagnostics.

Implements clean dark theme, rounded surfaces, strong typography hierarchy,
restrained color palette, responsive sidebar navigation, telemetry header,
and polished connection/loading/error states.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

# ============================================================================
# Google Material 3 Dark Palette & Design Tokens (RGB)
# ============================================================================

COLOR_SURFACE_DARK = (16, 18, 22)        # Background deep obsidian
COLOR_CONTAINER_LOW = (22, 26, 32)       # Sidebar and header background
COLOR_CONTAINER = (28, 34, 42)           # Card / Viewport container
COLOR_CONTAINER_HIGH = (38, 46, 56)      # Inactive buttons, chips
COLOR_CONTAINER_ACTIVE = (35, 58, 88)    # Active button container
COLOR_BORDER_SUBTLE = (42, 50, 62)       # Subtle dividers and borders
COLOR_BORDER_STRONG = (58, 70, 88)       # Highlighted borders
COLOR_BORDER_ACTIVE = (92, 138, 210)     # Active button border

COLOR_PRIMARY_ACCENT = (138, 180, 248)   # Google Blue accent (#8ab4f8)
COLOR_TEXT_PRIMARY = (242, 244, 246)     # High-contrast white (#f2f4f6)
COLOR_TEXT_SECONDARY = (156, 163, 172)   # Muted gray (#9ca3ac)
COLOR_TEXT_TERTIARY = (110, 118, 128)    # Low-contrast hints (#6e7680)

COLOR_STATUS_GREEN = (129, 201, 149)     # Google Green (Healthy/Fresh #81c995)
COLOR_STATUS_AMBER = (253, 214, 99)      # Google Yellow (Warning/Pending/Degraded #fdd663)
COLOR_STATUS_RED = (242, 139, 130)       # Google Coral Red (Error/Stale #f28b82)
COLOR_STATUS_CYAN = (120, 217, 236)      # Google Cyan (Info/CUDA #78d9ec)
COLOR_STATUS_PURPLE = (197, 138, 249)    # Google Purple (#c58af9)

# Sidebar Navigation Items
NAV_ITEMS: tuple[tuple[int, str, str], ...] = (
    (1, "RGB Camera", "Sensor Feed"),
    (2, "Depth Map", "Mariem CUDA"),
    (3, "Surface Normals", "CUDA Multiscale"),
    (4, "Temporal Conf", "Frame Stability"),
    (5, "Hand Tracking", "Talel Landmarks"),
    (6, "XYZ Contract", "Metric 3D (m)"),
    (7, "Hand Relight", "Live 3D Light"),
)


# ============================================================================
# Vector Drawing Primitives (OpenCV Anti-Aliased)
# ============================================================================

def draw_rounded_rect(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    radius: int,
    fill_color: tuple[int, int, int] | None = None,
    border_color: tuple[int, int, int] | None = None,
    border_thickness: int = 1,
) -> None:
    """Draw a smooth rounded rectangle with optional fill and anti-aliased border."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    if w <= 0 or h <= 0:
        return
    radius = max(1, min(radius, w // 2, h // 2))

    if fill_color is not None:
        cv2.rectangle(img, (x + radius, y), (x + w - radius, y + h), fill_color, -1)
        cv2.rectangle(img, (x, y + radius), (x + w, y + h - radius), fill_color, -1)
        cv2.circle(img, (x + radius, y + radius), radius, fill_color, -1, cv2.LINE_AA)
        cv2.circle(img, (x + w - radius, y + radius), radius, fill_color, -1, cv2.LINE_AA)
        cv2.circle(img, (x + radius, y + h - radius), radius, fill_color, -1, cv2.LINE_AA)
        cv2.circle(img, (x + w - radius, y + h - radius), radius, fill_color, -1, cv2.LINE_AA)

    if border_color is not None and border_thickness > 0:
        cv2.line(img, (x + radius, y), (x + w - radius, y), border_color, border_thickness, cv2.LINE_AA)
        cv2.line(img, (x + radius, y + h), (x + w - radius, y + h), border_color, border_thickness, cv2.LINE_AA)
        cv2.line(img, (x, y + radius), (x, y + h - radius), border_color, border_thickness, cv2.LINE_AA)
        cv2.line(img, (x + w, y + radius), (x + w, y + h - radius), border_color, border_thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + radius, y + radius), (radius, radius), 180, 0, 90, border_color, border_thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, border_color, border_thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, border_color, border_thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, border_color, border_thickness, cv2.LINE_AA)


def draw_rounded_rect_alpha(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    radius: int,
    fill_color: tuple[int, int, int],
    alpha: float = 0.85,
    border_color: tuple[int, int, int] | None = None,
    border_thickness: int = 1,
) -> None:
    """Draw a translucent rounded rectangle card."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    if w <= 0 or h <= 0:
        return
    img_h, img_w = img.shape[:2]
    x1, y1 = max(0, min(x, img_w)), max(0, min(y, img_h))
    x2, y2 = max(0, min(x + w, img_w)), max(0, min(y + h, img_h))
    if x2 <= x1 or y2 <= y1:
        return

    sub = img[y1:y2, x1:x2]
    overlay = sub.copy()
    ox = x - x1
    oy = y - y1
    draw_rounded_rect(overlay, ox, oy, w, h, radius, fill_color=fill_color)
    cv2.addWeighted(overlay, alpha, sub, 1.0 - alpha, 0, sub)

    if border_color is not None:
        draw_rounded_rect(img, x, y, w, h, radius, fill_color=None, border_color=border_color, border_thickness=border_thickness)


def draw_pill(
    img: np.ndarray,
    x: int,
    y: int,
    text: str,
    text_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
    dot_color: tuple[int, int, int] | None = None,
    border_color: tuple[int, int, int] | None = None,
    font_scale: float = 0.36,
    font_thickness: int = 1,
    padding_x: int = 8,
    padding_y: int = 4,
) -> tuple[int, int]:
    """Draw a compact Material 3 pill badge with optional status indicator dot."""
    x, y = int(x), int(y)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
    tw, th = text_size
    dot_w = 12 if dot_color is not None else 0
    pw = tw + dot_w + padding_x * 2
    ph = max(18, th + padding_y * 2 + 2)
    radius = ph // 2

    draw_rounded_rect(img, x, y, pw, ph, radius, fill_color=bg_color, border_color=border_color)
    tx = x + padding_x
    if dot_color is not None:
        cv2.circle(img, (x + padding_x + 4, y + ph // 2), 3, dot_color, -1, cv2.LINE_AA)
        tx += dot_w
    ty = y + padding_y + th
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)
    return pw, ph


# ============================================================================
# Responsive Layout Geometry
# ============================================================================

def compute_layout(display_size: tuple[int, int]) -> dict[str, Any]:
    """Compute responsive dimensions and bounds for header, sidebar, and viewport."""
    w, h = display_size
    is_fhd = (w >= 1400 or h >= 800)

    header_h = int(np.clip(h * 0.062, 42, 68)) if is_fhd else min(48, max(38, int(h * 0.08)))
    sidebar_w = int(np.clip(w * 0.15, 160, 290)) if is_fhd else min(210, max(140, int(w * 0.22)))
    vx = sidebar_w + (8 if is_fhd else 6)
    vy = header_h + (8 if is_fhd else 6)
    vw = max(100, w - vx - (10 if is_fhd else 8))
    vh = max(100, h - vy - (10 if is_fhd else 8))

    footer_h = 58 if is_fhd else 44
    nav_y0 = header_h + (30 if is_fhd else 24)
    nav_available_h = h - nav_y0 - footer_h - (16 if is_fhd else 12)
    button_h = int(np.clip(nav_available_h // len(NAV_ITEMS) - (8 if is_fhd else 4), 34, 88))
    button_gap = 6 if is_fhd else 4

    buttons_rects = []
    for idx, (mode_id, name, sub) in enumerate(NAV_ITEMS):
        bx = 10 if is_fhd else 8
        by = nav_y0 + idx * (button_h + button_gap)
        bw = sidebar_w - (20 if is_fhd else 16)
        buttons_rects.append((mode_id, bx, by, bw, button_h, name, sub))

    return {
        "w": w,
        "h": h,
        "is_fhd": is_fhd,
        "header_h": header_h,
        "sidebar_w": sidebar_w,
        "vx": vx,
        "vy": vy,
        "vw": vw,
        "vh": vh,
        "nav_y0": nav_y0,
        "button_h": button_h,
        "button_gap": button_gap,
        "buttons_rects": buttons_rects,
    }


def hit_test_navigation(x: int, y: int, display_size: tuple[int, int]) -> int | None:
    """Determine which navigation mode was clicked, if any."""
    layout = compute_layout(display_size)
    if x >= layout["sidebar_w"]:
        return None
    for mode_id, bx, by, bw, bh, _name, _sub in layout["buttons_rects"]:
        if bx <= x <= bx + bw and by <= y <= by + bh:
            return mode_id
    return None


# ============================================================================
# Header Telemetry Bar
# ============================================================================

def draw_header(
    canvas: np.ndarray,
    snapshot: Any,
    layout: dict[str, Any],
    display_fps: float | None = None,
) -> None:
    """Draw sleek top live-status header with camera, display, depth, and hardware telemetry."""
    w = layout["w"]
    header_h = layout["header_h"]
    is_fhd = layout.get("is_fhd", False)

    # Header background surface
    draw_rounded_rect(canvas, 0, 0, w, header_h, 0, fill_color=COLOR_CONTAINER_LOW)
    cv2.line(canvas, (0, header_h - 1), (w, header_h - 1), COLOR_BORDER_SUBTLE, 1, cv2.LINE_AA)

    # Left: Brand / Mode Indicator
    brand_x = 12 if is_fhd else 10
    brand_h = 28 if is_fhd else 24
    brand_y = (header_h - brand_h) // 2
    draw_pill(
        canvas,
        brand_x,
        brand_y,
        "P123 DIAGNOSTICS",
        COLOR_TEXT_PRIMARY,
        COLOR_CONTAINER,
        dot_color=COLOR_PRIMARY_ACCENT,
        border_color=COLOR_BORDER_SUBTLE,
        font_scale=0.42 if is_fhd else 0.36,
        padding_x=10 if is_fhd else 8,
        padding_y=4 if is_fhd else 3,
    )

    # Right: Telemetry chips
    metrics = snapshot.metrics if snapshot is not None else None
    cam_hz = (metrics.capture_hz if metrics else None) or 0.0
    depth_hz = (metrics.depth_hz if metrics else None) or 0.0
    norm_hz = (metrics.normal_hz if metrics else None) or 0.0
    hand_hz = (metrics.hand_hz if metrics else None) or 0.0
    depth_age = (metrics.depth_age_p95_ms if metrics else None) or 0.0

    xyz_age = max((item.age_ms for item in (snapshot.xyz if snapshot else ())), default=None)
    has_hands = snapshot is not None and snapshot.hand_state is not None and len(snapshot.hand_state.hands) > 0
    if snapshot is not None and snapshot.xyz:
        xyz_text = f"XYZ {xyz_age:.0f}ms" if xyz_age is not None else "XYZ OK"
        xyz_color = COLOR_STATUS_GREEN if (xyz_age or 9999) <= 220 else COLOR_STATUS_AMBER
    elif has_hands:
        xyz_text = "XYZ Pending"
        xyz_color = COLOR_STATUS_AMBER
    else:
        xyz_text = "No Target"
        xyz_color = COLOR_TEXT_TERTIARY

    cam_dot = COLOR_STATUS_GREEN if cam_hz >= 25.0 else COLOR_STATUS_AMBER if cam_hz >= 10.0 else COLOR_STATUS_RED
    ui_hz = display_fps or 0.0
    ui_dot = COLOR_PRIMARY_ACCENT if ui_hz >= 25.0 else COLOR_STATUS_AMBER if ui_hz >= 10.0 else COLOR_STATUS_RED

    chips: list[tuple[str, tuple[int, int, int], tuple[int, int, int] | None]] = []
    if w >= 740:
        chips.append((f"CAM {cam_hz:.1f} FPS", COLOR_TEXT_PRIMARY, cam_dot))
        chips.append((f"UI {ui_hz:.1f} FPS", COLOR_TEXT_PRIMARY, ui_dot))
        chips.append((f"DEPTH {depth_hz:.1f}Hz ({depth_age:.0f}ms)", COLOR_TEXT_PRIMARY, COLOR_STATUS_CYAN))
        chips.append((f"NORM {norm_hz:.1f}Hz", COLOR_TEXT_PRIMARY, COLOR_STATUS_PURPLE))
        chips.append((f"HANDS {hand_hz:.1f}Hz", COLOR_TEXT_PRIMARY, COLOR_STATUS_GREEN if has_hands else COLOR_TEXT_TERTIARY))
        chips.append((xyz_text, xyz_color, xyz_color))
        chips.append(("CUDA:0", COLOR_STATUS_CYAN, COLOR_STATUS_CYAN))
    else:
        chips.append((f"{cam_hz:.0f} CAM", COLOR_TEXT_PRIMARY, cam_dot))
        chips.append((f"{ui_hz:.0f} UI", COLOR_TEXT_PRIMARY, ui_dot))
        chips.append((f"{depth_hz:.0f} D", COLOR_TEXT_PRIMARY, COLOR_STATUS_CYAN))
        chips.append((xyz_text, xyz_color, xyz_color))
        chips.append(("CUDA", COLOR_STATUS_CYAN, None))

    curr_x = w - 12
    chip_h = 26 if is_fhd else 22
    chip_y = (header_h - chip_h) // 2
    chip_font = 0.38 if is_fhd else 0.34
    chip_pad_x = 8 if is_fhd else 6
    chip_pad_y = 3 if is_fhd else 2
    for text, tcolor, dcolor in reversed(chips):
        tsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, chip_font, 1)[0]
        dot_offset = 14 if dcolor is not None else 0
        pw = tsize[0] + dot_offset + chip_pad_x * 2
        curr_x -= pw + 6
        if curr_x < (240 if is_fhd else 180):
            break
        draw_pill(
            canvas,
            curr_x,
            chip_y,
            text,
            tcolor,
            COLOR_CONTAINER,
            dot_color=dcolor,
            border_color=COLOR_BORDER_SUBTLE,
            font_scale=chip_font,
            padding_x=chip_pad_x,
            padding_y=chip_pad_y,
        )


# ============================================================================
# Sidebar Navigation
# ============================================================================

def draw_sidebar(
    canvas: np.ndarray,
    active_mode: int,
    layout: dict[str, Any],
) -> None:
    """Draw responsive Material 3 sidebar with 6 mode buttons and shortcut hints."""
    sidebar_w = layout["sidebar_w"]
    header_h = layout["header_h"]
    h = layout["h"]
    is_fhd = layout.get("is_fhd", False)

    # Background surface
    draw_rounded_rect(canvas, 0, header_h, sidebar_w, h - header_h, 0, fill_color=COLOR_CONTAINER_LOW)
    cv2.line(canvas, (sidebar_w - 1, header_h), (sidebar_w - 1, h), COLOR_BORDER_SUBTLE, 1, cv2.LINE_AA)

    # Navigation section label
    cv2.putText(
        canvas,
        "VIEW MODES",
        (14 if is_fhd else 12, header_h + (20 if is_fhd else 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38 if is_fhd else 0.34,
        COLOR_TEXT_TERTIARY,
        1,
        cv2.LINE_AA,
    )

    # Mode Buttons
    for mode_id, bx, by, bw, bh, name, sub in layout["buttons_rects"]:
        is_active = (mode_id == active_mode)
        bg = COLOR_CONTAINER_ACTIVE if is_active else COLOR_CONTAINER
        border = COLOR_BORDER_ACTIVE if is_active else COLOR_BORDER_SUBTLE
        draw_rounded_rect(canvas, bx, by, bw, bh, 10 if is_fhd else 8, fill_color=bg, border_color=border)

        # Active indicator vertical bar
        if is_active:
            bar_w = 4 if is_fhd else 3
            draw_rounded_rect(canvas, bx + 2, by + 4, bar_w, bh - 8, 1, fill_color=COLOR_PRIMARY_ACCENT)

        # Mode number pill
        badge_w, badge_h = (22, 22) if is_fhd else (16, 16)
        badge_x = bx + (12 if is_active else 10)
        badge_y = by + (bh - badge_h) // 2
        badge_bg = COLOR_PRIMARY_ACCENT if is_active else COLOR_CONTAINER_HIGH
        badge_fg = COLOR_SURFACE_DARK if is_active else COLOR_TEXT_SECONDARY
        draw_rounded_rect(canvas, badge_x, badge_y, badge_w, badge_h, 5 if is_fhd else 4, fill_color=badge_bg)
        badge_font = 0.40 if is_fhd else 0.34
        badge_offset_y = 16 if is_fhd else 12
        badge_offset_x = 6 if is_fhd else 4
        cv2.putText(
            canvas,
            str(mode_id),
            (badge_x + badge_offset_x, badge_y + badge_offset_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            badge_font,
            badge_fg,
            1,
            cv2.LINE_AA,
        )

        # Text labels
        tx = badge_x + badge_w + (10 if is_fhd else 6)
        if bh >= 42:
            name_color = COLOR_TEXT_PRIMARY if is_active else COLOR_TEXT_SECONDARY
            name_font = 0.46 if is_fhd else 0.38
            name_y = by + (22 if is_fhd else 16)
            cv2.putText(canvas, name, (tx, name_y), cv2.FONT_HERSHEY_SIMPLEX, name_font, name_color, 1, cv2.LINE_AA)
            sub_color = COLOR_PRIMARY_ACCENT if is_active else COLOR_TEXT_TERTIARY
            sub_font = 0.36 if is_fhd else 0.30
            sub_y = by + (40 if is_fhd else 30)
            cv2.putText(canvas, sub, (tx, sub_y), cv2.FONT_HERSHEY_SIMPLEX, sub_font, sub_color, 1, cv2.LINE_AA)
        else:
            name_color = COLOR_TEXT_PRIMARY if is_active else COLOR_TEXT_SECONDARY
            cv2.putText(canvas, name, (tx, by + (bh + 4) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, name_color, 1, cv2.LINE_AA)

    # Shortcuts Card at bottom of sidebar
    card_h = 48 if is_fhd else 38
    card_y = h - card_h - (10 if is_fhd else 6)
    card_x = 10 if is_fhd else 8
    card_w = sidebar_w - (20 if is_fhd else 16)
    draw_rounded_rect(canvas, card_x, card_y, card_w, card_h, 8 if is_fhd else 6, fill_color=COLOR_CONTAINER, border_color=COLOR_BORDER_SUBTLE)
    hint_font = 0.34 if is_fhd else 0.30
    cv2.putText(canvas, "[1-7] Mode   [D] HUD", (card_x + 8, card_y + (18 if is_fhd else 15)), cv2.FONT_HERSHEY_SIMPLEX, hint_font, COLOR_TEXT_TERTIARY, 1, cv2.LINE_AA)
    cv2.putText(canvas, "[F] Fullscr  [Q] Exit", (card_x + 8, card_y + (36 if is_fhd else 29)), cv2.FONT_HERSHEY_SIMPLEX, hint_font, COLOR_TEXT_TERTIARY, 1, cv2.LINE_AA)


# ============================================================================
# Central Viewport & Connection/Loading States
# ============================================================================

def fit_viewport(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Scale image cleanly to fit inside target bounds preserving aspect ratio."""
    ih, iw = image.shape[:2]
    if iw <= 0 or ih <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / iw, target_h / ih)
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = COLOR_SURFACE_DARK
    ox = (target_w - nw) // 2
    oy = (target_h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def draw_state_card(
    canvas: np.ndarray,
    vx: int,
    vy: int,
    vw: int,
    vh: int,
    title: str,
    subtitle: str,
    status_tag: str,
    status_color: tuple[int, int, int] = COLOR_STATUS_AMBER,
    is_fhd: bool = False,
) -> None:
    """Render a Material 3 centered card for waiting, loading, degraded, or empty states."""
    card_w = min(540, vw - 48) if is_fhd else min(420, vw - 32)
    card_h = 160 if is_fhd else 130
    cx = vx + (vw - card_w) // 2
    cy = vy + (vh - card_h) // 2

    # Translucent card background with strong outline
    draw_rounded_rect_alpha(
        canvas,
        cx,
        cy,
        card_w,
        card_h,
        14 if is_fhd else 12,
        fill_color=COLOR_CONTAINER,
        alpha=0.92,
        border_color=COLOR_BORDER_STRONG,
        border_thickness=1,
    )

    # Status tag pill
    draw_pill(
        canvas,
        cx + (20 if is_fhd else 16),
        cy + (20 if is_fhd else 16),
        status_tag,
        status_color,
        COLOR_CONTAINER_HIGH,
        dot_color=status_color,
        border_color=COLOR_BORDER_SUBTLE,
        font_scale=0.38 if is_fhd else 0.34,
        padding_x=10 if is_fhd else 8,
        padding_y=4 if is_fhd else 3,
    )

    # Title
    cv2.putText(
        canvas,
        title,
        (cx + (20 if is_fhd else 16), cy + (72 if is_fhd else 58)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58 if is_fhd else 0.50,
        COLOR_TEXT_PRIMARY,
        1,
        cv2.LINE_AA,
    )

    # Subtitle
    cv2.putText(
        canvas,
        subtitle,
        (cx + (20 if is_fhd else 16), cy + (100 if is_fhd else 80)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42 if is_fhd else 0.36,
        COLOR_TEXT_SECONDARY,
        1,
        cv2.LINE_AA,
    )

    # Rhythmic pulsing dot animation
    phase = int((time.monotonic() * 3) % 3)
    for i in range(3):
        dot_x = cx + (20 if is_fhd else 16) + i * (18 if is_fhd else 14)
        dot_y = cy + (130 if is_fhd else 104)
        dot_col = status_color if i == phase else COLOR_TEXT_TERTIARY
        cv2.circle(canvas, (dot_x, dot_y), 4 if is_fhd else 3, dot_col, -1, cv2.LINE_AA)


def draw_viewport_hud(
    canvas: np.ndarray,
    vx: int,
    vy: int,
    vw: int,
    vh: int,
    title: str,
    mode: int,
    snapshot: Any,
    waiting: str | None,
    is_fhd: bool = False,
) -> None:
    """Render top-left mode tag and bottom contextual diagnostics inside the viewport."""
    # Top-Left View Title Pill
    title_text = f"M{mode} — {title.split('(')[0].strip()}"
    dot_color = COLOR_STATUS_GREEN if waiting is None else COLOR_STATUS_AMBER
    draw_pill(
        canvas,
        vx + (16 if is_fhd else 12),
        vy + (16 if is_fhd else 12),
        title_text,
        COLOR_TEXT_PRIMARY,
        COLOR_CONTAINER,
        dot_color=dot_color,
        border_color=COLOR_BORDER_STRONG,
        font_scale=0.46 if is_fhd else 0.38,
        padding_x=12 if is_fhd else 10,
        padding_y=5 if is_fhd else 4,
    )

    # Bottom Contextual Diagnostic Bar
    metrics = snapshot.metrics if snapshot is not None else None
    bottom_y = vy + vh - (40 if is_fhd else 32)
    b_font = 0.40 if is_fhd else 0.34
    b_pad_x = 10 if is_fhd else 8
    b_pad_y = 4 if is_fhd else 3
    if mode == 1:
        fid = snapshot.rgb_capture_id if snapshot else 0
        desc = f"Sensor Frame #{fid} | Cadence {(metrics.capture_hz or 0):.1f} Hz | Overwritten {metrics.overwritten_before_consumption if metrics else 0}"
        draw_pill(canvas, vx + (16 if is_fhd else 12), bottom_y, desc, COLOR_TEXT_SECONDARY, COLOR_CONTAINER, border_color=COLOR_BORDER_SUBTLE, font_scale=b_font, padding_x=b_pad_x, padding_y=b_pad_y)
    elif mode == 2:
        desc = "Scale: Warm Near (Yellow/Red) -> Cool Far (Blue/Purple) | Mariem CUDA Depth"
        draw_pill(canvas, vx + (16 if is_fhd else 12), bottom_y, desc, COLOR_STATUS_CYAN, COLOR_CONTAINER, border_color=COLOR_BORDER_SUBTLE, font_scale=b_font, padding_x=b_pad_x, padding_y=b_pad_y)
    elif mode == 3:
        desc = "Normals: +X Right (Red) | +Y Down (Green) | +Z Forward (Blue) | Multiscale R=1..4"
        draw_pill(canvas, vx + (16 if is_fhd else 12), bottom_y, desc, COLOR_STATUS_PURPLE, COLOR_CONTAINER, border_color=COLOR_BORDER_SUBTLE, font_scale=b_font, padding_x=b_pad_x, padding_y=b_pad_y)
    elif mode == 4:
        desc = "Confidence: Green = Stable Geometry | Dark/Red = Inconsistent / Motion"
        draw_pill(canvas, vx + (16 if is_fhd else 12), bottom_y, desc, COLOR_STATUS_GREEN, COLOR_CONTAINER, border_color=COLOR_BORDER_SUBTLE, font_scale=b_font, padding_x=b_pad_x, padding_y=b_pad_y)
    elif mode == 5:
        count = len(snapshot.hand_state.hands) if (snapshot and snapshot.hand_state) else 0
        desc = f"Talel Tracker: {count} hands tracked | MediaPipe landmarks (21 pts) + Optical Flow"
        draw_pill(canvas, vx + (16 if is_fhd else 12), bottom_y, desc, COLOR_PRIMARY_ACCENT, COLOR_CONTAINER, border_color=COLOR_BORDER_SUBTLE, font_scale=b_font, padding_x=b_pad_x, padding_y=b_pad_y)
    elif mode == 6:
        desc = "P123 XYZ Contract: Camera-relative metric coordinates (X right, Y down, Z forward in meters)"
        draw_pill(canvas, vx + (16 if is_fhd else 12), bottom_y, desc, COLOR_STATUS_GREEN, COLOR_CONTAINER, border_color=COLOR_BORDER_SUBTLE, font_scale=b_font, padding_x=b_pad_x, padding_y=b_pad_y)
    elif mode == 7:
        count = len(snapshot.hand_state.hands) if (snapshot and snapshot.hand_state) else 0
        desc = f"Hand-held 3D light: {count} tracked | screen-space ray shadows + volumetric scattering"
        draw_pill(canvas, vx + (16 if is_fhd else 12), bottom_y, desc, COLOR_STATUS_CYAN, COLOR_CONTAINER, border_color=COLOR_BORDER_SUBTLE, font_scale=b_font, padding_x=b_pad_x, padding_y=b_pad_y)


def draw_debug_overlay(
    canvas: np.ndarray,
    vx: int,
    vy: int,
    vw: int,
    vh: int,
    snapshot: Any,
    display_fps: float | None = None,
    is_fhd: bool = False,
) -> None:
    """Floating telemetry card toggled with 'D'."""
    metrics = snapshot.metrics if snapshot is not None else None
    if metrics is None:
        return

    card_w = min(560, vw - 32) if is_fhd else min(460, vw - 24)
    card_h = 190 if is_fhd else 160
    cx = vx + (16 if is_fhd else 12)
    cy = vy + (54 if is_fhd else 44)

    draw_rounded_rect_alpha(
        canvas,
        cx,
        cy,
        card_w,
        card_h,
        12 if is_fhd else 10,
        fill_color=COLOR_CONTAINER_LOW,
        alpha=0.92,
        border_color=COLOR_BORDER_ACTIVE,
        border_thickness=1,
    )

    title_scale = 0.46 if is_fhd else 0.40
    cv2.putText(canvas, "ENGINE PIPELINE TELEMETRY [DEBUG HUD]", (cx + 14, cy + (26 if is_fhd else 20)), cv2.FONT_HERSHEY_SIMPLEX, title_scale, COLOR_PRIMARY_ACCENT, 1, cv2.LINE_AA)
    lines = [
        f"Camera Capture:  {metrics.capture_hz or 0:.1f} Hz | Overwrites: {metrics.overwritten_before_consumption} | Total: {metrics.captured}",
        f"Display UI:      {display_fps or 0:.1f} FPS (decoupled render loop)",
        f"CUDA Depth:      {metrics.depth_hz or 0:.1f} Hz | Latency p95: {metrics.depth_age_p95_ms or 0:.1f} ms | Errors: {metrics.depth_errors}",
        f"CUDA Normals:    {metrics.normal_hz or 0:.1f} Hz | Latency p95: {metrics.normal_age_p95_ms or 0:.1f} ms",
        f"Temporal Conf:   {metrics.temporal_hz or 0:.1f} Hz | Geometry Age: {metrics.geometry_age_p95_ms or 0:.1f} ms",
        f"Hand Tracking:   {metrics.hand_hz or 0:.1f} Hz | Latency p95: {metrics.hand_age_p95_ms or 0:.1f} ms",
        f"XYZ Projection:  {metrics.xyz_hz or 0:.1f} Hz | Latency p95: {metrics.xyz_age_p95_ms or 0:.1f} ms",
    ]
    line_font = 0.36 if is_fhd else 0.32
    step_y = 20 if is_fhd else 16
    start_y = 52 if is_fhd else 40
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (cx + 14, cy + start_y + i * step_y), cv2.FONT_HERSHEY_SIMPLEX, line_font, COLOR_TEXT_SECONDARY, 1, cv2.LINE_AA)


# ============================================================================
# Main Canvas Assembler
# ============================================================================

def finish(
    image: np.ndarray | None,
    title: str,
    waiting: str | None,
    snapshot: Any,
    mode: int,
    display_size: tuple[int, int] | None = None,
    display_fps: float | None = None,
    show_debug: bool = False,
) -> np.ndarray:
    """Compose the modern Google Material 3 UI layout for the selected view."""
    # Determine target canvas size
    if display_size is not None:
        target_w, target_h = display_size
    elif image is not None:
        target_w = max(640, image.shape[1] + 200)
        target_h = max(480, image.shape[0] + 50)
    else:
        target_w, target_h = 960, 600

    layout = compute_layout((target_w, target_h))

    # Base Canvas Surface
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = COLOR_SURFACE_DARK

    vx, vy = layout["vx"], layout["vy"]
    vw, vh = layout["vw"], layout["vh"]

    # Render Viewport Content
    if image is not None:
        fitted = fit_viewport(image, vw, vh)
        canvas[vy:vy + vh, vx:vx + vw] = fitted
    else:
        # Camera unavailable / initial connection state
        canvas[vy:vy + vh, vx:vx + vw] = COLOR_CONTAINER_LOW

    # Viewport border outline
    draw_rounded_rect(canvas, vx, vy, vw, vh, 8, fill_color=None, border_color=COLOR_BORDER_SUBTLE, border_thickness=1)

    is_fhd = layout.get("is_fhd", False)

    # Handle Empty/Waiting/Loading States
    if waiting:
        if "camera" in waiting.lower():
            draw_state_card(
                canvas,
                vx,
                vy,
                vw,
                vh,
                "Camera Initializing",
                "Connecting to physical video device (/dev/video0)...",
                "CONNECTING",
                COLOR_STATUS_AMBER,
                is_fhd=is_fhd,
            )
        elif "depth" in waiting.lower():
            draw_state_card(
                canvas,
                vx,
                vy,
                vw,
                vh,
                "Mariem CUDA Depth Pending",
                "Warming up TensorRT / CUDA depth estimation worker...",
                "INITIALIZING",
                COLOR_STATUS_CYAN,
                is_fhd=is_fhd,
            )
        elif "geometry" in waiting.lower():
            draw_state_card(
                canvas,
                vx,
                vy,
                vw,
                vh,
                "Surface Normals Pending",
                "Awaiting valid depth frames for CUDA normals calculation...",
                "CALCULATING",
                COLOR_STATUS_PURPLE,
                is_fhd=is_fhd,
            )
        elif "temporal" in waiting.lower():
            draw_state_card(
                canvas,
                vx,
                vy,
                vw,
                vh,
                "Temporal Consistency Stabilizing",
                "Accumulating multi-frame depth confidence baseline...",
                "STABILIZING",
                COLOR_STATUS_GREEN,
                is_fhd=is_fhd,
            )
        else:
            draw_state_card(
                canvas,
                vx,
                vy,
                vw,
                vh,
                title,
                waiting,
                "STATUS",
                COLOR_STATUS_AMBER,
                is_fhd=is_fhd,
            )

    # In-viewport HUD overlays
    draw_viewport_hud(canvas, vx, vy, vw, vh, title, mode, snapshot, waiting, is_fhd=is_fhd)

    # Draw Header & Sidebar
    draw_header(canvas, snapshot, layout, display_fps=display_fps)
    draw_sidebar(canvas, mode, layout)

    # Debug HUD Overlay if enabled
    if show_debug:
        draw_debug_overlay(canvas, vx, vy, vw, vh, snapshot, display_fps=display_fps, is_fhd=is_fhd)

    return canvas
