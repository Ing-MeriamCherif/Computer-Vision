import cv2
import numpy as np

import p123.views.relight as relight
from p123.views.common import compute_layout


def test_mode7_controls_drag_power_and_range_and_cycle_color():
    renderer = relight._renderer
    previous = (renderer.light_intensity, renderer.light_range_m, renderer.light_color_index)
    display_size = (1280, 720)
    layout = compute_layout(display_size)
    controls = relight._control_layout((layout["vx"], layout["vy"], layout["vw"], layout["vh"]))
    try:
        intensity = controls["intensity"]
        assert relight.handle_control_mouse(cv2.EVENT_LBUTTONDOWN, intensity[0] + intensity[2], intensity[1], 0, display_size)
        assert relight.handle_control_mouse(cv2.EVENT_LBUTTONUP, intensity[0] + intensity[2], intensity[1], 0, display_size)
        assert renderer.light_intensity == relight.MAX_INTENSITY

        range_track = controls["range"]
        relight.handle_control_mouse(cv2.EVENT_LBUTTONDOWN, range_track[0], range_track[1], 0, display_size)
        relight.handle_control_mouse(cv2.EVENT_LBUTTONUP, range_track[0], range_track[1], 0, display_size)
        assert renderer.light_range_m == relight.MIN_RANGE_M

        color = controls["color"]
        relight.handle_control_mouse(cv2.EVENT_LBUTTONDOWN, color[0] + 4, color[1] + 4, 0, display_size)
        relight.handle_control_mouse(cv2.EVENT_LBUTTONUP, color[0] + 4, color[1] + 4, 0, display_size)
        assert renderer.light_color_index == 1

        light = relight.LightState(position_camera=np.array([0.0, 0.0, 1.0], dtype=np.float32))
        controlled = renderer.controlled_lights([light])[0]
        assert controlled.intensity == relight.MAX_INTENSITY
        assert controlled.range_m == relight.MIN_RANGE_M
        np.testing.assert_allclose(controlled.color_rgb, relight.COLOR_PRESETS[1][1])
    finally:
        renderer.light_intensity, renderer.light_range_m, renderer.light_color_index = previous
        relight._active_control = None


def test_mode7_control_bar_draws_inside_viewport():
    layout = compute_layout((1280, 720))
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    relight.draw_controls(canvas, (layout["vx"], layout["vy"], layout["vw"], layout["vh"]))
    panel = relight._control_layout((layout["vx"], layout["vy"], layout["vw"], layout["vh"]))["panel"]
    x, y, width, height = panel
    assert canvas[y:y + height, x:x + width].any()
