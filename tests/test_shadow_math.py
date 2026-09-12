"""CPU-reference tests for screen-space shadow rays and synthetic geometry."""

import unittest

import numpy as np

from renderer.shadow_math import sample_is_occluded, trace_shadow_visibility
from tools.renderer_synthetic_demo import make_synthetic_packet


class ShadowMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.packet = make_synthetic_packet(160, 120)
        depth = self.packet.depth
        self.intrinsics = {
            "fx": depth.fx,
            "fy": depth.fy,
            "cx": depth.cx,
            "cy": depth.cy,
            "shadow_steps": 12,
            "shadow_bias_m": 0.015,
            "shadow_thickness_m": 0.15,
            "ray_start_offset": 0.01,
        }

    def trace(self, x: int, y: int, light: tuple[float, float, float], packet=None) -> float:
        current = packet or self.packet
        return trace_shadow_visibility(
            current.depth.depth_m,
            current.depth.valid_mask,
            x,
            y,
            np.asarray(light, dtype=np.float32),
            **self.intrinsics,
        )

    def test_visibility_depth_test_uses_bias_and_thickness(self) -> None:
        self.assertFalse(sample_is_occluded(1.0, True, 1.01, shadow_bias_m=0.015, shadow_thickness_m=0.15))
        self.assertTrue(sample_is_occluded(1.0, True, 1.05, shadow_bias_m=0.015, shadow_thickness_m=0.15))
        self.assertFalse(sample_is_occluded(1.0, True, 1.20, shadow_bias_m=0.015, shadow_thickness_m=0.15))

    def test_invalid_depth_is_conservatively_unblocked(self) -> None:
        for invalid_depth in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(depth=invalid_depth):
                self.assertFalse(
                    sample_is_occluded(
                        invalid_depth,
                        True,
                        1.05,
                        shadow_bias_m=0.015,
                        shadow_thickness_m=0.15,
                    )
                )
        self.assertFalse(sample_is_occluded(1.0, False, 1.05, shadow_bias_m=0.015, shadow_thickness_m=0.15))

    def test_ray_projection_outside_camera_is_unblocked(self) -> None:
        visibility = self.trace(30, 60, (20.0, 0.0, 0.75))
        self.assertEqual(visibility, 1.0)

    def test_light_x_moves_shadow_in_opposite_direction(self) -> None:
        # The foreground occupies the center. A light to the right shadows the
        # left background; a light to the left shadows the right background.
        self.assertEqual(self.trace(30, 60, (0.6, 0.0, 0.75)), 0.0)
        self.assertEqual(self.trace(30, 60, (-0.6, 0.0, 0.75)), 1.0)
        self.assertEqual(self.trace(130, 60, (-0.6, 0.0, 0.75)), 0.0)
        self.assertEqual(self.trace(130, 60, (0.6, 0.0, 0.75)), 1.0)

    def test_light_y_moves_shadow_opposite_camera_y_direction(self) -> None:
        # +Y is down; the corresponding shadow shifts upward on the image.
        self.assertEqual(self.trace(80, 15, (0.0, 0.6, 0.75)), 0.0)
        self.assertEqual(self.trace(80, 15, (0.0, -0.6, 0.75)), 1.0)
        self.assertEqual(self.trace(80, 105, (0.0, -0.6, 0.75)), 0.0)
        self.assertEqual(self.trace(80, 105, (0.0, 0.6, 0.75)), 1.0)

    def test_light_z_changes_shadow_projection(self) -> None:
        self.assertEqual(self.trace(30, 60, (0.6, 0.0, 0.75)), 0.0)
        self.assertEqual(self.trace(30, 60, (0.6, 0.0, 1.3)), 1.0)

    def test_removing_or_moving_occluder_removes_shadow(self) -> None:
        no_occluder = make_synthetic_packet(160, 120, foreground_bounds=None)
        moved_occluder = make_synthetic_packet(
            160,
            120,
            foreground_bounds=(0.65, 0.25, 0.95, 0.75),
        )
        light = (0.6, 0.0, 0.75)
        self.assertEqual(self.trace(30, 60, light), 0.0)
        self.assertEqual(self.trace(30, 60, light, no_occluder), 1.0)
        self.assertEqual(self.trace(30, 60, light, moved_occluder), 1.0)

    def test_invalid_occluder_depth_does_not_make_random_shadow(self) -> None:
        depth = self.packet.depth.depth_m.copy()
        valid = self.packet.depth.valid_mask.copy()
        occluder = depth == np.float32(1.5)
        valid[occluder] = False
        depth[occluder] = 0.0
        visibility = trace_shadow_visibility(
            depth,
            valid,
            30,
            60,
            np.asarray((0.6, 0.0, 0.75), dtype=np.float32),
            **self.intrinsics,
        )
        self.assertEqual(visibility, 1.0)


if __name__ == "__main__":
    unittest.main()
