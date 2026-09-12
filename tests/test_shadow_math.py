"""CPU-reference tests for screen-space shadow rays and synthetic geometry."""

import unittest

import numpy as np

from renderer.shadow_math import (
    compute_depth_edge_mask,
    filter_shadow_samples_reference,
    sample_is_occluded,
    shadow_soft_offsets,
    trace_shadow_visibility,
)
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

    def test_scene_b_c_and_d_are_deterministic_depth_cases(self) -> None:
        scene_b = make_synthetic_packet(160, 120, scene_name="B")
        scene_c = make_synthetic_packet(160, 120, scene_name="C")
        scene_d = make_synthetic_packet(160, 120, scene_name="D")
        np.testing.assert_array_equal(
            scene_b.depth.depth_m,
            make_synthetic_packet(160, 120, scene_name="B").depth.depth_m,
        )
        np.testing.assert_allclose(np.unique(scene_b.depth.depth_m), (1.4, 2.1, 3.0))
        np.testing.assert_allclose(np.unique(scene_c.depth.depth_m), (1.5, 3.0))
        np.testing.assert_allclose(np.unique(scene_d.depth.depth_m), (1.25, 3.0))
        self.assertEqual(int(np.count_nonzero(scene_c.depth.depth_m == 1.5)), 5 * 91)
        self.assertFalse(make_synthetic_packet(160, 120, scene_name="D", include_occluders=False).depth.depth_m.min() < 3.0)

    def test_depth_edges_mark_both_sides_and_ignore_invalid_samples(self) -> None:
        depth = np.full((3, 5), 3.0, dtype=np.float32)
        depth[:, 2:] = 1.5
        valid = np.ones(depth.shape, dtype=np.bool_)
        edge_mask = compute_depth_edge_mask(depth, valid, threshold_m=0.2)
        self.assertTrue(np.all(edge_mask[:, 1:3]))
        self.assertFalse(np.any(edge_mask[:, 0]))
        valid[:, 2] = False
        edge_mask = compute_depth_edge_mask(depth, valid, threshold_m=0.2)
        self.assertFalse(np.any(edge_mask[:, 1:3]))

    def test_depth_edge_mask_does_not_mark_a_smooth_plane(self) -> None:
        depth = np.full((8, 8), 2.0, dtype=np.float32)
        valid = np.ones(depth.shape, dtype=np.bool_)
        self.assertFalse(np.any(compute_depth_edge_mask(depth, valid, threshold_m=0.2)))

    def test_depth_aware_upsampling_rejects_cross_surface_shadow_bleeding(self) -> None:
        visibility = np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float32)
        tap_depth = np.asarray((2.5, 2.5, 2.0, 2.0), dtype=np.float32)
        aware = filter_shadow_samples_reference(
            visibility,
            tap_depth,
            receiver_depth_m=2.0,
            depth_edge_threshold_m=0.2,
            edge_aware=True,
        )
        standard = filter_shadow_samples_reference(
            visibility,
            tap_depth,
            receiver_depth_m=2.0,
            depth_edge_threshold_m=0.2,
            edge_aware=False,
        )
        self.assertEqual(aware, 1.0)
        self.assertEqual(standard, 0.5)

    def test_unknown_depth_during_upsampling_falls_back_to_lit(self) -> None:
        visibility = np.asarray((0.0, 0.0), dtype=np.float32)
        unknown_depth = np.asarray((0.0, np.nan), dtype=np.float32)
        self.assertEqual(
            filter_shadow_samples_reference(
                visibility,
                unknown_depth,
                receiver_depth_m=2.0,
                depth_edge_threshold_m=0.2,
            ),
            1.0,
        )

    def test_soft_shadow_taps_are_symmetric_and_preserve_center(self) -> None:
        for count in (4, 8):
            offsets = shadow_soft_offsets(count, 1.25)
            self.assertEqual(offsets.shape, (count, 2))
            np.testing.assert_allclose(offsets.mean(axis=0), (0.0, 0.0), atol=1e-7)
            self.assertAlmostEqual(float(np.max(np.linalg.norm(offsets, axis=1))), 1.25, places=6)

    def test_light_x_y_z_sweep_preserves_expected_shadow_direction(self) -> None:
        right_light_visibilities = [self.trace(30, 60, (x, 0.0, 0.75)) for x in (-0.6, 0.0, 0.6)]
        self.assertEqual(right_light_visibilities, [1.0, 1.0, 0.0])
        down_light_visibilities = [self.trace(80, y, (0.0, 0.6, 0.75)) for y in (15, 60, 105)]
        self.assertEqual(down_light_visibilities, [0.0, 1.0, 1.0])
        z_visibilities = [self.trace(30, 60, (0.6, 0.0, z)) for z in (0.75, 1.0, 1.3)]
        self.assertEqual(z_visibilities, [0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
