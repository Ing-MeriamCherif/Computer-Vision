"""Configuration and deterministic projection/occlusion tests for light orbs."""

import unittest

import numpy as np

from renderer.config import LightOrbConfig
from renderer.orb_math import evaluate_light_orb


class LightOrbTests(unittest.TestCase):
    def setUp(self) -> None:
        self.depth = np.full((80, 100), 3.0, dtype=np.float32)
        self.valid = np.ones_like(self.depth, dtype=np.bool_)
        self.fx = 100.0
        self.fy = 100.0
        self.cx = 49.5
        self.cy = 39.5

    def evaluate(self, position, *, depth=None, valid=None, active=True, bias=0.02):
        return evaluate_light_orb(
            np.asarray(position, dtype=np.float32),
            self.depth if depth is None else depth,
            self.valid if valid is None else valid,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            occlusion_bias_m=bias,
            active=active,
        )

    def test_config_defaults_and_invalid_values(self) -> None:
        config = LightOrbConfig()
        self.assertFalse(config.light_orb_enabled)
        self.assertGreater(config.light_orb_radius_m, 0.0)
        for values in (
            {"light_orb_radius_m": 0.0},
            {"light_orb_radius_m": float("inf")},
            {"light_orb_intensity": -1.0},
            {"light_orb_halo_strength": float("nan")},
            {"light_orb_occlusion_bias_m": -0.01},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                LightOrbConfig(**values)

    def test_projection_uses_camera_intrinsics(self) -> None:
        projected = self.evaluate((0.2, -0.1, 2.0))
        self.assertAlmostEqual(projected.u_px, 59.5, places=5)
        self.assertAlmostEqual(projected.v_px, 34.5, places=5)
        self.assertTrue(projected.in_frame)

    def test_rejects_nonpositive_or_invalid_z(self) -> None:
        for z in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(z=z):
                projected = self.evaluate((0.0, 0.0, z))
                self.assertFalse(projected.visible)
                self.assertIn(projected.reason, ("behind-camera", "invalid"))

    def test_offscreen_light_is_not_drawn(self) -> None:
        projected = self.evaluate((100.0, 0.0, 1.0))
        self.assertFalse(projected.in_frame)
        self.assertFalse(projected.visible)
        self.assertEqual(projected.reason, "off-screen")

    def test_light_in_front_of_foreground_object_is_visible(self) -> None:
        depth = self.depth.copy()
        depth[30:50, 40:60] = 1.5
        projected = self.evaluate((0.0, 0.0, 1.0), depth=depth)
        self.assertTrue(projected.visible)
        self.assertFalse(projected.occluded)
        self.assertEqual(projected.scene_z_m, 1.5)

    def test_light_behind_foreground_object_is_occluded(self) -> None:
        depth = self.depth.copy()
        depth[30:50, 40:60] = 1.5
        projected = self.evaluate((0.0, 0.0, 2.0), depth=depth)
        self.assertFalse(projected.visible)
        self.assertTrue(projected.occluded)
        self.assertEqual(projected.reason, "occluded")

    def test_light_behind_background_plane_is_occluded(self) -> None:
        projected = self.evaluate((0.0, 0.0, 3.5))
        self.assertFalse(projected.visible)
        self.assertTrue(projected.occluded)
        self.assertEqual(projected.scene_z_m, 3.0)

    def test_moving_light_across_depth_order_changes_visibility(self) -> None:
        depth = self.depth.copy()
        depth[30:50, 40:60] = 1.5
        in_front = self.evaluate((0.0, 0.0, 1.4), depth=depth)
        behind = self.evaluate((0.0, 0.0, 1.7), depth=depth)
        self.assertTrue(in_front.visible)
        self.assertFalse(behind.visible)

    def test_two_lights_have_independent_depth_visibility(self) -> None:
        depth = self.depth.copy()
        depth[30:50, 20:40] = 1.5
        visible = self.evaluate((-0.2, 0.0, 1.0), depth=depth)
        occluded = self.evaluate((0.2, 0.0, 4.0), depth=depth)
        self.assertTrue(visible.visible)
        self.assertFalse(occluded.visible)

    def test_inactive_light_has_no_orb(self) -> None:
        projected = self.evaluate((0.0, 0.0, 1.0), active=False)
        self.assertFalse(projected.visible)
        self.assertEqual(projected.reason, "inactive")

    def test_invalid_depth_is_conservatively_visible_and_output_is_finite(self) -> None:
        depth = self.depth.copy()
        valid = self.valid.copy()
        depth[39:41, 49:51] = np.nan
        valid[39:41, 49:51] = False
        projected = self.evaluate((0.0, 0.0, 1.0), depth=depth, valid=valid)
        self.assertTrue(projected.visible)
        self.assertFalse(projected.depth_valid)
        for value in (projected.u_px, projected.v_px, projected.light_z_m):
            self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()
