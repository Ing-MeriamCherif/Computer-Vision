"""Validation for configurable screen-space shadow controls."""

import unittest

from renderer.config import ShadowConfig


class ShadowConfigTests(unittest.TestCase):
    def test_initial_values_match_shadow_phase_defaults(self) -> None:
        config = ShadowConfig()
        self.assertTrue(config.shadow_enabled)
        self.assertEqual(config.shadow_resolution_scale, 0.5)
        self.assertEqual(config.shadow_steps, 12)
        self.assertEqual(config.shadow_bias_m, 0.015)
        self.assertEqual(config.shadow_thickness_m, 0.15)
        self.assertEqual(config.ray_start_offset, 0.01)

    def test_rejects_invalid_parameters(self) -> None:
        invalid = (
            {"shadow_resolution_scale": 0.0},
            {"shadow_resolution_scale": 1.1},
            {"shadow_steps": 0},
            {"shadow_steps": 65},
            {"shadow_steps": 1.5},
            {"shadow_bias_m": -0.01},
            {"shadow_thickness_m": 0.01},
            {"ray_start_offset": float("nan")},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                ShadowConfig(**values)


if __name__ == "__main__":
    unittest.main()
