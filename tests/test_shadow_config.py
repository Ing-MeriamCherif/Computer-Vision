"""Validation for configurable screen-space shadow controls."""

import unittest

from renderer.config import ShadowConfig, ShadowQualityProfile


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

    def test_quality_profiles_match_their_documented_tradeoffs(self) -> None:
        baseline = ShadowConfig.for_profile(ShadowQualityProfile.BASELINE)
        safe = ShadowConfig.for_profile("safe")
        balanced = ShadowConfig.for_profile("balanced")
        high = ShadowConfig.for_profile("high")

        self.assertEqual((baseline.shadow_resolution_scale, baseline.shadow_steps), (0.5, 12))
        self.assertFalse(baseline.shadow_softening_enabled)
        self.assertFalse(baseline.shadow_edge_aware_upsampling)
        self.assertEqual((safe.shadow_resolution_scale, safe.shadow_steps), (0.5, 8))
        self.assertFalse(safe.shadow_softening_enabled)
        self.assertTrue(safe.shadow_edge_aware_upsampling)
        self.assertEqual((balanced.shadow_resolution_scale, balanced.shadow_steps), (0.5, 16))
        self.assertTrue(balanced.shadow_softening_enabled)
        self.assertEqual(balanced.shadow_soft_samples, 4)
        self.assertTrue(balanced.shadow_edge_aware_upsampling)
        self.assertEqual((high.shadow_resolution_scale, high.shadow_steps), (0.5, 20))
        self.assertTrue(high.shadow_softening_enabled)
        self.assertEqual(high.shadow_soft_samples, 8)
        self.assertTrue(high.shadow_edge_aware_upsampling)

    def test_rejects_invalid_soft_shadow_settings(self) -> None:
        for values in (
            {"shadow_soft_samples": 5},
            {"shadow_soft_radius": 0.0, "shadow_softening_enabled": True},
            {"depth_edge_threshold_m": 0.0},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ShadowConfig(**values)


if __name__ == "__main__":
    unittest.main()
