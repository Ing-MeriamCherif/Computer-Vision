"""Tests for configurable single-light rendering constants."""

import unittest

from renderer.config import LightingConfig


class LightingConfigTests(unittest.TestCase):
    def test_plan_defaults_are_used(self) -> None:
        config = LightingConfig()
        self.assertEqual(config.ambient_strength, 0.15)
        self.assertEqual(config.specular_strength, 0.20)
        self.assertEqual(config.shininess, 48.0)
        self.assertEqual(config.attenuation_k, 0.6)

    def test_rejects_negative_or_nonfinite_parameters(self) -> None:
        for field in ("ambient_strength", "specular_strength", "shininess", "attenuation_k"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                LightingConfig(**{field: float("nan")})
        with self.assertRaises(ValueError):
            LightingConfig(specular_strength=-0.1)
        with self.assertRaises(ValueError):
            LightingConfig(shininess=0.0)


if __name__ == "__main__":
    unittest.main()
