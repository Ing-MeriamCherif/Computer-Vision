"""Configuration and deterministic reference tests for volumetric scattering."""

import unittest

import numpy as np

from contracts.render_types import Light
from renderer.config import VolumetricConfig, VolumetricQualityProfile
from renderer.volumetric_math import (
    composite_scattering_reference,
    reference_scattering_for_ray,
)


def make_light(position, color, *, active=True, intensity=1.0) -> Light:
    return Light(
        position_camera_m=np.asarray(position, dtype=np.float32),
        color_rgb=np.asarray(color, dtype=np.float32),
        intensity=intensity,
        active=active,
    )


class VolumetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enabled = VolumetricConfig(volumetric_enabled=True)

    def test_quality_profiles_have_documented_resolution_and_samples(self) -> None:
        safe = VolumetricConfig.for_profile(VolumetricQualityProfile.SAFE)
        balanced = VolumetricConfig.for_profile("balanced")
        high = VolumetricConfig.for_profile("high")
        self.assertEqual((safe.volumetric_resolution_scale, safe.volumetric_samples), (0.25, 8))
        self.assertEqual((balanced.volumetric_resolution_scale, balanced.volumetric_samples), (0.25, 12))
        self.assertEqual((high.volumetric_resolution_scale, high.volumetric_samples), (0.5, 20))
        self.assertFalse(balanced.volumetric_enabled)

    def test_invalid_configuration_values_are_rejected(self) -> None:
        invalid = (
            {"volumetric_resolution_scale": 0.0},
            {"volumetric_samples": 0},
            {"volumetric_samples": 25},
            {"volumetric_samples": 1.5},
            {"volumetric_density": -0.1},
            {"volumetric_intensity": float("inf")},
            {"volumetric_decay": 1.01},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                VolumetricConfig(**values)

    def test_disabled_composition_is_exact_surface_passthrough(self) -> None:
        surface = np.asarray((0.12, 0.47, 0.92), dtype=np.float32)
        result = composite_scattering_reference(
            surface,
            np.asarray((100.0, 100.0, 100.0), dtype=np.float32),
            enabled=False,
        )
        np.testing.assert_array_equal(result, surface)

    def test_invalid_depth_safely_contributes_no_scattering(self) -> None:
        light = make_light((0.0, 0.0, 1.0), (1.0, 0.7, 0.45))
        for depth in (0.0, -1.0, float("nan"), float("inf")):
            result = reference_scattering_for_ray(
                np.asarray((0.0, 0.0, 1.0)), depth, True, [light], self.enabled
            )
            np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float32))
        invalid_mask_result = reference_scattering_for_ray(
            np.asarray((0.0, 0.0, 1.0)), 3.0, False, [light], self.enabled
        )
        np.testing.assert_array_equal(invalid_mask_result, np.zeros(3, dtype=np.float32))

    def test_one_and_two_light_scattering_accumulate_independently(self) -> None:
        warm = make_light((-0.2, 0.0, 1.0), (1.0, 0.70, 0.45))
        cool = make_light((0.2, 0.0, 1.0), (0.45, 0.70, 1.0))
        ray = np.asarray((0.0, 0.0, 1.0))
        warm_only = reference_scattering_for_ray(ray, 3.0, True, [warm], self.enabled)
        cool_only = reference_scattering_for_ray(ray, 3.0, True, [cool], self.enabled)
        both = reference_scattering_for_ray(ray, 3.0, True, [warm, cool], self.enabled)
        np.testing.assert_allclose(both, warm_only + cool_only, rtol=1e-6, atol=1e-7)
        self.assertGreater(warm_only[0], warm_only[2])
        self.assertGreater(cool_only[2], cool_only[0])
        dim_warm = make_light((-0.2, 0.0, 1.0), (1.0, 0.70, 0.45), intensity=0.25)
        dim_value = reference_scattering_for_ray(ray, 3.0, True, [dim_warm], self.enabled)
        np.testing.assert_allclose(dim_value, warm_only * 0.25, rtol=1e-6, atol=1e-7)

        cool.active = False
        with_inactive_cool = reference_scattering_for_ray(ray, 3.0, True, [warm, cool], self.enabled)
        np.testing.assert_array_equal(with_inactive_cool, warm_only)

    def test_scattering_moves_with_light_xyz_and_changes_with_color(self) -> None:
        center_ray = np.asarray((0.0, 0.0, 1.0))
        center_light = make_light((0.0, 0.0, 1.0), (1.0, 0.7, 0.45))
        moved_x = make_light((0.45, 0.0, 1.0), (1.0, 0.7, 0.45))
        moved_y = make_light((0.0, 0.45, 1.0), (1.0, 0.7, 0.45))
        moved_z = make_light((0.0, 0.0, 2.2), (1.0, 0.7, 0.45))
        center_value = reference_scattering_for_ray(center_ray, 3.0, True, [center_light], self.enabled)
        x_value = reference_scattering_for_ray(center_ray, 3.0, True, [moved_x], self.enabled)
        y_value = reference_scattering_for_ray(center_ray, 3.0, True, [moved_y], self.enabled)
        z_value = reference_scattering_for_ray(center_ray, 3.0, True, [moved_z], self.enabled)
        self.assertFalse(np.allclose(center_value, x_value))
        self.assertFalse(np.allclose(center_value, y_value))
        self.assertFalse(np.allclose(center_value, z_value))

        blue = make_light((0.0, 0.0, 1.0), (0.45, 0.70, 1.0))
        blue_value = reference_scattering_for_ray(center_ray, 3.0, True, [blue], self.enabled)
        self.assertGreater(blue_value[2], blue_value[0])

    def test_foreground_depth_terminates_ray_integration(self) -> None:
        light_behind_front_object = make_light((0.0, 0.0, 2.0), (1.0, 0.7, 0.45))
        ray = np.asarray((0.0, 0.0, 1.0))
        foreground = reference_scattering_for_ray(
            ray, 1.5, True, [light_behind_front_object], self.enabled
        )
        background = reference_scattering_for_ray(
            ray, 3.0, True, [light_behind_front_object], self.enabled
        )
        self.assertGreater(float(background.sum()), float(foreground.sum()))

    def test_scattering_and_composition_remain_finite_and_below_white(self) -> None:
        config = VolumetricConfig(
            volumetric_enabled=True,
            volumetric_density=100.0,
            volumetric_intensity=100.0,
            volumetric_samples=24,
        )
        intense = make_light((0.0, 0.0, 0.75), (1.0, 1.0, 1.0), intensity=10.0)
        volume = reference_scattering_for_ray(
            np.asarray((0.0, 0.0, 1.0)), 3.0, True, [intense], config
        )
        self.assertTrue(np.isfinite(volume).all())
        combined = composite_scattering_reference(
            np.asarray((0.98, 0.5, 0.1), dtype=np.float32), volume, enabled=True
        )
        self.assertTrue(np.isfinite(combined).all())
        self.assertTrue(np.all(combined <= 1.0))


if __name__ == "__main__":
    unittest.main()
