"""Final stage-profile and zero-copy upload preparation tests."""

import unittest

import numpy as np

from renderer.config import SecondaryShadowMode, StageQualityProfile, stage_quality_settings
from renderer.resources import RendererResources


class StageQualityTests(unittest.TestCase):
    def test_profiles_resolve_complete_pipeline_settings(self) -> None:
        safe = stage_quality_settings(StageQualityProfile.SAFE)
        balanced = stage_quality_settings("balanced")
        high = stage_quality_settings("high")

        self.assertEqual((safe.shadow.shadow_resolution_scale, safe.shadow.shadow_steps), (0.5, 8))
        self.assertFalse(safe.shadow.shadow_softening_enabled)
        self.assertEqual(safe.secondary_shadow_mode, SecondaryShadowMode.SAFE)
        self.assertEqual((safe.volumetric.volumetric_resolution_scale, safe.volumetric.volumetric_samples), (0.25, 8))

        self.assertEqual((balanced.shadow.shadow_resolution_scale, balanced.shadow.shadow_steps), (0.5, 16))
        self.assertTrue(balanced.shadow.shadow_softening_enabled)
        self.assertEqual(balanced.secondary_shadow_mode, SecondaryShadowMode.BALANCED)
        self.assertEqual((balanced.volumetric.volumetric_resolution_scale, balanced.volumetric.volumetric_samples), (0.25, 12))

        self.assertEqual((high.shadow.shadow_resolution_scale, high.shadow.shadow_steps), (0.5, 20))
        self.assertEqual(high.shadow.shadow_soft_samples, 8)
        self.assertEqual((high.volumetric.volumetric_resolution_scale, high.volumetric.volumetric_samples), (0.5, 20))
        for settings in (safe, balanced, high):
            self.assertTrue(settings.volumetric.volumetric_enabled)
            self.assertTrue(settings.light_orb_enabled)

    def test_unknown_stage_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "stage profile"):
            stage_quality_settings("cinematic")

    def test_contiguous_matching_array_uses_original_storage(self) -> None:
        array = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        view = RendererResources._buffer(array)
        self.assertIs(view.obj, array)
        self.assertEqual(view.nbytes, array.nbytes)

    def test_dtype_conversion_occurs_only_when_required(self) -> None:
        half = np.ones((2, 2, 3), dtype=np.float16)
        converted = RendererResources._buffer(half, np.dtype(np.float32))
        self.assertEqual(converted.format, "f")
        self.assertEqual(converted.nbytes, half.size * np.dtype(np.float32).itemsize)


if __name__ == "__main__":
    unittest.main()
