"""Tests for renderer test-input preparation."""

import unittest

import cv2
import numpy as np

from tools.renderer_synthetic_demo import letterbox_bgr_to_rgb


class RendererInputTests(unittest.TestCase):
    def test_letterbox_preserves_aspect_ratio_and_converts_bgr_to_rgb(self) -> None:
        source_bgr = np.zeros((2, 4, 3), dtype=np.uint8)
        source_bgr[:, :] = [10, 20, 30]
        destination_rgb = np.full((4, 4, 3), 255, dtype=np.uint8)

        placement = letterbox_bgr_to_rgb(source_bgr, destination_rgb, cv2)

        self.assertEqual(placement, (0, 1, 4, 2))
        expected_rgb = np.broadcast_to(np.array([30, 20, 10], dtype=np.uint8), (2, 4, 3))
        np.testing.assert_array_equal(destination_rgb[1:3], expected_rgb)
        self.assertFalse(destination_rgb[0].any())
        self.assertFalse(destination_rgb[3].any())

    def test_letterbox_reuses_destination_array(self) -> None:
        source_bgr = np.zeros((3, 3, 3), dtype=np.uint8)
        destination_rgb = np.empty((6, 8, 3), dtype=np.uint8)
        identity = id(destination_rgb)

        letterbox_bgr_to_rgb(source_bgr, destination_rgb, cv2)

        self.assertEqual(id(destination_rgb), identity)


if __name__ == "__main__":
    unittest.main()
