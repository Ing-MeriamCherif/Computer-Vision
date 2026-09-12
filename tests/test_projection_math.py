"""Camera projection/back-projection tests for the agreed camera convention."""

import unittest

import numpy as np

from renderer.projection import project_camera_point, reconstruct_camera_point


class ProjectionMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = 800.0
        self.fy = 800.0
        self.cx = 319.5
        self.cy = 239.5

    def reconstruct(self, u: float, v: float, z_m: float) -> np.ndarray:
        return reconstruct_camera_point(
            u, v, z_m, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )

    def test_principal_point_reconstructs_to_camera_axis(self) -> None:
        point = self.reconstruct(self.cx, self.cy, 2.0)
        np.testing.assert_allclose(point, [0.0, 0.0, 2.0], rtol=0.0, atol=1e-7)

    def test_right_and_left_pixels_have_expected_x_sign(self) -> None:
        self.assertGreater(self.reconstruct(self.cx + 100.0, self.cy, 2.0)[0], 0.0)
        self.assertLess(self.reconstruct(self.cx - 100.0, self.cy, 2.0)[0], 0.0)

    def test_below_and_above_pixels_have_expected_y_sign(self) -> None:
        self.assertGreater(self.reconstruct(self.cx, self.cy + 100.0, 2.0)[1], 0.0)
        self.assertLess(self.reconstruct(self.cx, self.cy - 100.0, 2.0)[1], 0.0)

    def test_projection_round_trip(self) -> None:
        point = self.reconstruct(417.25, 302.75, 2.4)
        u, v = project_camera_point(
            point, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )
        self.assertAlmostEqual(u, 417.25, places=10)
        self.assertAlmostEqual(v, 302.75, places=10)

    def test_screen_space_shadow_ray_projection_uses_camera_axes(self) -> None:
        receiver = self.reconstruct(self.cx, self.cy, 3.0)
        light = np.array([0.6, 0.4, 0.75], dtype=np.float64)
        ray_sample = receiver + 0.5 * (light - receiver)
        u, v = project_camera_point(
            ray_sample, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )
        self.assertGreater(u, self.cx)  # +X projects right.
        self.assertGreater(v, self.cy)  # +Y projects down.

    def test_rejects_nonpositive_or_nonfinite_depth(self) -> None:
        for depth in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                self.reconstruct(self.cx, self.cy, depth)


if __name__ == "__main__":
    unittest.main()
