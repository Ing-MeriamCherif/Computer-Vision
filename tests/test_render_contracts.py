"""Validation tests for the shared renderer contracts."""

import unittest

import numpy as np

from contracts.render_types import MAX_LIGHTS, DepthFrame, Light, LightState, NormalFrame, RenderPacket
from renderer.renderer import Renderer


def make_packet(width: int = 4, height: int = 3, lights: list[Light] | None = None) -> RenderPacket:
    timestamp = 12.5
    valid = np.ones((height, width), dtype=np.bool_)
    return RenderPacket(
        rgb=np.zeros((height, width, 3), dtype=np.uint8),
        depth=DepthFrame(
            depth_m=np.full((height, width), 3.0, dtype=np.float32),
            valid_mask=valid.copy(),
            fx=500.0,
            fy=500.0,
            cx=(width - 1) / 2.0,
            cy=(height - 1) / 2.0,
            timestamp_s=timestamp,
        ),
        normals=NormalFrame(
            normals_camera=np.broadcast_to(
                np.array([0.0, 0.0, -1.0], dtype=np.float32), (height, width, 3)
            ).copy(),
            valid_mask=valid.copy(),
            timestamp_s=timestamp,
        ),
        lights=LightState(lights=[make_light()] if lights is None else lights, timestamp_s=timestamp),
        frame_id=7,
        timestamp_s=timestamp,
    )


def make_light(position: tuple[float, float, float] = (0.0, 0.0, 2.0)) -> Light:
    return Light(
        position_camera_m=np.array(position, dtype=np.float32),
        color_rgb=np.array([1.0, 0.9, 0.8], dtype=np.float32),
        intensity=1.0,
        active=True,
        confidence=0.95,
    )


class RenderContractTests(unittest.TestCase):
    def test_render_packet_constructs_with_expected_shapes_and_dtypes(self) -> None:
        packet = make_packet()
        self.assertEqual(packet.rgb.shape, (3, 4, 3))
        self.assertEqual(packet.rgb.dtype, np.uint8)
        self.assertEqual(packet.depth.depth_m.dtype, np.float32)
        self.assertEqual(packet.depth.valid_mask.dtype, np.bool_)
        self.assertEqual(packet.normals.normals_camera.dtype, np.float32)
        self.assertEqual(packet.normals.normals_camera.shape, (3, 4, 3))
        self.assertEqual(packet.lights.lights[0].position_camera_m.tolist(), [0.0, 0.0, 2.0])

    def test_front_facing_normal_uses_negative_z(self) -> None:
        packet = make_packet()
        np.testing.assert_array_equal(packet.normals.normals_camera[0, 0], [0.0, 0.0, -1.0])

    def test_normal_frame_accepts_plan_float16_or_float32(self) -> None:
        normals = np.zeros((2, 2, 3), dtype=np.float16)
        normals[:, :, 2] = -1.0
        NormalFrame(normals, np.ones((2, 2), dtype=np.bool_), 1.0)

    def test_rejects_rgb_shape_and_dtype_mismatches(self) -> None:
        packet = make_packet()
        with self.assertRaisesRegex(ValueError, "HxWx3"):
            RenderPacket(np.zeros((3, 4), dtype=np.uint8), packet.depth, packet.normals, packet.lights, 0, 1.0)
        with self.assertRaisesRegex(TypeError, "uint8"):
            RenderPacket(np.zeros((3, 4, 3), dtype=np.float32), packet.depth, packet.normals, packet.lights, 0, 1.0)

    def test_rejects_mismatched_frame_resolutions(self) -> None:
        packet = make_packet()
        with self.assertRaisesRegex(ValueError, "resolutions must match"):
            RenderPacket(
                np.zeros((5, 4, 3), dtype=np.uint8), packet.depth, packet.normals, packet.lights, 0, 1.0
            )

    def test_rejects_wrong_mask_shape_and_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid_mask"):
            DepthFrame(np.ones((2, 3), np.float32), np.ones((3, 2), np.bool_), 1, 1, 0, 0, 1)
        with self.assertRaisesRegex(TypeError, "bool"):
            DepthFrame(np.ones((2, 3), np.float32), np.ones((2, 3), np.uint8), 1, 1, 0, 0, 1)

    def test_rejects_invalid_depth_and_nonfinite_light_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid depth_m"):
            DepthFrame(np.zeros((2, 3), np.float32), np.ones((2, 3), np.bool_), 1, 1, 0, 0, 1)
        with self.assertRaisesRegex(ValueError, "finite"):
            make_light((float("nan"), 0.0, 2.0))

    def test_accepts_up_to_two_lights_and_rejects_more(self) -> None:
        second = make_light((0.5, 0.0, 2.0))
        packet = make_packet(lights=[make_light(), second])
        self.assertEqual(MAX_LIGHTS, 2)
        self.assertEqual(len(packet.lights.lights), MAX_LIGHTS)
        self.assertTrue(packet.lights.lights[1].active)
        np.testing.assert_array_equal(packet.lights.lights[1].position_camera_m, [0.5, 0.0, 2.0])
        with self.assertRaisesRegex(ValueError, "at most 2"):
            LightState([make_light(), second, make_light()], 1.0)

    def test_single_light_packet_remains_supported(self) -> None:
        packet = make_packet(lights=[make_light()])
        self.assertEqual(len(packet.lights.lights), 1)
        self.assertTrue(packet.lights.lights[0].active)

    def test_second_light_can_be_inactive_without_changing_first(self) -> None:
        first = make_light((-0.4, 0.0, 0.75))
        second = Light(
            np.array([0.6, 0.2, 0.8], dtype=np.float32),
            np.array([0.45, 0.70, 1.0], dtype=np.float32),
            0.75,
            active=False,
        )
        packet = make_packet(lights=[first, second])
        np.testing.assert_allclose(packet.lights.lights[0].color_rgb, [1.0, 0.9, 0.8])
        self.assertEqual(packet.lights.lights[0].intensity, 1.0)
        self.assertFalse(packet.lights.lights[1].active)
        np.testing.assert_allclose(packet.lights.lights[1].color_rgb, [0.45, 0.70, 1.0])
        self.assertEqual(packet.lights.lights[1].intensity, 0.75)

    def test_mutated_invalid_light_is_deactivated_safely(self) -> None:
        invalid = make_light()
        invalid.position_camera_m[0] = np.nan
        _position, _color, _intensity, active = Renderer._safe_light(invalid)
        self.assertFalse(active)

        valid = make_light((0.4, 0.0, 0.75))
        position, color, intensity, active = Renderer._safe_light(valid)
        self.assertTrue(active)
        np.testing.assert_allclose(position, (0.4, 0.0, 0.75))
        np.testing.assert_allclose(color, (1.0, 0.9, 0.8))
        self.assertEqual(intensity, 1.0)

    def test_rejects_wrong_light_coordinate_dtype_or_range(self) -> None:
        with self.assertRaisesRegex(TypeError, "float32"):
            Light(np.array([0, 0, 2], dtype=np.int32), np.ones(3, dtype=np.float32), 1.0)
        with self.assertRaisesRegex(ValueError, "0..1"):
            Light(np.zeros(3, dtype=np.float32), np.array([1.2, 0.0, 0.0], dtype=np.float32), 1.0)


if __name__ == "__main__":
    unittest.main()
