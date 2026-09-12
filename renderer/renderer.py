"""Fullscreen ModernGL debug and single-light Level 02 renderer."""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path

from contracts.render_types import RenderPacket
from .config import LightingConfig
from .resources import RendererResources


class DebugMode(IntEnum):
    RGB = 1
    DEPTH = 2
    NORMALS = 3
    LAMBERTIAN = 4
    SPECULAR = 5
    FINAL = 6


class Renderer:
    """Draw packet inputs through a fullscreen GLSL pass."""

    def __init__(
        self,
        context,
        initial_packet: RenderPacket,
        config: LightingConfig | None = None,
    ) -> None:
        try:
            import moderngl
        except ImportError as exc:  # pragma: no cover - environment-dependent message
            raise RuntimeError("ModernGL is required. Install project dependencies with `python -m pip install -r requirements.txt`.") from exc

        shader_dir = Path(__file__).resolve().parent / "shaders"
        self.context = context
        self._moderngl = moderngl
        self.config = config or LightingConfig()
        self.debug_program = context.program(
            vertex_shader=(shader_dir / "fullscreen.vert").read_text(encoding="utf-8"),
            fragment_shader=(shader_dir / "debug.frag").read_text(encoding="utf-8"),
        )
        self.lighting_program = context.program(
            vertex_shader=(shader_dir / "fullscreen.vert").read_text(encoding="utf-8"),
            fragment_shader=(shader_dir / "lighting.frag").read_text(encoding="utf-8"),
        )
        self.debug_vertex_array = context.vertex_array(self.debug_program, [])
        self.lighting_vertex_array = context.vertex_array(self.lighting_program, [])
        self.resources = RendererResources(context, initial_packet)
        for program in (self.debug_program, self.lighting_program):
            program["u_rgb"].value = 0
            program["u_depth"].value = 1
            program["u_normals"].value = 2
            program["u_depth_valid"].value = 3
            program["u_normal_valid"].value = 4
        self.debug_program["u_depth_min_m"].value = 0.5
        self.debug_program["u_depth_max_m"].value = 3.5
        self.lighting_program["u_ambient_strength"].value = self.config.ambient_strength
        self.lighting_program["u_specular_strength"].value = self.config.specular_strength
        self.lighting_program["u_shininess"].value = self.config.shininess
        self.lighting_program["u_attenuation_k"].value = self.config.attenuation_k

    def upload_packet(self, packet: RenderPacket) -> None:
        """Update the existing input textures, without reallocating them."""
        self.resources.upload(packet)

    def upload_rgb(self, rgb) -> None:
        """Update just RGB when depth and normals are unchanged, as in webcam mock mode."""
        self.resources.upload_rgb(rgb)

    def render(
        self,
        packet: RenderPacket,
        mode: DebugMode = DebugMode.RGB,
        *,
        upload_inputs: bool = True,
    ) -> None:
        """Draw the selected view; benchmarks may time uploads separately."""
        if upload_inputs:
            self.upload_packet(packet)
        self.context.screen.use()
        self.context.viewport = (0, 0, *self.context.screen.size)
        self.context.clear(0.04, 0.04, 0.05, 1.0)
        self.resources.rgb_texture.use(location=0)
        self.resources.depth_texture.use(location=1)
        self.resources.normal_texture.use(location=2)
        self.resources.depth_valid_texture.use(location=3)
        self.resources.normal_valid_texture.use(location=4)

        if mode in (DebugMode.RGB, DebugMode.DEPTH, DebugMode.NORMALS):
            self.debug_program["u_debug_mode"].value = int(mode)
            self.debug_vertex_array.render(mode=self._moderngl.TRIANGLES, vertices=3)
            return

        depth = packet.depth
        self.lighting_program["u_lighting_mode"].value = int(mode)
        self.lighting_program["u_fx"].value = float(depth.fx)
        self.lighting_program["u_fy"].value = float(depth.fy)
        self.lighting_program["u_cx"].value = float(depth.cx)
        self.lighting_program["u_cy"].value = float(depth.cy)
        self.lighting_program["u_image_width"].value = float(packet.rgb.shape[1])
        self.lighting_program["u_image_height"].value = float(packet.rgb.shape[0])

        if packet.lights.lights:
            light = packet.lights.lights[0]
            self.lighting_program["u_light_position_camera_m"].value = tuple(
                float(value) for value in light.position_camera_m
            )
            self.lighting_program["u_light_color_rgb"].value = tuple(
                float(value) for value in light.color_rgb
            )
            self.lighting_program["u_light_intensity"].value = float(light.intensity)
            self.lighting_program["u_light_active"].value = int(light.active)
        else:
            self.lighting_program["u_light_position_camera_m"].value = (0.0, 0.0, 0.0)
            self.lighting_program["u_light_color_rgb"].value = (0.0, 0.0, 0.0)
            self.lighting_program["u_light_intensity"].value = 0.0
            self.lighting_program["u_light_active"].value = 0

        self.lighting_vertex_array.render(mode=self._moderngl.TRIANGLES, vertices=3)

    def release(self) -> None:
        self.debug_vertex_array.release()
        self.lighting_vertex_array.release()
        self.debug_program.release()
        self.lighting_program.release()
        self.resources.release()
