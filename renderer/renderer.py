"""Fullscreen ModernGL renderer with Level 02 lighting and Level 04 shadows."""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path

from contracts.render_types import RenderPacket
from .config import LightingConfig, ShadowConfig
from .resources import RendererResources


class DebugMode(IntEnum):
    RGB = 1
    DEPTH = 2
    NORMALS = 3
    LAMBERTIAN = 4
    SPECULAR = 5
    FINAL = 6
    SHADOW_MASK = 7
    SHADOW_FINAL = 8


class Renderer:
    """Draw packet inputs through persistent GLSL programs and framebuffers."""

    def __init__(
        self,
        context,
        initial_packet: RenderPacket,
        config: LightingConfig | None = None,
        shadow_config: ShadowConfig | None = None,
    ) -> None:
        try:
            import moderngl
        except ImportError as exc:  # pragma: no cover - environment-dependent message
            raise RuntimeError("ModernGL is required. Install project dependencies with `python -m pip install -r requirements.txt`.") from exc

        shader_dir = Path(__file__).resolve().parent / "shaders"
        self.context = context
        self._moderngl = moderngl
        self.config = config or LightingConfig()
        self.shadow_config = shadow_config or ShadowConfig()
        vertex_shader = (shader_dir / "fullscreen.vert").read_text(encoding="utf-8")
        self.debug_program = context.program(
            vertex_shader=vertex_shader,
            fragment_shader=(shader_dir / "debug.frag").read_text(encoding="utf-8"),
        )
        self.lighting_program = context.program(
            vertex_shader=vertex_shader,
            fragment_shader=(shader_dir / "lighting.frag").read_text(encoding="utf-8"),
        )
        self.shadow_program = context.program(
            vertex_shader=vertex_shader,
            fragment_shader=(shader_dir / "shadow.frag").read_text(encoding="utf-8"),
        )
        self.composite_program = context.program(
            vertex_shader=vertex_shader,
            fragment_shader=(shader_dir / "composite.frag").read_text(encoding="utf-8"),
        )
        self.debug_vertex_array = context.vertex_array(self.debug_program, [])
        self.lighting_vertex_array = context.vertex_array(self.lighting_program, [])
        self.shadow_vertex_array = context.vertex_array(self.shadow_program, [])
        self.composite_vertex_array = context.vertex_array(self.composite_program, [])
        self.resources = RendererResources(context, initial_packet, self.shadow_config)

        for program in (self.debug_program, self.lighting_program):
            program["u_rgb"].value = 0
            program["u_depth"].value = 1
            program["u_normals"].value = 2
            program["u_depth_valid"].value = 3
            program["u_normal_valid"].value = 4
        self.debug_program["u_shadow_visibility"].value = 5
        self.shadow_program["u_depth"].value = 1
        self.shadow_program["u_depth_valid"].value = 3
        self.composite_program["u_ambient"].value = 0
        self.composite_program["u_direct"].value = 1
        self.composite_program["u_shadow_visibility"].value = 2

        self.debug_program["u_depth_min_m"].value = 0.5
        self.debug_program["u_depth_max_m"].value = 3.5
        self.lighting_program["u_ambient_strength"].value = self.config.ambient_strength
        self.lighting_program["u_specular_strength"].value = self.config.specular_strength
        self.lighting_program["u_shininess"].value = self.config.shininess
        self.lighting_program["u_attenuation_k"].value = self.config.attenuation_k

        self.shadow_program["u_shadow_steps"].value = self.shadow_config.shadow_steps
        self.shadow_program["u_shadow_bias_m"].value = self.shadow_config.shadow_bias_m
        self.shadow_program["u_shadow_thickness_m"].value = self.shadow_config.shadow_thickness_m
        self.shadow_program["u_ray_start_offset"].value = self.shadow_config.ray_start_offset

    def upload_packet(self, packet: RenderPacket) -> None:
        """Update existing input textures without reallocating them."""
        self.resources.upload(packet)

    def upload_rgb(self, rgb) -> None:
        """Update only RGB when depth and normals remain resident (webcam mock mode)."""
        self.resources.upload_rgb(rgb)

    @staticmethod
    def _draw(vertex_array, moderngl, query=None) -> None:
        if query is None:
            vertex_array.render(mode=moderngl.TRIANGLES, vertices=3)
        else:
            with query:
                vertex_array.render(mode=moderngl.TRIANGLES, vertices=3)

    def _bind_input_textures(self) -> None:
        self.resources.rgb_texture.use(location=0)
        self.resources.depth_texture.use(location=1)
        self.resources.normal_texture.use(location=2)
        self.resources.depth_valid_texture.use(location=3)
        self.resources.normal_valid_texture.use(location=4)

    @staticmethod
    def _set_camera_uniforms(program, packet: RenderPacket) -> None:
        depth = packet.depth
        program["u_fx"].value = float(depth.fx)
        program["u_fy"].value = float(depth.fy)
        program["u_cx"].value = float(depth.cx)
        program["u_cy"].value = float(depth.cy)
        program["u_image_width"].value = float(packet.rgb.shape[1])
        program["u_image_height"].value = float(packet.rgb.shape[0])

    @staticmethod
    def _light_uniforms(program, packet: RenderPacket) -> None:
        if packet.lights.lights:
            light = packet.lights.lights[0]
            program["u_light_position_camera_m"].value = tuple(
                float(value) for value in light.position_camera_m
            )
            program["u_light_active"].value = int(light.active)
            if "u_light_color_rgb" in program:
                program["u_light_color_rgb"].value = tuple(float(value) for value in light.color_rgb)
                program["u_light_intensity"].value = float(light.intensity)
        else:
            program["u_light_position_camera_m"].value = (0.0, 0.0, 0.0)
            program["u_light_active"].value = 0
            if "u_light_color_rgb" in program:
                program["u_light_color_rgb"].value = (0.0, 0.0, 0.0)
                program["u_light_intensity"].value = 0.0

    def _render_shadow_pass(self, packet: RenderPacket, query=None) -> None:
        resources = self.resources
        if not self.shadow_config.shadow_enabled:
            resources.shadow_framebuffer.use()
            self.context.viewport = (0, 0, *resources.shadow_size)
            self.context.clear(1.0, 1.0, 1.0, 1.0)
            return

        resources.shadow_framebuffer.use()
        self.context.viewport = (0, 0, *resources.shadow_size)
        self.context.clear(1.0, 1.0, 1.0, 1.0)
        resources.depth_texture.use(location=1)
        resources.depth_valid_texture.use(location=3)
        self._set_camera_uniforms(self.shadow_program, packet)
        self._light_uniforms(self.shadow_program, packet)
        self._draw(self.shadow_vertex_array, self._moderngl, query)

    def _render_lighting(self, packet: RenderPacket, mode: DebugMode, *, layered: bool, query=None) -> None:
        if layered:
            self.resources.lighting_framebuffer.use()
            self.context.viewport = (0, 0, *self.resources.size)
            self.context.clear(0.0, 0.0, 0.0, 0.0)
        else:
            self.context.screen.use()
            self.context.viewport = (0, 0, *self.context.screen.size)
            self.context.clear(0.04, 0.04, 0.05, 1.0)

        self._bind_input_textures()
        self.lighting_program["u_lighting_mode"].value = 8 if layered else int(mode)
        self._set_camera_uniforms(self.lighting_program, packet)
        self._light_uniforms(self.lighting_program, packet)
        self._draw(self.lighting_vertex_array, self._moderngl, query)

    def _render_composite(self, query=None) -> None:
        self.context.screen.use()
        self.context.viewport = (0, 0, *self.context.screen.size)
        self.context.clear(0.04, 0.04, 0.05, 1.0)
        self.resources.ambient_texture.use(location=0)
        self.resources.direct_texture.use(location=1)
        self.resources.shadow_texture.use(location=2)
        self._draw(self.composite_vertex_array, self._moderngl, query)

    def render(
        self,
        packet: RenderPacket,
        mode: DebugMode = DebugMode.RGB,
        *,
        upload_inputs: bool = True,
        gpu_queries: dict[str, object] | None = None,
    ) -> None:
        """Render one mode. Optional per-pass queries are used only by benchmarks."""
        if upload_inputs:
            self.upload_packet(packet)
        queries = gpu_queries or {}
        self._bind_input_textures()

        if mode == DebugMode.SHADOW_MASK:
            self._render_shadow_pass(packet, queries.get("shadow"))
            self.context.screen.use()
            self.context.viewport = (0, 0, *self.context.screen.size)
            self.context.clear(0.04, 0.04, 0.05, 1.0)
            self.resources.shadow_texture.use(location=5)
            self.debug_program["u_debug_mode"].value = int(mode)
            self._draw(self.debug_vertex_array, self._moderngl)
            return

        if mode == DebugMode.SHADOW_FINAL:
            self._render_shadow_pass(packet, queries.get("shadow"))
            self._render_lighting(packet, mode, layered=True, query=queries.get("lighting"))
            self._render_composite(queries.get("composition"))
            return

        if mode in (DebugMode.RGB, DebugMode.DEPTH, DebugMode.NORMALS):
            self.context.screen.use()
            self.context.viewport = (0, 0, *self.context.screen.size)
            self.context.clear(0.04, 0.04, 0.05, 1.0)
            self._bind_input_textures()
            self.debug_program["u_debug_mode"].value = int(mode)
            self._draw(self.debug_vertex_array, self._moderngl)
            return

        self._render_lighting(packet, mode, layered=False, query=queries.get("lighting"))

    def release(self) -> None:
        self.debug_vertex_array.release()
        self.lighting_vertex_array.release()
        self.shadow_vertex_array.release()
        self.composite_vertex_array.release()
        self.debug_program.release()
        self.lighting_program.release()
        self.shadow_program.release()
        self.composite_program.release()
        self.resources.release()
