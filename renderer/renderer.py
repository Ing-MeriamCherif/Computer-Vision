"""Fullscreen ModernGL renderer with independent point-light shadow layers."""

from __future__ import annotations

from dataclasses import replace
from enum import IntEnum
import math
from pathlib import Path

import numpy as np

from contracts.render_types import MAX_LIGHTS, Light, RenderPacket
from .config import LightingConfig, SecondaryShadowMode, ShadowConfig, VolumetricConfig
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
    DEPTH_EDGES = 9
    SHADOW_MASK_2 = 10
    VOLUMETRIC = 11


def secondary_shadow_configs(
    primary: ShadowConfig,
    mode: SecondaryShadowMode | str,
) -> tuple[ShadowConfig, ShadowConfig]:
    """Resolve an independent secondary shadow quality policy."""
    try:
        selected = mode if isinstance(mode, SecondaryShadowMode) else SecondaryShadowMode(mode.lower())
    except (AttributeError, ValueError) as exc:
        choices = ", ".join(item.value for item in SecondaryShadowMode)
        raise ValueError(f"secondary shadow mode must be one of: {choices}") from exc

    if selected is SecondaryShadowMode.BALANCED:
        secondary = primary
    elif selected is SecondaryShadowMode.SAFE:
        from .config import ShadowQualityProfile

        safe = ShadowConfig.for_profile(ShadowQualityProfile.SAFE)
        secondary = replace(
            primary,
            shadow_steps=safe.shadow_steps,
            shadow_softening_enabled=safe.shadow_softening_enabled,
            shadow_soft_samples=safe.shadow_soft_samples,
            shadow_soft_radius=safe.shadow_soft_radius,
            shadow_edge_aware_upsampling=safe.shadow_edge_aware_upsampling,
        )
    else:
        secondary = replace(primary, shadow_enabled=False)
    return primary, secondary


class Renderer:
    """Draw packet inputs through persistent GLSL programs and framebuffers."""

    def __init__(
        self,
        context,
        initial_packet: RenderPacket,
        config: LightingConfig | None = None,
        shadow_config: ShadowConfig | None = None,
        secondary_shadow_mode: SecondaryShadowMode | str = SecondaryShadowMode.BALANCED,
        volumetric_config: VolumetricConfig | None = None,
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
        self.secondary_shadow_mode = (
            secondary_shadow_mode
            if isinstance(secondary_shadow_mode, SecondaryShadowMode)
            else SecondaryShadowMode(secondary_shadow_mode.lower())
        )
        self.shadow_configs = secondary_shadow_configs(self.shadow_config, self.secondary_shadow_mode)
        self.volumetric_config = volumetric_config or VolumetricConfig()
        self.volumetric_enabled = self.volumetric_config.volumetric_enabled
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
        self.volumetric_program = context.program(
            vertex_shader=vertex_shader,
            fragment_shader=(shader_dir / "volumetric.frag").read_text(encoding="utf-8"),
        )
        self.debug_vertex_array = context.vertex_array(self.debug_program, [])
        self.lighting_vertex_array = context.vertex_array(self.lighting_program, [])
        self.shadow_vertex_array = context.vertex_array(self.shadow_program, [])
        self.composite_vertex_array = context.vertex_array(self.composite_program, [])
        self.volumetric_vertex_array = context.vertex_array(self.volumetric_program, [])
        self.resources = RendererResources(
            context,
            initial_packet,
            self.shadow_config,
            self.volumetric_config,
        )

        for program in (self.debug_program, self.lighting_program):
            program["u_rgb"].value = 0
            program["u_depth"].value = 1
            program["u_normals"].value = 2
            program["u_depth_valid"].value = 3
            program["u_normal_valid"].value = 4
        self.debug_program["u_shadow_visibility"].value = 5
        self.debug_program["u_shadow_visibility_2"].value = 6
        self.debug_program["u_volumetric"].value = 7
        self.shadow_program["u_depth"].value = 1
        self.shadow_program["u_depth_valid"].value = 3

        for name, unit in (
            ("u_ambient", 0),
            ("u_direct_1", 1),
            ("u_direct_2", 5),
            ("u_shadow_visibility_1", 2),
            ("u_shadow_visibility_2", 6),
            ("u_depth", 3),
            ("u_depth_valid", 4),
            ("u_volumetric", 7),
        ):
            self.composite_program[name].value = unit
        self.composite_program["u_volumetric_enabled"].value = 0

        self.volumetric_program["u_depth"].value = 1
        self.volumetric_program["u_depth_valid"].value = 3
        self.volumetric_program["u_shadow_visibility_1"].value = 5
        self.volumetric_program["u_shadow_visibility_2"].value = 6
        self.volumetric_program["u_attenuation_k"].value = self.config.attenuation_k
        self.volumetric_program["u_volumetric_samples"].value = self.volumetric_config.volumetric_samples
        self.volumetric_program["u_volumetric_density"].value = self.volumetric_config.volumetric_density
        self.volumetric_program["u_volumetric_intensity"].value = self.volumetric_config.volumetric_intensity
        self.volumetric_program["u_volumetric_decay"].value = self.volumetric_config.volumetric_decay
        self.debug_program["u_volumetric_debug_scale"].value = 2.0

        self.debug_program["u_depth_min_m"].value = 0.5
        self.debug_program["u_depth_max_m"].value = 3.5
        self.debug_program["u_depth_edge_threshold_m"].value = self.shadow_config.depth_edge_threshold_m
        self.lighting_program["u_ambient_strength"].value = self.config.ambient_strength
        self.lighting_program["u_specular_strength"].value = self.config.specular_strength
        self.lighting_program["u_shininess"].value = self.config.shininess
        self.lighting_program["u_attenuation_k"].value = self.config.attenuation_k

        for light_index, shadow in enumerate(self.shadow_configs):
            self._set_shadow_config_uniforms(self.composite_program, light_index, shadow)

    @staticmethod
    def _set_shadow_config_uniforms(program, light_index: int, config: ShadowConfig) -> None:
        suffix = light_index + 1
        program[f"u_shadow_softening_enabled_{suffix}"].value = int(config.shadow_softening_enabled)
        program[f"u_shadow_soft_samples_{suffix}"].value = config.shadow_soft_samples
        program[f"u_shadow_edge_aware_upsampling_{suffix}"].value = int(config.shadow_edge_aware_upsampling)
        program[f"u_shadow_soft_radius_{suffix}"].value = config.shadow_soft_radius
        program[f"u_depth_edge_threshold_m_{suffix}"].value = config.depth_edge_threshold_m

    def upload_packet(self, packet: RenderPacket) -> None:
        """Update existing input textures without reallocating them."""
        self.resources.upload(packet)

    def upload_rgb(self, rgb) -> None:
        """Update only RGB when depth and normals remain resident (webcam mock mode)."""
        self.resources.upload_rgb(rgb)

    def set_volumetric_enabled(self, enabled: bool) -> None:
        """Enable/disable scattering immediately without reallocating GPU resources."""
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        self.volumetric_enabled = enabled

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
    def _safe_light(light: Light | None) -> tuple[np.ndarray, np.ndarray, float, bool]:
        """Sanitize mutated light arrays so one invalid light cannot poison the other."""
        zeros = np.zeros(3, dtype=np.float32)
        if light is None:
            return zeros, zeros.copy(), 0.0, False
        try:
            position = np.asarray(light.position_camera_m, dtype=np.float64)
            color = np.asarray(light.color_rgb, dtype=np.float64)
            intensity = float(light.intensity)
            active = bool(light.active)
        except (TypeError, ValueError, OverflowError):
            return zeros, zeros.copy(), 0.0, False
        if (
            position.shape != (3,)
            or color.shape != (3,)
            or not np.isfinite(position).all()
            or not np.isfinite(color).all()
            or np.any(color < 0.0)
            or np.any(color > 1.0)
            or not math.isfinite(intensity)
            or intensity < 0.0
        ):
            return zeros, zeros.copy(), 0.0, False
        return position.astype(np.float32), color.astype(np.float32), intensity, active

    def _light_values(self, packet: RenderPacket) -> list[tuple[np.ndarray, np.ndarray, float, bool]]:
        return [self._safe_light(light) for light in packet.lights.lights[:MAX_LIGHTS]]

    def _set_lighting_light_uniforms(self, packet: RenderPacket, program=None) -> None:
        lights = self._light_values(packet)
        positions = np.zeros((MAX_LIGHTS, 3), dtype=np.float32)
        colors = np.zeros((MAX_LIGHTS, 3), dtype=np.float32)
        intensities = np.zeros(MAX_LIGHTS, dtype=np.float32)
        active = np.zeros(MAX_LIGHTS, dtype=np.int32)
        for index, (position, color, intensity, enabled) in enumerate(lights):
            positions[index] = position
            colors[index] = color
            intensities[index] = intensity
            active[index] = int(enabled and intensity > 0.0)
        program = self.lighting_program if program is None else program
        program["u_light_count"].value = len(lights)
        program["u_light_positions_camera_m"].value = tuple(
            tuple(float(v) for v in row) for row in positions
        )
        program["u_light_colors_rgb"].value = tuple(
            tuple(float(v) for v in row) for row in colors
        )
        program["u_light_intensities"].value = tuple(float(v) for v in intensities)
        program["u_light_active"].value = tuple(int(v) for v in active)

    def _set_shadow_light_uniforms(self, packet: RenderPacket, light_index: int) -> bool:
        lights = self._light_values(packet)
        if light_index >= len(lights):
            position, _color, _intensity, active = self._safe_light(None)
        else:
            position, _color, intensity, active = lights[light_index]
            active = active and intensity > 0.0
        self.shadow_program["u_light_position_camera_m"].value = tuple(float(v) for v in position)
        self.shadow_program["u_light_active"].value = int(active)
        return active

    def _clear_shadow(self, light_index: int) -> None:
        resources = self.resources
        resources.shadow_framebuffers[light_index].use()
        self.context.viewport = (0, 0, *resources.shadow_size)
        self.context.clear(1.0, 1.0, 1.0, 1.0)

    def _render_shadow_pass(self, packet: RenderPacket, light_index: int, query=None) -> bool:
        resources = self.resources
        config = self.shadow_configs[light_index]
        active = self._set_shadow_light_uniforms(packet, light_index)
        if not config.shadow_enabled or not active:
            self._clear_shadow(light_index)
            return False

        resources.shadow_framebuffers[light_index].use()
        self.context.viewport = (0, 0, *resources.shadow_size)
        self.context.clear(1.0, 1.0, 1.0, 1.0)
        resources.depth_texture.use(location=1)
        resources.depth_valid_texture.use(location=3)
        self._set_camera_uniforms(self.shadow_program, packet)
        self.shadow_program["u_shadow_steps"].value = config.shadow_steps
        self.shadow_program["u_shadow_bias_m"].value = config.shadow_bias_m
        self.shadow_program["u_shadow_thickness_m"].value = config.shadow_thickness_m
        self.shadow_program["u_ray_start_offset"].value = config.ray_start_offset
        self._draw(self.shadow_vertex_array, self._moderngl, query)
        return True

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
        self._set_lighting_light_uniforms(packet)
        self._draw(self.lighting_vertex_array, self._moderngl, query)

    def _render_composite(self, query=None) -> None:
        self.context.screen.use()
        self.context.viewport = (0, 0, *self.context.screen.size)
        self.context.clear(0.04, 0.04, 0.05, 1.0)
        self.resources.ambient_texture.use(location=0)
        self.resources.direct_textures[0].use(location=1)
        self.resources.shadow_textures[0].use(location=2)
        self.resources.depth_texture.use(location=3)
        self.resources.depth_valid_texture.use(location=4)
        self.resources.direct_textures[1].use(location=5)
        self.resources.shadow_textures[1].use(location=6)
        self.resources.volumetric_texture.use(location=7)
        self.composite_program["u_volumetric_enabled"].value = int(self.volumetric_enabled)
        self._draw(self.composite_vertex_array, self._moderngl, query)

    def _clear_volumetric(self) -> None:
        self.resources.volumetric_framebuffer.use()
        self.context.viewport = (0, 0, *self.resources.volumetric_size)
        self.context.clear(0.0, 0.0, 0.0, 0.0)

    def _render_volumetric(self, packet: RenderPacket, query=None, *, clear_when_disabled: bool = False) -> bool:
        if not self.volumetric_enabled:
            if clear_when_disabled:
                self._clear_volumetric()
            return False

        self.resources.volumetric_framebuffer.use()
        self.context.viewport = (0, 0, *self.resources.volumetric_size)
        self.context.clear(0.0, 0.0, 0.0, 0.0)
        self.resources.depth_texture.use(location=1)
        self.resources.depth_valid_texture.use(location=3)
        self.resources.shadow_textures[0].use(location=5)
        self.resources.shadow_textures[1].use(location=6)
        self._set_camera_uniforms(self.volumetric_program, packet)
        self._set_lighting_light_uniforms(packet, self.volumetric_program)
        self._draw(self.volumetric_vertex_array, self._moderngl, query)
        return True

    def _render_volumetric_debug(self, packet: RenderPacket, queries: dict[str, object] | None = None) -> None:
        queries = queries or {}
        if self.volumetric_enabled:
            self._render_shadow_pass(packet, 0, queries.get("shadow_1"))
            self._render_shadow_pass(packet, 1, queries.get("shadow_2"))
        self._render_volumetric(
            packet,
            queries.get("volumetric"),
            clear_when_disabled=True,
        )
        self.context.screen.use()
        self.context.viewport = (0, 0, *self.context.screen.size)
        self.context.clear(0.0, 0.0, 0.0, 1.0)
        self.resources.volumetric_texture.use(location=7)
        self.debug_program["u_debug_mode"].value = int(DebugMode.VOLUMETRIC)
        self._draw(self.debug_vertex_array, self._moderngl)

    def _render_shadow_debug(self, packet: RenderPacket, light_index: int, query=None) -> None:
        self._render_shadow_pass(packet, light_index, query)
        self.context.screen.use()
        self.context.viewport = (0, 0, *self.context.screen.size)
        self.context.clear(0.04, 0.04, 0.05, 1.0)
        self.resources.shadow_textures[light_index].use(location=5 + light_index)
        self.debug_program["u_debug_mode"].value = int(
            DebugMode.SHADOW_MASK if light_index == 0 else DebugMode.SHADOW_MASK_2
        )
        self._draw(self.debug_vertex_array, self._moderngl)

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
            self._render_shadow_debug(packet, 0, queries.get("shadow_1", queries.get("shadow")))
            return
        if mode == DebugMode.SHADOW_MASK_2:
            self._render_shadow_debug(packet, 1, queries.get("shadow_2"))
            return
        if mode == DebugMode.VOLUMETRIC:
            self._render_volumetric_debug(packet, queries)
            return
        if mode == DebugMode.SHADOW_FINAL:
            self._render_shadow_pass(packet, 0, queries.get("shadow_1", queries.get("shadow")))
            self._render_shadow_pass(packet, 1, queries.get("shadow_2"))
            self._render_lighting(packet, mode, layered=True, query=queries.get("lighting"))
            self._render_volumetric(packet, queries.get("volumetric"))
            self._render_composite(queries.get("composition"))
            return

        if mode in (DebugMode.RGB, DebugMode.DEPTH, DebugMode.NORMALS, DebugMode.DEPTH_EDGES):
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
        self.volumetric_vertex_array.release()
        self.debug_program.release()
        self.lighting_program.release()
        self.shadow_program.release()
        self.composite_program.release()
        self.volumetric_program.release()
        self.resources.release()
