"""Small fullscreen ModernGL renderer for synthetic input/debug views."""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path

from contracts.render_types import RenderPacket
from .resources import RendererResources


class DebugMode(IntEnum):
    RGB = 1
    DEPTH = 2
    NORMALS = 3


class Renderer:
    """Draw packet inputs through a fullscreen GLSL pass."""

    def __init__(self, context, initial_packet: RenderPacket) -> None:
        try:
            import moderngl
        except ImportError as exc:  # pragma: no cover - environment-dependent message
            raise RuntimeError("ModernGL is required. Install project dependencies with `python -m pip install -r requirements.txt`.") from exc

        shader_dir = Path(__file__).resolve().parent / "shaders"
        self.context = context
        self._moderngl = moderngl
        self.program = context.program(
            vertex_shader=(shader_dir / "fullscreen.vert").read_text(encoding="utf-8"),
            fragment_shader=(shader_dir / "debug.frag").read_text(encoding="utf-8"),
        )
        self.vertex_array = context.vertex_array(self.program, [])
        self.resources = RendererResources(context, initial_packet)
        self.program["u_rgb"].value = 0
        self.program["u_depth"].value = 1
        self.program["u_normals"].value = 2
        self.program["u_depth_min_m"].value = 0.5
        self.program["u_depth_max_m"].value = 3.5

    def upload_packet(self, packet: RenderPacket) -> None:
        """Update the existing input textures, without reallocating them."""
        self.resources.upload(packet)

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
        self.program["u_debug_mode"].value = int(mode)
        self.vertex_array.render(mode=self._moderngl.TRIANGLES, vertices=3)

    def release(self) -> None:
        self.vertex_array.release()
        self.program.release()
        self.resources.release()
