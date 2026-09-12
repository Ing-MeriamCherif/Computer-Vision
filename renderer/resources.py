"""Reusable GPU textures for renderer inputs."""

from __future__ import annotations

import numpy as np

from contracts.render_types import MAX_LIGHTS, RenderPacket
from .config import ShadowConfig, VolumetricConfig


class RendererResources:
    """Allocate input textures once and update their contents in place."""

    def __init__(
        self,
        context,
        packet: RenderPacket,
        shadow_config: ShadowConfig | None = None,
        volumetric_config: VolumetricConfig | None = None,
    ) -> None:
        import moderngl

        self.context = context
        self.shadow_config = shadow_config or ShadowConfig()
        self.volumetric_config = volumetric_config or VolumetricConfig()
        self.size = (int(packet.rgb.shape[1]), int(packet.rgb.shape[0]))
        self.shadow_size = (
            max(1, int(round(self.size[0] * self.shadow_config.shadow_resolution_scale))),
            max(1, int(round(self.size[1] * self.shadow_config.shadow_resolution_scale))),
        )
        self.volumetric_size = (
            max(1, int(round(self.size[0] * self.volumetric_config.volumetric_resolution_scale))),
            max(1, int(round(self.size[1] * self.volumetric_config.volumetric_resolution_scale))),
        )
        self.rgb_texture = context.texture(self.size, components=3, dtype="f1", alignment=1)
        self.depth_texture = context.texture(self.size, components=1, dtype="f4", alignment=1)
        self.normal_texture = context.texture(self.size, components=3, dtype="f4", alignment=1)
        self.depth_valid_texture = context.texture(self.size, components=1, dtype="f1", alignment=1)
        self.normal_valid_texture = context.texture(self.size, components=1, dtype="f1", alignment=1)
        self.shadow_textures = [
            context.texture(self.shadow_size, components=1, dtype="f1", alignment=1)
            for _ in range(MAX_LIGHTS)
        ]
        self.ambient_texture = context.texture(self.size, components=4, dtype="f2", alignment=1)
        self.volumetric_texture = context.texture(self.volumetric_size, components=3, dtype="f2", alignment=1)
        self.volumetric_framebuffer = context.framebuffer(color_attachments=[self.volumetric_texture])
        self.direct_textures = [
            context.texture(self.size, components=4, dtype="f2", alignment=1)
            for _ in range(MAX_LIGHTS)
        ]
        # Preserve the original single-light resource names for callers that
        # inspect them, while the renderer uses the fixed-size collections.
        self.shadow_texture = self.shadow_textures[0]
        self.shadow_texture_2 = self.shadow_textures[1]
        self.direct_texture = self.direct_textures[0]
        self.direct_texture_2 = self.direct_textures[1]
        self.shadow_framebuffers = [
            context.framebuffer(color_attachments=[texture]) for texture in self.shadow_textures
        ]
        self.shadow_framebuffer = self.shadow_framebuffers[0]
        self.lighting_framebuffer = context.framebuffer(
            color_attachments=[self.ambient_texture, *self.direct_textures]
        )

        for texture in (
            self.rgb_texture,
            self.depth_texture,
            self.normal_texture,
            self.depth_valid_texture,
            self.normal_valid_texture,
        ):
            texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            texture.repeat_x = False
            texture.repeat_y = False
        for texture in self.shadow_textures:
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = False
            texture.repeat_y = False
        for texture in (self.ambient_texture, *self.direct_textures, self.volumetric_texture):
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = False
            texture.repeat_y = False
        self.upload(packet)

    @staticmethod
    def _buffer(array: np.ndarray, dtype: np.dtype | None = None) -> memoryview:
        """Expose contiguous array storage without an unconditional bytes copy."""
        prepared = np.asarray(array, dtype=dtype, order="C")
        if not prepared.flags.c_contiguous:
            prepared = np.ascontiguousarray(prepared)
        return memoryview(prepared)

    def upload(self, packet: RenderPacket) -> None:
        """Upload a packet into existing textures; texture objects are never recreated."""
        size = (int(packet.rgb.shape[1]), int(packet.rgb.shape[0]))
        if size != self.size:
            raise ValueError(f"frame resolution changed from {self.size} to {size}; recreate Renderer")
        self.rgb_texture.write(self._buffer(packet.rgb), alignment=1)
        self.depth_texture.write(self._buffer(packet.depth.depth_m), alignment=1)
        self.normal_texture.write(self._buffer(packet.normals.normals_camera, np.dtype(np.float32)), alignment=1)
        # NumPy bool uses one byte per element, matching the R8 mask textures.
        self.depth_valid_texture.write(self._buffer(packet.depth.valid_mask), alignment=1)
        self.normal_valid_texture.write(self._buffer(packet.normals.valid_mask), alignment=1)

    def upload_rgb(self, rgb: np.ndarray) -> None:
        """Update only the changing webcam RGB texture; mock geometry stays resident."""
        if rgb.dtype != np.uint8 or rgb.shape != (self.size[1], self.size[0], 3):
            raise ValueError(
                f"RGB frame must be uint8 with shape {(self.size[1], self.size[0], 3)}, "
                f"got {rgb.dtype} {rgb.shape}"
            )
        self.rgb_texture.write(self._buffer(rgb), alignment=1)

    def reconfigure_volumetric(self, config: VolumetricConfig) -> None:
        """Resize the persistent volume target only when a profile changes."""
        import moderngl

        new_size = (
            max(1, int(round(self.size[0] * config.volumetric_resolution_scale))),
            max(1, int(round(self.size[1] * config.volumetric_resolution_scale))),
        )
        self.volumetric_config = config
        if new_size == self.volumetric_size:
            return
        old_framebuffer = self.volumetric_framebuffer
        old_texture = self.volumetric_texture
        self.volumetric_size = new_size
        self.volumetric_texture = self.context.texture(new_size, components=3, dtype="f2", alignment=1)
        self.volumetric_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.volumetric_texture.repeat_x = False
        self.volumetric_texture.repeat_y = False
        self.volumetric_framebuffer = self.context.framebuffer(color_attachments=[self.volumetric_texture])
        old_framebuffer.release()
        old_texture.release()

    def release(self) -> None:
        self.lighting_framebuffer.release()
        self.volumetric_framebuffer.release()
        for framebuffer in self.shadow_framebuffers:
            framebuffer.release()
        for texture in (
            self.rgb_texture,
            self.depth_texture,
            self.normal_texture,
            self.depth_valid_texture,
            self.normal_valid_texture,
            self.volumetric_texture,
            *self.shadow_textures,
            self.ambient_texture,
            *self.direct_textures,
        ):
            texture.release()
