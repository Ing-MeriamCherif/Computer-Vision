"""Reusable GPU textures for renderer inputs."""

from __future__ import annotations

import numpy as np

from contracts.render_types import RenderPacket


class RendererResources:
    """Allocate input textures once and update their contents in place."""

    def __init__(self, context, packet: RenderPacket) -> None:
        import moderngl

        self.context = context
        self.size = (int(packet.rgb.shape[1]), int(packet.rgb.shape[0]))
        self.rgb_texture = context.texture(self.size, components=3, dtype="f1", alignment=1)
        self.depth_texture = context.texture(self.size, components=1, dtype="f4", alignment=1)
        self.normal_texture = context.texture(self.size, components=3, dtype="f4", alignment=1)

        self.rgb_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.depth_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.normal_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        for texture in (self.rgb_texture, self.depth_texture, self.normal_texture):
            texture.repeat_x = False
            texture.repeat_y = False
        self.upload(packet)

    @staticmethod
    def _bytes(array: np.ndarray, dtype: np.dtype | None = None) -> bytes:
        prepared = np.asarray(array, dtype=dtype, order="C")
        if not prepared.flags.c_contiguous:
            prepared = np.ascontiguousarray(prepared)
        return prepared.tobytes(order="C")

    def upload(self, packet: RenderPacket) -> None:
        """Upload a packet into existing textures; texture objects are never recreated."""
        size = (int(packet.rgb.shape[1]), int(packet.rgb.shape[0]))
        if size != self.size:
            raise ValueError(f"frame resolution changed from {self.size} to {size}; recreate Renderer")
        self.rgb_texture.write(self._bytes(packet.rgb), alignment=1)
        self.depth_texture.write(self._bytes(packet.depth.depth_m), alignment=1)
        self.normal_texture.write(self._bytes(packet.normals.normals_camera, np.dtype(np.float32)), alignment=1)

    def release(self) -> None:
        for texture in (self.rgb_texture, self.depth_texture, self.normal_texture):
            texture.release()
