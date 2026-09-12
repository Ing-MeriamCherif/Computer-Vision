"""Native GLFW + OpenGL display window for real-time camera rendering.

Provides a 60+ FPS double-buffered OpenGL viewport with texture streaming,
fullscreen switching, and deterministic event dispatching.
"""

from __future__ import annotations

import sys
import numpy as np


class NativeOpenGLWindow:
    """Manages an OpenGL context via GLFW with texture uploading and key handling."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        title: str = "NRW Computer Vision — Native Live Challenge",
        fullscreen: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.is_fullscreen = fullscreen

        self.window = None
        self.texture_id = None
        self.key_handlers: list = []
        self._is_open = False
        self._saved_pos = (100, 100)
        self._saved_size = (width, height)

        self._init_window()

    def _init_window(self) -> bool:
        try:
            import glfw
            import OpenGL.GL as gl
        except ImportError:
            return False

        if not glfw.init():
            return False

        # Set window hints
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        monitor = glfw.get_primary_monitor() if self.is_fullscreen else None
        self.window = glfw.create_window(self.width, self.height, self.title, monitor, None)
        if not self.window:
            glfw.terminate()
            return False

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # V-Sync

        # Set callbacks
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)

        # Setup OpenGL state
        gl.glEnable(gl.GL_TEXTURE_2D)
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        glfw.show_window(self.window)
        self._is_open = True
        return True

    def _on_resize(self, window, width: int, height: int) -> None:
        try:
            import OpenGL.GL as gl
            gl.glViewport(0, 0, width, height)
            self.width = width
            self.height = height
        except Exception:
            pass

    def _on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        try:
            import glfw
            if action in (glfw.PRESS, glfw.REPEAT):
                if key == glfw.KEY_F and action == glfw.PRESS:
                    self.toggle_fullscreen()
                for handler in self.key_handlers:
                    handler(key, action, mods)
        except Exception:
            pass

    def toggle_fullscreen(self) -> None:
        """Toggle native fullscreen mode on the primary monitor."""
        try:
            import glfw
            if not self.window:
                return
            if not self.is_fullscreen:
                self._saved_pos = glfw.get_window_pos(self.window)
                self._saved_size = glfw.get_window_size(self.window)
                monitor = glfw.get_primary_monitor()
                mode = glfw.get_video_mode(monitor)
                glfw.set_window_monitor(self.window, monitor, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
                self.is_fullscreen = True
            else:
                glfw.set_window_monitor(self.window, None, self._saved_pos[0], self._saved_pos[1], self._saved_size[0], self._saved_size[1], 0)
                self.is_fullscreen = False
        except Exception:
            pass

    def should_close(self) -> bool:
        try:
            import glfw
            if not self.window:
                return True
            return bool(glfw.window_should_close(self.window))
        except Exception:
            return True

    def poll_events(self) -> None:
        try:
            import glfw
            if self.window:
                glfw.poll_events()
        except Exception:
            pass

    def render_frame(self, rgb_image: np.ndarray) -> None:
        """Upload RGB frame to OpenGL texture and draw a full-viewport quad."""
        if not self._is_open or not self.window:
            return
        try:
            import glfw
            import OpenGL.GL as gl

            glfw.make_context_current(self.window)
            fb_w, fb_h = glfw.get_framebuffer_size(self.window)
            gl.glViewport(0, 0, fb_w, fb_h)
            gl.glClearColor(0.05, 0.05, 0.05, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            h, w = rgb_image.shape[:2]
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_RGB,
                w,
                h,
                0,
                gl.GL_RGB,
                gl.GL_UNSIGNED_BYTE,
                rgb_image.data,
            )

            # Draw quad
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glOrtho(0.0, 1.0, 1.0, 0.0, -1.0, 1.0)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()

            gl.glBegin(gl.GL_QUADS)
            gl.glTexCoord2f(0.0, 0.0)
            gl.glVertex2f(0.0, 0.0)
            gl.glTexCoord2f(1.0, 0.0)
            gl.glVertex2f(1.0, 0.0)
            gl.glTexCoord2f(1.0, 1.0)
            gl.glVertex2f(1.0, 1.0)
            gl.glTexCoord2f(0.0, 1.0)
            gl.glVertex2f(0.0, 1.0)
            gl.glEnd()

            glfw.swap_buffers(self.window)
        except Exception:
            pass

    def close(self) -> None:
        self._is_open = False
        try:
            import glfw
            if self.window:
                glfw.destroy_window(self.window)
                self.window = None
            glfw.terminate()
        except Exception:
            pass
