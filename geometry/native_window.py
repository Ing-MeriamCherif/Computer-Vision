"""Native GLFW/OpenGL core-profile window for the existing NumPy display API."""

from __future__ import annotations

import ctypes
import numpy as np


_VERTEX_SHADER = r"""#version 330 core
layout(location=0) in vec2 aPosition;
layout(location=1) in vec2 aUv;
out vec2 vUv;
void main() {
    vUv = aUv;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
"""

_FRAGMENT_SHADER = r"""#version 330 core
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uFrame;
void main() {
    oColor = vec4(texture(uFrame, vUv).rgb, 1.0);
}
"""


class NativeOpenGLWindow:
    """Persistent texture upload and shader-based fullscreen-triangle display."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        title: str = "NRW Computer Vision - Native Live Challenge",
        fullscreen: bool = False,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.title = title
        self.is_fullscreen = bool(fullscreen)
        self.window = None
        self.texture_id = 0
        self.program = 0
        self.vao = 0
        self.vbo = 0
        self.key_handlers: list = []
        self._is_open = False
        self._saved_pos = (100, 100)
        self._saved_size = (self.width, self.height)
        self._texture_size: tuple[int, int] | None = None
        self.last_error: str | None = None
        self._glfw = None
        self._gl = None
        self._glfw_acquired = False
        self._init_window()

    def _init_window(self) -> bool:
        try:
            import glfw
            import OpenGL.GL as gl
        except ImportError as exc:
            self.last_error = f"GLFW/PyOpenGL unavailable: {exc}"
            print(f"NATIVE OPENGL WINDOW UNAVAILABLE: {self.last_error}")
            return False

        self._glfw, self._gl = glfw, gl
        from .glfw_support import acquire

        if not acquire(glfw):
            self.last_error = "GLFW initialization failed"
            print(f"NATIVE OPENGL WINDOW UNAVAILABLE: {self.last_error}")
            return False
        self._glfw_acquired = True

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        monitor = glfw.get_primary_monitor() if self.is_fullscreen else None
        self.window = glfw.create_window(self.width, self.height, self.title, monitor, None)
        if not self.window:
            from .glfw_support import release

            release(glfw)
            self._glfw_acquired = False
            self.last_error = "could not create an OpenGL 3.3 core window"
            print(f"NATIVE OPENGL WINDOW UNAVAILABLE: {self.last_error}")
            return False

        try:
            self._configure_context(glfw, gl)
        except Exception:
            self.close()
            raise
        return True

    def _configure_context(self, glfw, gl) -> None:
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.gl_version = gl.glGetString(gl.GL_VERSION).decode("ascii", "replace")
        self.program = self._link_program(_VERTEX_SHADER, _FRAGMENT_SHADER)
        self.vao = int(gl.glGenVertexArrays(1))
        self.vbo = int(gl.glGenBuffers(1))
        vertices = np.asarray(
            [
                -1.0, -1.0, 0.0, 0.0,
                 3.0, -1.0, 2.0, 0.0,
                -1.0,  3.0, 0.0, 2.0,
            ],
            dtype=np.float32,
        )
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * vertices.itemsize, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * vertices.itemsize, ctypes.c_void_p(2 * vertices.itemsize))

        self.texture_id = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_BLEND)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.show_window(self.window)
        self._is_open = True

    def _link_program(self, vertex_source: str, fragment_source: str) -> int:
        gl = self._gl
        shaders: list[int] = []
        program = 0
        try:
            for kind, source in ((gl.GL_VERTEX_SHADER, vertex_source), (gl.GL_FRAGMENT_SHADER, fragment_source)):
                shader = gl.glCreateShader(kind)
                shaders.append(shader)
                gl.glShaderSource(shader, source)
                gl.glCompileShader(shader)
                if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
                    log = gl.glGetShaderInfoLog(shader).decode("utf-8", "replace")
                    raise RuntimeError(f"native display shader compile failed: {log}")
            program = gl.glCreateProgram()
            for shader in shaders:
                gl.glAttachShader(program, shader)
            gl.glLinkProgram(program)
            if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
                log = gl.glGetProgramInfoLog(program).decode("utf-8", "replace")
                raise RuntimeError(f"native display shader link failed: {log}")
            return int(program)
        except Exception:
            if program:
                gl.glDeleteProgram(program)
            raise
        finally:
            for shader in shaders:
                gl.glDeleteShader(shader)

    def _on_resize(self, _window, width: int, height: int) -> None:
        self._gl.glViewport(0, 0, width, height)
        self.width = int(width)
        self.height = int(height)

    def _on_key(self, _window, key: int, _scancode: int, action: int, mods: int) -> None:
        glfw = self._glfw
        if action in (glfw.PRESS, glfw.REPEAT):
            if key == glfw.KEY_F and action == glfw.PRESS:
                self.toggle_fullscreen()
            for handler in self.key_handlers:
                handler(key, action, mods)

    def toggle_fullscreen(self) -> None:
        """Toggle native fullscreen mode on the primary monitor."""
        glfw = self._glfw
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

    def should_close(self) -> bool:
        return not self.window or bool(self._glfw.window_should_close(self.window))

    def poll_events(self) -> None:
        if self.window:
            self._glfw.poll_events()

    def render_frame(self, rgb_image: np.ndarray) -> None:
        """Upload/update one RGB texture and draw it with the core-profile shader."""
        if not self._is_open or not self.window:
            return
        gl, glfw = self._gl, self._glfw
        glfw.make_context_current(self.window)
        frame = np.ascontiguousarray(np.asarray(rgb_image)[..., :3], dtype=np.uint8)
        height, width = frame.shape[:2]
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glClearColor(0.05, 0.05, 0.05, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        if self._texture_size != (width, height):
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
            self._texture_size = (width, height)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, frame)
        self._present_bound_texture()

    def render_texture(self, texture_id: int, width: int, height: int) -> None:
        """Present a texture produced by a GPU renderer sharing this GLFW context."""
        if not self._is_open or not self.window:
            return
        if int(texture_id) <= 0 or int(width) <= 0 or int(height) <= 0:
            raise ValueError("a valid shared texture and positive dimensions are required")
        gl, glfw = self._gl, self._glfw
        glfw.make_context_current(self.window)
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glClearColor(0.05, 0.05, 0.05, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, int(texture_id))
        self._present_bound_texture()

    def _present_bound_texture(self) -> None:
        gl, glfw = self._gl, self._glfw
        gl.glUseProgram(self.program)
        gl.glUniform1i(gl.glGetUniformLocation(self.program, "uFrame"), 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        glfw.swap_buffers(self.window)

    def close(self) -> None:
        if self._gl is None or self._glfw is None:
            return
        gl, glfw = self._gl, self._glfw
        if self.window:
            glfw.make_context_current(self.window)
            if self.texture_id:
                gl.glDeleteTextures(1, [self.texture_id])
                self.texture_id = 0
            if self.program:
                gl.glDeleteProgram(self.program)
                self.program = 0
            if self.vbo:
                gl.glDeleteBuffers(1, [self.vbo])
                self.vbo = 0
            if self.vao:
                gl.glDeleteVertexArrays(1, [self.vao])
                self.vao = 0
            glfw.destroy_window(self.window)
            self.window = None
        if self._glfw_acquired:
            from .glfw_support import release

            release(glfw)
            self._glfw_acquired = False
        self._is_open = False
