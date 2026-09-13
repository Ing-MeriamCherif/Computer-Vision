from __future__ import annotations

from types import SimpleNamespace

from geometry import glfw_support


def test_glfw_process_lifetime_is_reference_counted(monkeypatch):
    calls = []
    monkeypatch.setattr(glfw_support, "_users", 0)
    glfw = SimpleNamespace(
        init=lambda: calls.append("init") or True,
        terminate=lambda: calls.append("terminate"),
    )

    assert glfw_support.acquire(glfw)
    assert glfw_support.acquire(glfw)
    assert calls == ["init"]
    glfw_support.release(glfw)
    assert calls == ["init"]
    glfw_support.release(glfw)
    assert calls == ["init", "terminate"]


def test_native_window_close_is_idempotent_before_initialization():
    from geometry.native_window import NativeOpenGLWindow

    window = NativeOpenGLWindow.__new__(NativeOpenGLWindow)
    window._gl = None
    window._glfw = None
    window.close()
    window.close()
