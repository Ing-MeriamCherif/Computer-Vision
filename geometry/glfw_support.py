"""Process-wide GLFW lifetime shared by native and offscreen GL windows."""

from __future__ import annotations

import threading

_lock = threading.Lock()
_users = 0


def acquire(glfw) -> bool:
    global _users
    with _lock:
        if _users == 0 and not glfw.init():
            return False
        _users += 1
        return True


def release(glfw) -> None:
    global _users
    with _lock:
        if _users <= 0:
            return
        _users -= 1
        if _users == 0:
            glfw.terminate()
