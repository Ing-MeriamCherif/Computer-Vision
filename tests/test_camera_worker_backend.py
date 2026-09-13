from types import SimpleNamespace

import geometry.camera_worker as camera_worker


def test_default_camera_backend_matches_platform(monkeypatch):
    cv2 = SimpleNamespace(CAP_MSMF=1400, CAP_DSHOW=700, CAP_V4L2=200, CAP_ANY=0)

    monkeypatch.setattr(camera_worker.sys, "platform", "win32")
    assert camera_worker._default_camera_backend(cv2) == cv2.CAP_MSMF

    monkeypatch.setattr(camera_worker.sys, "platform", "linux")
    assert camera_worker._default_camera_backend(cv2) == cv2.CAP_V4L2

    monkeypatch.setattr(camera_worker.sys, "platform", "darwin")
    assert camera_worker._default_camera_backend(cv2) == cv2.CAP_ANY
