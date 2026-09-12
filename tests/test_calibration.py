from unittest.mock import MagicMock
import numpy as np
import pytest

from geometry.calibration import calibrate_checkerboard


def test_mixed_calibration_image_resolutions_rejected(tmp_path) -> None:
    cv2 = pytest.importorskip("cv2")
    # Create two dummy images of different resolutions
    img1 = np.zeros((100, 120), dtype=np.uint8)  # height=100, width=120
    img2 = np.zeros((150, 180), dtype=np.uint8)  # height=150, width=180

    p1 = tmp_path / "calib_01.png"
    p2 = tmp_path / "calib_02.png"
    cv2.imwrite(str(p1), img1)
    cv2.imwrite(str(p2), img2)

    with pytest.raises(ValueError) as exc_info:
        calibrate_checkerboard([p1, p2], board_size=(7, 5))

    msg = str(exc_info.value)
    # Check that error includes expected and received dimensions
    assert "width=180" in msg and "height=150" in msg
    assert "width=120" in msg and "height=100" in msg
    assert str(p2) in msg


def test_mixed_calibration_resolutions_mocked(monkeypatch) -> None:
    # Test resolution consistency logic independently of cv2 installation
    mock_cv2 = MagicMock()
    mock_cv2.findChessboardCorners.return_value = (False, None)
    img1 = np.zeros((100, 120), dtype=np.uint8)  # H=100, W=120
    img2 = np.zeros((150, 180), dtype=np.uint8)  # H=150, W=180

    mock_cv2.imread.side_effect = [img1, img2]
    monkeypatch.setattr("geometry.calibration._cv2", lambda: mock_cv2)

    with pytest.raises(ValueError) as exc_info:
        calibrate_checkerboard(["fake1.png", "fake2.png"])

    msg = str(exc_info.value)
    assert "width=180" in msg and "height=150" in msg
    assert "width=120" in msg and "height=100" in msg


def test_matching_calibration_resolutions_accepted(monkeypatch) -> None:
    # Matching resolutions must not raise resolution mismatch error
    mock_cv2 = MagicMock()
    mock_cv2.findChessboardCorners.return_value = (False, None)
    img1 = np.zeros((100, 120), dtype=np.uint8)
    img2 = np.zeros((100, 120), dtype=np.uint8)

    mock_cv2.imread.side_effect = [img1, img2]
    monkeypatch.setattr("geometry.calibration._cv2", lambda: mock_cv2)

    with pytest.raises(ValueError, match="no checkerboard views were detected"):
        calibrate_checkerboard(["img1.png", "img2.png"])


def test_calibration_unreadable_image_rejected(tmp_path) -> None:
    cv2 = pytest.importorskip("cv2")
    non_existent = tmp_path / "does_not_exist.png"
    with pytest.raises(ValueError, match="could not read calibration image"):
        calibrate_checkerboard([non_existent])


def test_calibration_unreadable_image_mocked(monkeypatch) -> None:
    mock_cv2 = MagicMock()
    mock_cv2.imread.return_value = None
    monkeypatch.setattr("geometry.calibration._cv2", lambda: mock_cv2)
    with pytest.raises(ValueError, match="could not read calibration image"):
        calibrate_checkerboard(["missing.png"])


def test_calibration_invalid_board_or_square_size(monkeypatch) -> None:
    mock_cv2 = MagicMock()
    # Dummy image of 100x120
    img = np.zeros((100, 120), dtype=np.uint8)
    mock_cv2.imread.return_value = img
    monkeypatch.setattr("geometry.calibration._cv2", lambda: mock_cv2)

    with pytest.raises(ValueError, match="square_size must be positive"):
        calibrate_checkerboard(["img.png"], square_size=0.0)

    with pytest.raises(ValueError, match="at least 2 corners"):
        calibrate_checkerboard(["img.png"], board_size=(1, 5))


def test_calibration_missing_cv2_raises_runtime_error(monkeypatch) -> None:
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("no cv2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from geometry.calibration import _cv2
    with pytest.raises(RuntimeError, match="OpenCV is required for calibration"):
        _cv2()
