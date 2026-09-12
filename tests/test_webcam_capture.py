"""Webcam latest-frame mailbox and worker tests without a physical camera."""

import time
import unittest

import numpy as np

from renderer.capture import LatestFrameBuffer, WebcamCaptureWorker


class FakeCV2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    COLOR_BGR2RGB = 6

    @staticmethod
    def cvtColor(frame, _conversion):
        return np.ascontiguousarray(frame[:, :, ::-1])


class FakeCapture:
    def __init__(self, _camera_index):
        self.released = False
        self.frame_id = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            FakeCV2.CAP_PROP_FRAME_WIDTH: 4,
            FakeCV2.CAP_PROP_FRAME_HEIGHT: 3,
            FakeCV2.CAP_PROP_FPS: 30.0,
        }[prop]

    def read(self):
        time.sleep(0.002)
        self.frame_id += 1
        # BGR values change over time so the latest-only worker can be observed.
        frame = np.empty((3, 4, 3), dtype=np.uint8)
        frame[:, :] = [self.frame_id % 255, 20, 200]
        return True, frame

    def release(self):
        self.released = True


class LatestFrameBufferTests(unittest.TestCase):
    def test_keeps_only_latest_reference_and_counts_unconsumed_replacements(self) -> None:
        mailbox = LatestFrameBuffer()
        first_rgb = np.full((2, 3, 3), 1, dtype=np.uint8)
        second_rgb = np.full((2, 3, 3), 2, dtype=np.uint8)
        third_rgb = np.full((2, 3, 3), 3, dtype=np.uint8)

        first = mailbox.publish(first_rgb, 1.0)
        self.assertIs(mailbox.get_latest(), first)
        mailbox.publish(second_rgb, 2.0)
        mailbox.publish(third_rgb, 3.0)

        latest = mailbox.get_latest()
        self.assertEqual(latest.sequence, 3)
        self.assertIs(latest.rgb, third_rgb)
        self.assertEqual(int(latest.rgb[0, 0, 0]), 3)
        stats = mailbox.stats()
        self.assertEqual(stats.frames_captured, 3)
        self.assertEqual(stats.frames_replaced, 1)
        self.assertEqual(stats.latest_sequence, 3)

    def test_validates_rgb_array_contract(self) -> None:
        mailbox = LatestFrameBuffer()
        with self.assertRaisesRegex(TypeError, "uint8"):
            mailbox.publish(np.zeros((2, 2, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "HxWx3"):
            mailbox.publish(np.zeros((2, 2), dtype=np.uint8))


class WebcamCaptureWorkerTests(unittest.TestCase):
    def test_worker_publishes_rgb_and_releases_camera_after_join(self) -> None:
        captures = []

        def make_capture(index):
            capture = FakeCapture(index)
            captures.append(capture)
            return capture

        worker = WebcamCaptureWorker(
            0,
            capture_factory=make_capture,
            cv2_module=FakeCV2(),
        ).start()
        try:
            first = worker.frames.get_latest()
            self.assertIsNotNone(first)
            np.testing.assert_array_equal(first.rgb[0, 0], [200, 20, 1])
            time.sleep(0.02)
            stats = worker.frames.stats()
            self.assertGreater(stats.frames_captured, 1)
            self.assertGreaterEqual(stats.frames_replaced, 1)
            self.assertEqual(worker.reported_fps, 30.0)
            self.assertEqual(worker.received_width, 4)
            self.assertEqual(worker.received_height, 3)
        finally:
            worker.stop()

        self.assertTrue(worker.released)
        self.assertTrue(captures[0].released)
        self.assertFalse(worker._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
