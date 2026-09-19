import importlib
import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The unit tests exercise preprocessing and postprocessing only. Hardware and
# runtime modules are replaced before importing the application module.
picamera2 = types.ModuleType("picamera2")
picamera2.Picamera2 = object
sys.modules.setdefault("picamera2", picamera2)

onnxruntime = types.ModuleType("onnxruntime")
onnxruntime.SessionOptions = object
onnxruntime.InferenceSession = object
sys.modules.setdefault("onnxruntime", onnxruntime)

posture_detection = importlib.import_module("posture_detection")
beep = importlib.import_module("beep")


class PoseDetectorLogicTest(unittest.TestCase):
    def setUp(self):
        self.detector = posture_detection.PoseDetector.__new__(posture_detection.PoseDetector)
        self.detector.input_height = 640
        self.detector.input_width = 480
        self.detector.conf_threshold = 0.5

    def test_preprocess_handles_odd_padding(self):
        image = np.zeros((333, 777, 3), dtype=np.uint8)

        processed, _, _, _ = self.detector._preprocess(image)

        self.assertEqual(processed.shape, (1, 3, 640, 480))

    def test_postprocess_keeps_keypoints_from_one_person(self):
        output = np.zeros((1, 56, 2), dtype=np.float32)
        output[0, 4, :] = [0.9, 0.8]
        for keypoint_index in (0, 6, 12, 14):
            base = 5 + keypoint_index * 3
            output[0, base : base + 3, 0] = [100 + keypoint_index, 200 + keypoint_index, 0.7]
            output[0, base : base + 3, 1] = [300 + keypoint_index, 400 + keypoint_index, 0.99]

        keypoints = self.detector._postprocess(output, 1.0, 0, 0, 640, 480)

        self.assertEqual(len(keypoints), 4)
        self.assertTrue(all(point["x"] < 200 for point in keypoints))


class PostureAlerterLogicTest(unittest.TestCase):
    def test_missing_keypoint_is_unknown_and_stops_timer(self):
        alerter = beep.PostureAlerter(orange_duration=0.0, red_duration=0.0)
        alerter.update(30, 0, 0)
        self.assertTrue(alerter.is_alerting)

        alerter.update(None, 0, 0)

        self.assertEqual(alerter.metrics["head"]["state"], "unknown")
        self.assertFalse(alerter.is_alerting)
        alerter.cleanup()


if __name__ == "__main__":
    unittest.main()
