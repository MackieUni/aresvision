import pytest
import sys
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).parent.parent.parent))
from models.detector import AresVisionDetector
from configs.settings import get_model_config


class TestAresVisionDetector:
    def setup_method(self):
        self.detector = AresVisionDetector()

    def test_detector_initializes_correctly(self):
        assert self.detector.model is None
        assert self.detector.device == "cpu"
        assert self.detector.confidence == 0.5
        assert self.detector.iou == 0.45
        assert self.detector.input_size == 640

    def test_detector_loads_model_successfully(self):
        result = self.detector.load_model()
        assert result is True
        assert self.detector.model is not None

    def test_detector_load_time_is_recorded(self):
        self.detector.load_model()
        assert self.detector.load_time_ms > 0

    def test_detector_raises_error_without_model(self):
        with pytest.raises(RuntimeError) as exc_info:
            self.detector.detect("fake_image.jpg")
        assert "Model not loaded" in str(exc_info.value)

    def test_get_system_info_returns_required_keys(self):
        self.detector.load_model()
        info = self.detector.get_system_info()
        required_keys = [
            "cpu_percent",
            "memory_percent",
            "memory_available_gb",
            "device",
            "torch_version",
            "model_loaded",
            "load_time_ms"
        ]
        for key in required_keys:
            assert key in info, f"Missing required key: {key}"

    def test_system_info_device_is_cpu(self):
        self.detector.load_model()
        info = self.detector.get_system_info()
        assert info["device"] == "cpu"

    def test_system_info_memory_is_positive(self):
        self.detector.load_model()
        info = self.detector.get_system_info()
        assert info["memory_available_gb"] > 0

    def test_system_info_model_loaded_flag(self):
        info_before = self.detector.get_system_info()
        assert info_before["model_loaded"] is False
        self.detector.load_model()
        info_after = self.detector.get_system_info()
        assert info_after["model_loaded"] is True

    def test_custom_model_path_is_accepted(self):
        detector = AresVisionDetector(model_path="yolov8n.pt")
        assert detector.model_path == "yolov8n.pt"

    def test_confidence_threshold_from_config(self):
        config = get_model_config()
        assert self.detector.confidence == config["confidence_threshold"]

    def test_iou_threshold_from_config(self):
        config = get_model_config()
        assert self.detector.iou == config["iou_threshold"]

    def test_input_size_from_config(self):
        config = get_model_config()
        assert self.detector.input_size == config["input_size"]


class TestDetectionResults:
    def setup_method(self):
        self.detector = AresVisionDetector()
        self.detector.load_model()
        self.test_image = "/tmp/test.jpg"

    def test_detect_returns_dict(self):
        if not Path(self.test_image).exists():
            pytest.skip("Test image not available")
        result = self.detector.detect(self.test_image)
        assert isinstance(result, dict)

    def test_detect_result_has_required_keys(self):
        if not Path(self.test_image).exists():
            pytest.skip("Test image not available")
        result = self.detector.detect(self.test_image)
        required_keys = [
            "detections", "count",
            "inference_ms", "device", "model"
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_detection_count_matches_list_length(self):
        if not Path(self.test_image).exists():
            pytest.skip("Test image not available")
        result = self.detector.detect(self.test_image)
        assert result["count"] == len(result["detections"])

    def test_inference_time_is_positive(self):
        if not Path(self.test_image).exists():
            pytest.skip("Test image not available")
        result = self.detector.detect(self.test_image)
        assert result["inference_ms"] > 0

    def test_detection_confidence_above_threshold(self):
        if not Path(self.test_image).exists():
            pytest.skip("Test image not available")
        result = self.detector.detect(self.test_image)
        for detection in result["detections"]:
            assert detection["confidence"] >= 0.5

    def test_detection_bbox_has_four_coordinates(self):
        if not Path(self.test_image).exists():
            pytest.skip("Test image not available")
        result = self.detector.detect(self.test_image)
        for detection in result["detections"]:
            assert len(detection["bbox"]) == 4

    def test_detection_class_name_is_string(self):
        if not Path(self.test_image).exists():
            pytest.skip("Test image not available")
        result = self.detector.detect(self.test_image)
        for detection in result["detections"]:
            assert isinstance(detection["class_name"], str)


class TestDetectorEdgeCases:
    def setup_method(self):
        self.detector = AresVisionDetector()

    def test_detect_invalid_image_raises_error(self):
        self.detector.load_model()
        with pytest.raises(Exception):
            self.detector.detect("nonexistent_image.jpg")

    def test_export_onnx_without_model_raises_error(self):
        with pytest.raises(RuntimeError) as exc_info:
            self.detector.export_onnx()
        assert "Model not loaded" in str(exc_info.value)

    def test_load_model_with_explicit_path(self):
        detector = AresVisionDetector(model_path="yolov8n.pt")
        result = detector.load_model()
        assert result is True

    def test_system_info_cpu_percent_in_valid_range(self):
        self.detector.load_model()
        info = self.detector.get_system_info()
        assert 0 <= info["cpu_percent"] <= 100

    def test_system_info_torch_version_is_string(self):
        self.detector.load_model()
        info = self.detector.get_system_info()
        assert isinstance(info["torch_version"], str)
        assert len(info["torch_version"]) > 0

    def test_multiple_detections_on_same_image(self):
        if not Path("/tmp/test.jpg").exists():
            pytest.skip("Test image not available")
        self.detector.load_model()
        result1 = self.detector.detect("/tmp/test.jpg")
        result2 = self.detector.detect("/tmp/test.jpg")
        assert result1["count"] == result2["count"]

    def test_detection_result_device_matches_config(self):
        if not Path("/tmp/test.jpg").exists():
            pytest.skip("Test image not available")
        self.detector.load_model()
        result = self.detector.detect("/tmp/test.jpg")
        assert result["device"] == "cpu"
