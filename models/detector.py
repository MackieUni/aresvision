# models/detector.py
# AresVision Object Detection Model
# Handles model loading, training, and ONNX export

import torch
import yaml
import time
import psutil
from pathlib import Path
from ultralytics import YOLO
import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.settings import get_model_config, get_data_config, ROOT_DIR


class AresVisionDetector:
    """
    Core object detection engine for AresVision.
    Designed for deployment on air-gapped defense networks.
    Supports CPU inference with ONNX export for edge deployment.
    """

    def __init__(self, model_path: str = None):
        self.config = get_model_config()
        self.data_config = get_data_config()
        self.device = self.config["device"]
        self.confidence = self.config["confidence_threshold"]
        self.iou = self.config["iou_threshold"]
        self.input_size = self.config["input_size"]
        self.model = None
        self.model_path = model_path or self.config["name"]
        self.load_time_ms = 0.0

    def load_model(self) -> bool:
        """
        Load YOLOv8 model.
        Returns True if successful.
        """
        try:
            print(f"Loading model: {self.model_path}")
            start = time.time()
            self.model = YOLO(self.model_path)
            self.load_time_ms = (time.time() - start) * 1000
            print(f"Model loaded in {self.load_time_ms:.2f}ms")
            return True
        except Exception as e:
            print(f"ERROR loading model: {e}")
            return False

    def detect(self, image_path: str) -> dict:
        """
        Run object detection on a single image.
        Returns detection results with metadata.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start = time.time()

        results = self.model(
            image_path,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.input_size,
            device=self.device,
            verbose=False
        )

        inference_ms = (time.time() - start) * 1000

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    detections.append({
                        "class_id": int(box.cls[0]),
                        "class_name": result.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist()
                    })

        return {
            "detections": detections,
            "count": len(detections),
            "inference_ms": round(inference_ms, 2),
            "device": self.device,
            "model": self.model_path
        }

    def export_onnx(self, output_path: str = None) -> str:
        """
        Export model to ONNX format for edge deployment.
        Critical for air-gapped network deployment.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        output_path = output_path or str(
            ROOT_DIR / "models" / "aresvision.onnx"
        )

        print(f"Exporting model to ONNX: {output_path}")
        self.model.export(
            format="onnx",
            imgsz=self.input_size,
            simplify=True
        )
        print(f"ONNX export complete: {output_path}")
        return output_path

    def get_system_info(self) -> dict:
        """
        Return system resource information.
        Used for performance monitoring on edge devices.
        """
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": round(
                psutil.virtual_memory().available / (1024**3), 2
            ),
            "device": self.device,
            "torch_version": torch.__version__,
            "model_loaded": self.model is not None,
            "load_time_ms": round(self.load_time_ms, 2)
        }


if __name__ == "__main__":
    print("Initializing AresVision Detector...")
    detector = AresVisionDetector()

    success = detector.load_model()
    if success:
        print("\nSystem Info:")
        info = detector.get_system_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        print("\nDetector ready for inference!")
    else:
        print("Failed to load model!")
