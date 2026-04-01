# inference/engine_onnx.py
# AresVision ONNX Inference Engine
# Optimized for CPU deployment on air-gapped edge devices
# This engine runs on ANY hardware without GPU requirements

import onnxruntime as ort
import numpy as np
import cv2
import time
import psutil
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.settings import get_model_config, get_inference_config


class ONNXInferenceEngine:
    """
    ONNX-based inference engine for edge deployment.
    Designed for classified network environments where
    internet access and GPU hardware are not available.
    Runs deterministically on CPU with predictable latency.
    """

    def __init__(self, model_path: str = None):
        self.model_config = get_model_config()
        self.inference_config = get_inference_config()
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_size = self.model_config["input_size"]
        self.confidence = self.model_config["confidence_threshold"]
        self.iou = self.model_config["iou_threshold"]
        self.latency_history = []

    def load(self) -> bool:
        """
        Load ONNX model and configure inference session.
        Optimized for CPU execution on edge hardware.
        """
        try:
            if not self.model_path or not Path(self.model_path).exists():
                print(f"ONNX model not found: {self.model_path}")
                print("Run export_onnx() from detector.py first.")
                return False

            print(f"Loading ONNX engine: {self.model_path}")

            # Configure session for CPU optimization
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.inference_config[
                "num_threads"
            ]
            opts.inter_op_num_threads = self.inference_config[
                "num_threads"
            ]
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            opts.enable_profiling = self.inference_config[
                "enable_profiling"
            ]

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"]
            )

            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [
                o.name for o in self.session.get_outputs()
            ]

            print(f"ONNX engine ready!")
            print(f"Input: {self.input_name}")
            print(f"Outputs: {self.output_names}")
            return True

        except Exception as e:
            print(f"ERROR loading ONNX engine: {e}")
            return False

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ONNX inference.
        Simulates what a CUDA kernel would do on GPU hardware.
        """
        # Resize to model input size
        resized = cv2.resize(
            image,
            (self.input_size, self.input_size)
        )

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Transpose to CHW format (channels first)
        chw = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(chw, axis=0)

        return batched

    def postprocess(
        self,
        outputs: list,
        orig_shape: tuple
    ) -> list:
        """
        Post-process raw ONNX outputs into detections.
        """
        detections = []
        predictions = outputs[0]

        if len(predictions.shape) == 3:
            predictions = predictions[0]

        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        for pred in predictions.T:
            x, y, w, h = pred[:4]
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < self.confidence:
                continue

            x1 = int((x - w / 2) * scale_x)
            y1 = int((y - h / 2) * scale_y)
            x2 = int((x + w / 2) * scale_x)
            y2 = int((y + h / 2) * scale_y)

            detections.append({
                "class_id": class_id,
                "confidence": round(confidence, 4),
                "bbox": [x1, y1, x2, y2]
            })

        return detections

    def infer(self, image_path: str) -> dict:
        """
        Run full inference pipeline on an image file.
        Returns detections with performance metrics.
        """
        if self.session is None:
            raise RuntimeError(
                "Engine not loaded. Call load() first."
            )

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        orig_shape = image.shape

        # Preprocess
        t0 = time.time()
        input_tensor = self.preprocess(image)
        preprocess_ms = (time.time() - t0) * 1000

        # Inference
        t1 = time.time()
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor}
        )
        inference_ms = (time.time() - t1) * 1000

        # Postprocess
        t2 = time.time()
        detections = self.postprocess(outputs, orig_shape)
        postprocess_ms = (time.time() - t2) * 1000

        total_ms = preprocess_ms + inference_ms + postprocess_ms
        self.latency_history.append(total_ms)

        return {
            "detections": detections,
            "count": len(detections),
            "performance": {
                "preprocess_ms": round(preprocess_ms, 2),
                "inference_ms": round(inference_ms, 2),
                "postprocess_ms": round(postprocess_ms, 2),
                "total_ms": round(total_ms, 2)
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_available_gb": round(
                    psutil.virtual_memory().available / (1024**3), 2
                )
            }
        }

    def get_performance_stats(self) -> dict:
        """
        Return latency statistics over all inferences.
        Used for benchmarking and SLA validation.
        """
        if not self.latency_history:
            return {"message": "No inferences run yet"}

        arr = np.array(self.latency_history)
        return {
            "total_inferences": len(arr),
            "mean_latency_ms": round(float(np.mean(arr)), 2),
            "min_latency_ms": round(float(np.min(arr)), 2),
            "max_latency_ms": round(float(np.max(arr)), 2),
            "p95_latency_ms": round(
                float(np.percentile(arr, 95)), 2
            ),
            "p99_latency_ms": round(
                float(np.percentile(arr, 99)), 2
            )
        }


if __name__ == "__main__":
    engine = ONNXInferenceEngine(
        model_path="models/aresvision.onnx"
    )
    loaded = engine.load()
    if loaded:
        print("ONNX engine ready for edge deployment!")
    else:
        print("Export ONNX model first using detector.py")
