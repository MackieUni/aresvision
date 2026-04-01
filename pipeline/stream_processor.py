import time
import sys
import queue
import threading
import psutil
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from models.detector import AresVisionDetector

class DetectionResult:
    def __init__(self, frame_id, detections, inference_ms, timestamp):
        self.frame_id = frame_id
        self.detections = detections
        self.inference_ms = inference_ms
        self.timestamp = timestamp
        self.threat_detected = len(detections) > 0

class StreamProcessor:
    def __init__(self, max_queue_size=10):
        self.detector = AresVisionDetector()
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        self.is_running = False
        self.frame_count = 0
        self.processed_count = 0
        self.dropped_count = 0
        self.start_time = None
        self.processing_thread = None
        self.stats_lock = threading.Lock()
        self.latency_history = []

    def initialize(self):
        print("Initializing AresVision stream processor...")
        success = self.detector.load_model()
        if success:
            print("Stream processor ready!")
            return True
        return False

    def submit_frame(self, image_path):
        try:
            self.input_queue.put_nowait(image_path)
            with self.stats_lock:
                self.frame_count += 1
            return True
        except queue.Full:
            with self.stats_lock:
                self.dropped_count += 1
            return False

    def _process_frame(self, image_path):
        timestamp = datetime.now().isoformat()
        result = self.detector.detect(image_path)
        return DetectionResult(
            frame_id=self.processed_count,
            detections=result["detections"],
            inference_ms=result["inference_ms"],
            timestamp=timestamp
        )

    def _processing_loop(self):
        print("Processing loop started...")
        while self.is_running:
            try:
                image_path = self.input_queue.get(timeout=1.0)
                result = self._process_frame(image_path)
                with self.stats_lock:
                    self.processed_count += 1
                    self.latency_history.append(result.inference_ms)
                try:
                    self.output_queue.put_nowait(result)
                except queue.Full:
                    pass
                self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print("Processing error:", e)
                continue
        print("Processing loop stopped.")

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.start_time = time.time()
            self.processing_thread = threading.Thread(
                target=self._processing_loop, daemon=True
            )
            self.processing_thread.start()
            print("Stream processor started!")

    def stop(self):
        print("Stopping stream processor...")
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        print("Stream processor stopped!")

    def get_result(self, timeout=5.0):
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_pipeline_stats(self):
        with self.stats_lock:
            elapsed = time.time() - self.start_time if self.start_time else 0
            throughput = self.processed_count / elapsed if elapsed > 0 else 0
            mean_lat = round(sum(self.latency_history) / len(self.latency_history), 2) if self.latency_history else 0
            return {
                "frames_submitted": self.frame_count,
                "frames_processed": self.processed_count,
                "frames_dropped": self.dropped_count,
                "throughput_fps": round(throughput, 3),
                "mean_latency_ms": mean_lat,
                "elapsed_seconds": round(elapsed, 2),
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "is_running": self.is_running
            }

if __name__ == "__main__":
    processor = StreamProcessor(max_queue_size=5)
    if not processor.initialize():
        sys.exit(1)
    processor.start()
    time.sleep(1)
    test_image = "/tmp/test.jpg"
    print("Simulating ISR sensor feed...")
    for i in range(5):
        submitted = processor.submit_frame(test_image)
        status = "submitted" if submitted else "DROPPED"
        print("Frame", i+1, ":", status)
        time.sleep(0.5)
    print("Collecting results...")
    for i in range(5):
        result = processor.get_result(timeout=60.0)
        if result:
            print("Frame", result.frame_id, ":",
                  len(result.detections), "detections |",
                  result.inference_ms, "ms | Threat:",
                  result.threat_detected)
        else:
            print("Frame", i+1, ": timeout")
    stats = processor.get_pipeline_stats()
    print("Pipeline Stats:")
    for key, value in stats.items():
        print(" ", key, ":", value)
    processor.stop()
