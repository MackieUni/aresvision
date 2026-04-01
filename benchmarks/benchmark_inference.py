# benchmarks/benchmark_inference.py
# AresVision Performance Benchmark Suite
# Measures inference latency, throughput, and system resources
# Results used to validate deployment readiness

import time
import sys
import json
import psutil
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from models.detector import AresVisionDetector
from inference.engine_onnx import ONNXInferenceEngine
from configs.settings import get_benchmark_config, ROOT_DIR


class AresVisionBenchmark:
    """
    Performance benchmark suite for AresVision.
    Validates system meets latency and throughput
    requirements for defense deployment.
    """

    def __init__(self):
        self.config = get_benchmark_config()
        self.results = {}
        self.test_image = "/tmp/test.jpg"
        self.timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

    def _check_test_image(self) -> bool:
        """Verify test image exists."""
        if not Path(self.test_image).exists():
            print(f"Test image not found: {self.test_image}")
            print("Run: curl -L -o /tmp/test.jpg "
                  "https://images.pexels.com/"
                  "photos/170811/pexels-photo-170811.jpeg")
            return False
        return True

    def benchmark_pytorch(self) -> dict:
        """
        Benchmark PyTorch detector performance.
        Measures load time, warmup, and steady state latency.
        """
        print("\nBenchmarking PyTorch detector...")
        detector = AresVisionDetector()

        # Measure load time
        t0 = time.time()
        detector.load_model()
        load_ms = (time.time() - t0) * 1000

        warmup = self.config["warmup_iterations"]
        iterations = self.config["test_iterations"]
        latencies = []

        # Warmup runs
        print(f"  Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            detector.detect(self.test_image)

        # Benchmark runs
        print(f"  Benchmarking ({iterations} iterations)...")
        for i in range(iterations):
            t = time.time()
            detector.detect(self.test_image)
            latencies.append((time.time() - t) * 1000)
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{iterations}")

        arr = np.array(latencies)
        results = {
            "engine": "PyTorch CPU",
            "load_time_ms": round(load_ms, 2),
            "iterations": iterations,
            "mean_ms": round(float(np.mean(arr)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "fps": round(1000 / float(np.mean(arr)), 2),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }

        print(f"  Mean latency: {results['mean_ms']}ms")
        print(f"  Throughput: {results['fps']} FPS")
        return results

    def benchmark_onnx(self) -> dict:
        """
        Benchmark ONNX Runtime inference performance.
        Validates edge deployment performance targets.
        """
        print("\nBenchmarking ONNX Runtime...")
        onnx_path = str(ROOT_DIR / "models" / "aresvision.onnx")
        engine = ONNXInferenceEngine(model_path=onnx_path)

        if not engine.load():
            return {"error": "ONNX model not found"}

        warmup = self.config["warmup_iterations"]
        iterations = self.config["test_iterations"]
        latencies = []

        # Warmup runs
        print(f"  Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            engine.infer(self.test_image)

        # Benchmark runs
        print(f"  Benchmarking ({iterations} iterations)...")
        for i in range(iterations):
            result = engine.infer(self.test_image)
            latencies.append(
                result["performance"]["total_ms"]
            )
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{iterations}")

        arr = np.array(latencies)
        results = {
            "engine": "ONNX Runtime CPU",
            "iterations": iterations,
            "mean_ms": round(float(np.mean(arr)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "fps": round(1000 / float(np.mean(arr)), 2),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }

        print(f"  Mean latency: {results['mean_ms']}ms")
        print(f"  Throughput: {results['fps']} FPS")
        return results

    def benchmark_system(self) -> dict:
        """
        Capture system resource baseline.
        Documents hardware specs for benchmark context.
        """
        print("\nCapturing system info...")
        vm = psutil.virtual_memory()
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_mhz": psutil.cpu_freq().current
            if psutil.cpu_freq() else "N/A",
            "total_memory_gb": round(vm.total / (1024**3), 2),
            "available_memory_gb": round(
                vm.available / (1024**3), 2
            ),
            "python_version": sys.version,
            "platform": sys.platform
        }

    def run_all(self) -> dict:
        """
        Run complete benchmark suite.
        Saves results to JSON file for README reporting.
        """
        print("=" * 50)
        print("AresVision Benchmark Suite")
        print(f"Timestamp: {self.timestamp}")
        print("=" * 50)

        if not self._check_test_image():
            return {}

        self.results = {
            "timestamp": self.timestamp,
            "system": self.benchmark_system(),
            "pytorch": self.benchmark_pytorch(),
            "onnx": self.benchmark_onnx()
        }

        # Save results
        results_dir = ROOT_DIR / "benchmarks" / "results"
        results_dir.mkdir(exist_ok=True)
        output_file = results_dir / f"benchmark_{self.timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        self._print_summary()
        print(f"\nResults saved: {output_file}")
        return self.results

    def _print_summary(self):
        """Print formatted benchmark summary."""
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)

        engines = ["pytorch", "onnx"]
        for engine in engines:
            if engine in self.results:
                r = self.results[engine]
                if "error" not in r:
                    print(f"\n{r.get('engine', engine)}:")
                    print(f"  Mean latency : {r['mean_ms']}ms")
                    print(f"  P95 latency  : {r['p95_ms']}ms")
                    print(f"  P99 latency  : {r['p99_ms']}ms")
                    print(f"  Throughput   : {r['fps']} FPS")
                    print(f"  CPU usage    : {r['cpu_percent']}%")


if __name__ == "__main__":
    benchmark = AresVisionBenchmark()
    benchmark.run_all()
