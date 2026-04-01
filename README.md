# AresVision — Defense AI Computer Vision System

Real-time object detection system designed for
Intelligence, Surveillance, and Reconnaissance (ISR)
applications. Built for deployment on air-gapped,
classified defense networks using CPU-optimized
ONNX inference.

## Defense Context

Modern ISR platforms require AI inference systems
that operate in contested, resource-constrained
environments without internet connectivity. AresVision
addresses this by combining YOLOv8 detection with
ONNX Runtime for deterministic, hardware-agnostic
inference deployable on classified networks.

## Architecture
```
Sensor Input → Preprocessing → YOLOv8 Detection
→ ONNX Inference Engine → FastAPI Microservice
→ Prometheus Monitoring → Kubernetes Deployment
```

## Performance Benchmarks

| Platform | Inference | Throughput |
|---|---|---|
| Intel i3 CPU | ~1984ms | ~0.5 FPS |
| NVIDIA T4 GPU* | ~12ms | ~83 FPS |
| ONNX CPU optimized | ~800ms | ~1.2 FPS |

*GPU benchmarks validated on Google Colab T4

## Tech Stack

- PyTorch 2.2.2 + YOLOv8
- ONNX Runtime 1.23.2
- FastAPI + Uvicorn
- Prometheus metrics
- Kubernetes + Helm ready
- Docker containerized

## Quick Start
```bash
# Clone repository
git clone https://github.com/MackieUni/aresvision.git
cd aresvision

# Install dependencies
pip install -r requirements.txt

# Run API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Test detection
curl -X POST http://localhost:8000/detect \
  -F "file=@your_image.jpg"
```

## Test Suite
```bash
# Run full TDD test suite
pytest tests/unit/ -v --cov=models --cov-report=term-missing
```

Results: 26 tests passing | 72% coverage

## Security Architecture

Designed for classified network deployment:
- No external network calls during inference
- ONNX model runs fully air-gapped
- All dependencies pinned in requirements.txt
- Kubernetes namespace isolation ready
- Prometheus metrics for audit logging

## How I Led This

As a former CEO who managed operations across
4 countries, I applied the same systems thinking
to this architecture — designing for reliability,
observability, and deployment in constrained
environments where failure is not an option.

## Author

Inmaculada Rondon
M.S. Artificial Intelligence 2026
GitHub: MackieUni
