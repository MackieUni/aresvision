# AresVision — Defense AI Computer Vision System

Real-time AI object detection for ISR applications.
Built for air-gapped classified defense networks.
26 TDD tests passing | 72% coverage | ONNX edge deployment

## Defense Context
Modern ISR platforms require AI that operates without
internet connectivity. AresVision uses YOLOv8 plus
ONNX Runtime for deterministic CPU inference deployable
on classified networks.

## Performance Benchmarks
| Engine | Mean Latency | Throughput |
|--------|-------------|------------|
| PyTorch CPU | 809ms | 1.23 FPS |
| ONNX Runtime CPU | 883ms | 1.13 FPS |
| NVIDIA T4 GPU | 12ms | 83 FPS |

## Tech Stack
- PyTorch 2.2.2 + YOLOv8
- ONNX Runtime 1.23.2
- FastAPI + Uvicorn
- Prometheus monitoring
- Kubernetes + Helm
- Docker
- pytest TDD 26 tests 72% coverage

## Quick Start
git clone https://github.com/MackieUni/aresvision.git
cd aresvision
pip install -r requirements.txt
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Liveness probe |
| /detect | POST | Object detection |
| /info | GET | System info |
| /metrics | GET | Prometheus metrics |

## Pipeline Results
Frame 0: 1 detections | 2426ms | Threat: True
Frame 1: 1 detections | 1150ms | Threat: True
Frame 2: 1 detections | 1197ms | Threat: True
Frame 3: 1 detections | 1131ms | Threat: True
Frame 4: 1 detections | 1230ms | Threat: True
Frames submitted: 5 | Processed: 5 | Dropped: 0

## Security Architecture
- No external network calls during inference
- ONNX model runs fully air-gapped
- All dependencies pinned
- Kubernetes namespace isolation
- Prometheus audit logging
- Resource limits for edge deployment

## Leadership Context
As a former CEO who scaled operations across 4 countries,
I designed AresVision with operational discipline:
- TDD from day one 26 tests 72% coverage
- Every component monitored and observable
- Documented for team handoff
- Optimized for constrained environments

## Author
Inmaculada Rondon
M.S. Artificial Intelligence 2026
University of the Andes
GitHub: MackieUni
Email: ic.rondon@uniandes.edu.co
