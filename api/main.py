# api/main.py
# AresVision FastAPI Microservice
# REST API for real-time object detection
# Designed for deployment on classified defense networks

import time
import psutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
from starlette.responses import Response
import shutil
import tempfile
import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.settings import get_model_config, get_api_config
from models.detector import AresVisionDetector

# --- Prometheus Metrics ---
REQUEST_COUNT = Counter(
    "aresvision_requests_total",
    "Total number of detection requests",
    ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "aresvision_request_latency_ms",
    "Request latency in milliseconds",
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500]
)
DETECTION_COUNT = Counter(
    "aresvision_detections_total",
    "Total number of objects detected"
)
MEMORY_USAGE = Gauge(
    "aresvision_memory_percent",
    "Current memory usage percentage"
)

# --- App Initialization ---
app = FastAPI(
    title="AresVision API",
    description="""
    Real-time AI object detection for ISR applications.
    Designed for deployment on air-gapped defense networks.
    Supports edge deployment via ONNX inference engine.
    """,
    version="1.0.0",
)

# Load detector on startup
detector = AresVisionDetector()
model_loaded = False


@app.on_event("startup")
async def startup_event():
    """Load model when API starts."""
    global model_loaded
    print("AresVision API starting...")
    model_loaded = detector.load_model()
    if model_loaded:
        print("Model loaded successfully!")
    else:
        print("WARNING: Model failed to load!")


# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """
    System health check endpoint.
    Used by Kubernetes liveness and readiness probes.
    """
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": round(
                psutil.virtual_memory().available / (1024**3), 2
            )
        },
        "version": "1.0.0"
    }


# --- Detection Endpoint ---
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Run object detection on uploaded image.
    Returns detections with confidence scores and bounding boxes.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )

    # Validate file type
    if not file.content_type.startswith("image/"):
        REQUEST_COUNT.labels(
            endpoint="detect", status="error"
        ).inc()
        raise HTTPException(
            status_code=400,
            detail="File must be an image."
        )

    start = time.time()

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(file.filename).suffix
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Run detection
        results = detector.detect(tmp_path)
        latency_ms = (time.time() - start) * 1000

        # Update metrics
        REQUEST_COUNT.labels(
            endpoint="detect", status="success"
        ).inc()
        REQUEST_LATENCY.observe(latency_ms)
        DETECTION_COUNT.inc(results["count"])

        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "detections": results["detections"],
            "count": results["count"],
            "performance": {
                "inference_ms": results["inference_ms"],
                "total_ms": round(latency_ms, 2)
            }
        })

    except Exception as e:
        REQUEST_COUNT.labels(
            endpoint="detect", status="error"
        ).inc()
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


# --- System Info Endpoint ---
@app.get("/info")
async def system_info():
    """
    Return system and model information.
    Used for deployment verification on classified networks.
    """
    return {
        "project": "AresVision",
        "version": "1.0.0",
        "description": "Defense AI Computer Vision System",
        "model": get_model_config(),
        "system": detector.get_system_info()
    }


# --- Prometheus Metrics Endpoint ---
@app.get("/metrics")
async def metrics():
    """
    Expose Prometheus metrics for monitoring.
    Scraped by Prometheus server in Kubernetes deployment.
    """
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    import uvicorn
    config = get_api_config()
    uvicorn.run(
        "api.main:app",
        host=config["host"],
        port=config["port"],
        reload=False,
        workers=config["workers"]
    )
