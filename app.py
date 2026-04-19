
# app.py — FastAPI prediction API with logging & monitoring

import time
import logging
import psutil
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# ── logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── load model ───────────────────────────────────────────────────────────────
try:
    model = joblib.load("models/iris_model.joblib")
    target_names = np.load("models/target_names.npy", allow_pickle=True)
    log.info("Model loaded successfully")
except Exception as e:
    log.error(f"Failed to load model: {e}")
    model = None

app = FastAPI(
    title="Iris Flower Predictor API",
    description="MLOps Assignment 2 — CI/CD + Docker Deployment",
    version="2.0.0"
)

# ── request counter for monitoring ───────────────────────────────────────────
request_count = 0

# ── input schema ─────────────────────────────────────────────────────────────
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width:  float
    petal_length: float
    petal_width:  float

# ── middleware: log every request with time taken ─────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    global request_count
    request_count += 1
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    log.info(f"[{request.method}] {request.url.path} | {response.status_code} | {duration}ms")
    return response

# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Iris Prediction API is running!",
        "version": "2.0.0",
        "author": "HIMANSHU CHAUHAN",
        "enrollment": "01618012723"
    }

@app.get("/health")
def health():
    """health check endpoint — used in CI/CD pipeline"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/metrics")
def metrics():
    """bonus: system monitoring endpoint"""
    return {
        "total_requests": request_count,
        "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
        "ram_usage_percent": psutil.virtual_memory().percent,
        "ram_used_mb": round(psutil.virtual_memory().used / 1024 / 1024, 1),
    }

@app.post("/predict")
def predict(data: IrisInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    values = [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
    if any(v <= 0 for v in values):
        raise HTTPException(status_code=400, detail="All input values must be positive")

    X = np.array(values).reshape(1, -1)
    prediction = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    confidence = round(float(probs[prediction]) * 100, 2)

    log.info(f"Prediction: {target_names[prediction]} | Confidence: {confidence}%")

    return {
        "predicted_class": str(target_names[prediction]),
        "class_index": int(prediction),
        "confidence": f"{confidence}%",
        "all_probabilities": {
            str(name): round(float(prob) * 100, 2)
            for name, prob in zip(target_names, probs)
        }
    }
