"""
Sentiment Prediction API.

Endpoints:
    GET  /health          — service health status
    POST /predict         — single headline sentiment
    POST /predict/batch   — batch headline sentiment
    GET  /metrics         — Prometheus metrics
"""

import json

# (Task 6)
# Configure the Python logger using basicConfig.
import logging
import os
import time
from contextlib import asynccontextmanager
import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from utils import load_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

logger = logging.getLogger("sentiment_api")

# TODO (Task 6)
# Implement a structured logging helper that emits each log entry as a JSON
# object containing at least a timestamp, level, and message field.
def log(level: str, message: str, **kwargs) -> None:
    entry = {
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "level": level,
        "message": message,
        **kwargs,
    }
    logger.log(getattr(logging, level.upper()), json.dumps(entry))


# TODO (Task 6)
# Define the following Prometheus metrics:
# 1. A Counter for the total number of prediction requests, labelled by sentiment.
# 2. A Histogram for prediction latency in milliseconds.
# 3. A Counter for the total number of prediction errors.
# Then instrument run_predictions() to update each metric accordingly.
# Documentation: https://prometheus.io/docs/concepts/metric_types/
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["sentiment"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_ms",
    "Prediction latency in milliseconds",
    buckets=[1, 5, 10, 20, 50, 100, 200, 500, 1000],
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
)


load_dotenv()

classifiers = {}


class PredictRequest(BaseModel):
    text: str


class PredictBatchRequest(BaseModel):
    texts: list[str]


class PredictionResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    log("INFO", "Loading model...")
    classifiers["sentiment"] = load_classifier()
    log("INFO", "Model loaded successfully")
    yield
    classifiers.clear()
    log("INFO", "Model unloaded")


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


def run_predictions_old(texts: list[str]) -> list[PredictionResult]:
    try:
        # (Task 3): Log a warning using log() if any input text exceeds 2000
        # characters — the model will silently truncate it, so this makes it visible.
        # Warn on long inputs
        for t in texts:
            if len(t) > 2000:
                log("WARNING", "Input text exceeds 2000 characters; will be truncated", text_length=len(t))

        # (Task 3): Record the start time, run batch inference using
        # classifiers["sentiment"], and compute latency_ms from start to finish.
        start = time.perf_counter()
        results = classifiers["sentiment"](texts, batch_size=32)
        latency_ms = (time.perf_counter() - start) * 1000
                
        # (Task 3): Build and return a list of PredictionResult objects from
        # the inference results (each result has "label" and "score" keys).
        prediction_results = [
            PredictionResult(
                    text=text,
                    sentiment=r["label"],
                    confidence=float(r["score"]),
                    latency_ms=latency_ms,
                ) for text, r in zip(texts, results)
            ]
        
        # (Task 6): Observe latency_ms on PREDICTION_LATENCY and increment
        # PREDICTION_REQUESTS (labelled by sentiment) for each prediction.
        for r in prediction_results:
            sentiment = r.sentiment
            confidence = r.confidence

             # Prometheus metrics
            PREDICTION_REQUESTS.labels(sentiment=sentiment).inc()
            PREDICTION_LATENCY.observe(latency_ms)
            
            # (Task 6): Log each prediction using log() with sentiment, confidence,
            # and latency_ms fields.
            log(
                "INFO",
                "Prediction completed",
                sentiment=sentiment,
                confidence=confidence,
                latency_ms=latency_ms,
            )

        return prediction_results
    except Exception as e:
        # (Task 6): Increment PREDICTION_ERRORS, log the error using log(),
        # then re-raise.
        print(f"Prediction error: {e!s}")
        PREDICTION_ERRORS.inc()
        log("ERROR", "Prediction error", error=str(e))
        raise

def run_predictions(texts: list[str]) -> list[PredictionResult]:
    try:
        log("INFO", "Running predictions", num_texts=len(texts))

        for t in texts:
            if len(t) > 2000:
                log("WARNING", "Input text exceeds 2000 characters", text_length=len(t))

        start = time.perf_counter()
        results = classifiers["sentiment"](texts, batch_size=32)
        latency_ms = (time.perf_counter() - start) * 1000

        prediction_results = []

        for text, r in zip(texts, results):
            # HuggingFace returns {"label": "...", "score": ...}
            sentiment = r["label"]
            confidence = float(r["score"])

            # Prometheus
            PREDICTION_REQUESTS.labels(sentiment=sentiment).inc()
            PREDICTION_LATENCY.observe(latency_ms)

            log(
                "INFO",
                "Prediction completed",
                sentiment=sentiment,
                confidence=confidence,
                latency_ms=latency_ms,
            )

            prediction_results.append(
                PredictionResult(
                    text=text,
                    sentiment=sentiment,
                    confidence=confidence,
                    latency_ms=latency_ms,
                )
            )

        return prediction_results

    except Exception as e:
        print(f"Prediction error: {e!s}")
        PREDICTION_ERRORS.inc()
        log("ERROR", "Prediction error", error=str(e))
        raise


# (Task 6): Implement the /metrics endpoint.
# Return the Prometheus metrics in the correct format using generate_latest()
# and CONTENT_TYPE_LATEST.
# Documentation: https://prometheus.io/docs/instrumenting/exposition_formats/
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# (Task 6): Implement the /health endpoint.
# Return {"status": "ok"} when the model
# is loaded, and raise an HTTP 503 error when it is not.
@app.get("/health")
def health():
    if "sentiment" not in classifiers:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


# (Task 6): Implement the POST /predict endpoint.
# Accept a PredictRequest and return a PredictionResult.
# Return HTTP 503 if the model is not loaded
# Return HTTP 422 if the text is empty
@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest):
    if "sentiment" not in classifiers:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")

    return run_predictions([request.text])[0]


# Implement the POST /predict/batch endpoint.
# Accept a PredictBatchRequest and return a list of PredictionResult.
# Return HTTP 503 if the model is not loaded
# and HTTP 422 if the texts list is empty.
@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(request: PredictBatchRequest):
    if "sentiment" not in classifiers:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.texts:
        raise HTTPException(status_code=422, detail="Texts list cannot be empty")

    return run_predictions(request.texts)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
    )
