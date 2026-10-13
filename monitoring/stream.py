"""
Production monitoring stream for FinBERT sentiment API.

Reads headlines from data/stream.csv, sends them to the /predict endpoint,
and logs aggregated metrics to MLflow every WINDOW_SIZE predictions.

Each observation window logs:
    - Sentiment distribution (% positive, % negative, % neutral)
    - Average confidence score
    - Average latency (ms)

Run from the project root:
    python monitoring/stream.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', 8000)}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation")
WINDOW_SIZE = 50  # number of predictions per observation window
SLEEP_MS = 100    # delay between requests to simulate real traffic (ms)


def predict(text: str) -> dict:
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()

# (Task 6)
# Implement log_window() to aggregate results from one observation window and
# log them as MLflow metrics. For each window, compute and log:
# - Sentiment distribution (% positive, % negative, % neutral)
# - Average confidence score
# - Average latency in milliseconds
# Documentation: https://mlflow.org/docs/latest/tracking.html#logging-metrics
def log_window(window: list[dict], window_idx: int) -> None:
    """Log aggregated metrics for one observation window to MLflow."""
    if not window:
        return

    # Count sentiment distribution
    total = len(window)
    positive = sum(1 for r in window if r["sentiment"].lower() == "positive")
    negative = sum(1 for r in window if r["sentiment"].lower() == "negative")
    neutral = sum(1 for r in window if r["sentiment"].lower() == "neutral")

    pct_positive = positive / total
    pct_negative = negative / total
    pct_neutral = neutral / total

    # Average confidence
    avg_confidence = sum(r["confidence"] for r in window) / total

    # Average latency
    avg_latency = sum(r["latency_ms"] for r in window) / total

    # Log metrics to MLflow
    mlflow.log_metric("pct_positive", pct_positive, step=window_idx)
    mlflow.log_metric("pct_negative", pct_negative, step=window_idx)
    mlflow.log_metric("pct_neutral", pct_neutral, step=window_idx)
    mlflow.log_metric("avg_confidence", avg_confidence, step=window_idx)
    mlflow.log_metric("avg_latency_ms", avg_latency, step=window_idx)

    print(
        f"[Window {window_idx}] "
        f"pos={pct_positive:.2f}, neg={pct_negative:.2f}, neu={pct_neutral:.2f}, "
        f"conf={avg_confidence:.3f}, latency={avg_latency:.1f}ms"
    )


# (Task 6)
# Implement main() to:
# 1. Load headlines from data/stream.csv
# 2. Start an MLflow run for monitoring
# 3. Loop through headlines, call predict() for each, and collect results in windows
# 4. Call log_window() every WINDOW_SIZE predictions to log metrics to MLflow
def main():
    print(f"Connecting to API at {API_URL}")
    print(f"Logging to MLflow at {MLFLOW_TRACKING_URI}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Load streaming headlines
    stream_df = pd.read_csv("data/stream.csv")
    headlines = stream_df["text"].tolist()

    window = []
    window_idx = 0

    with mlflow.start_run(run_name="production-monitoring"):
        mlflow.log_param("window_size", WINDOW_SIZE)
        mlflow.log_param("sleep_ms", SLEEP_MS)

        for idx, text in enumerate(headlines, start=1):
            try:
                result = predict(text)
                window.append(result)
            except Exception as e:
                print(f"Prediction error at index {idx}: {e}")
                continue

            # Log window when full
            if len(window) == WINDOW_SIZE:
                log_window(window, window_idx)
                window_idx += 1
                window = []

            # Simulate real traffic
            time.sleep(SLEEP_MS / 1000.0)

        # Log remaining partial window
        if window:
            log_window(window, window_idx)

    print("Monitoring stream completed.")



if __name__ == "__main__":
    main()
