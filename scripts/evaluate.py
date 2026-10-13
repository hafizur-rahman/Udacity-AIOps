"""
Evaluate sentiment model on the test set and register it
in the MLflow Model Registry.

Run:
    python scripts/evaluate.py
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import pipeline

# TODO: add mlflow required dependencies
import mlflow
import mlflow.transformers

load_dotenv()


def load_test_data(test_path: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(test_path)
    return df["text"].tolist(), df["label"].tolist()


def build_classifier(model_id: str):
    print(f"Loading {model_id}...")
    return pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id,
        truncation=True,
        max_length=512,
    )


def run_inference(classifier, texts: list[str]) -> list[str]:
    print(f"Running inference on {len(texts)} samples...")
    results = classifier(texts, batch_size=32)
    return [r["label"] for r in results]

def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
    }


def main():
    model_id = os.getenv("HF_MODEL_ID", "baptle/FinBERT_market_based")
    test_path = os.path.join("data", "test.csv")

    texts, y_true = load_test_data(test_path)
    classifier = build_classifier(model_id)
    y_pred = run_inference(classifier, texts)

    metrics = compute_metrics(y_true, y_pred)

    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Use MLflow to track the evaluation run.
    # Set the tracking URI and experiment name from environment variables,
    # start a run, log the model parameters and computed metrics, then log
    # and register the pipeline in the Model Registry.
    # Documentation: https://mlflow.org/docs/latest/tracking.html
    # MLflow tracking setup
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "sentiment-eval")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="sentiment_model_evaluation"):
        # Log parameters
        mlflow.log_param("model_id", model_id)
        mlflow.log_param("test_path", test_path)

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # Log classification report as an artifact
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_true, y_pred))
        mlflow.log_artifact(report_path)

        # Log and register the HF pipeline
        model_name = os.getenv("MODEL_NAME", "sentiment-classifier")

        mlflow.transformers.log_model(
            transformers_model=classifier,
            artifact_path="model",
            registered_model_name=model_name,
            input_example={"text": texts[0]},
        )

        print(f"\nModel logged and registered as: {model_name}")

if __name__ == "__main__":
    main()
