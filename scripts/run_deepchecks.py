"""
Model quality check (drift) using deepchecks

Run:
    python scripts/run_deepchecks.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import yaml
from deepchecks.nlp import TextData
from deepchecks.nlp.checks import PredictionDrift, PropertyDrift
from dotenv import load_dotenv

from app.utils import load_classifier

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["deepchecks"]


def run_predictions(classifier, texts: list[str]) -> list[str]:
    results = classifier(texts, batch_size=32)
    return [r["label"] for r in results]


def main():
    params = load_params()
    property_drift_threshold = params["property_drift_threshold"]
    prediction_drift_threshold = params["prediction_drift_threshold"]

    print("Loading production model...")

    classifier = load_classifier()

    stream_df = pd.read_csv("data/stream.csv")
    test_df = pd.read_csv("data/test.csv")

    stream_texts = stream_df["text"].tolist()
    test_texts = test_df["text"].tolist()

    # Create two TextData objects from the text lists above: one for stream_texts
    # and one for test_texts. Use task_type="text_classification".
    # Then call calculate_builtin_properties() on each to compute NLP properties.
    # https://docs.deepchecks.com/stable/nlp/usage_guides/text_data_object.html#nlp-textdata-object
    # Create Deepchecks TextData objects
    test_dataset = TextData(
        data=test_texts,
        label=test_df["label"].tolist(),
        task_type="text_classification"
    )

    stream_dataset = TextData(
        data=stream_texts,
        label=None,
        task_type="text_classification"
    )

    # Compute NLP properties
    test_dataset.calculate_builtin_properties()
    stream_dataset.calculate_builtin_properties()

    # TODO: Compute NLP Property Drift between test_dataset (reference) and stream_dataset (current).
    # Extract the "Sentiment" drift score from the result and compare it against
    # property_drift_threshold.
    # If it exceeds the threshold, print an error message and terminate with failure exit code
    # https://docs.deepchecks.com/stable/nlp/auto_checks/train_test_validation/plot_property_drift.html#nlp-property-drift
    print("Running NLP Property Drift...")

    property_drift_check = PropertyDrift()
    property_drift_result = property_drift_check.run(
        train_dataset=test_dataset,
        test_dataset=stream_dataset
    )

    # Extract drift score for the "Sentiment" property
    sentiment_drift_score = property_drift_result.value["Sentiment"]["drift_score"]

    print(f"Sentiment Property Drift Score: {sentiment_drift_score:.4f}")

    if sentiment_drift_score > property_drift_threshold:
        print(
            f"ERROR: Sentiment property drift ({sentiment_drift_score:.4f}) "
            f"exceeds threshold ({property_drift_threshold})"
        )
        sys.exit(1)


    # TODO: Run predictions on both datasets with run_predictions(), then compute Prediction Drift.
    # Add a condition to fail if drift score is above the prediction_drift_threshold
    # If drift exceeds threshold, print an error message and terminate with failure exit code
    # https://docs.deepchecks.com/stable/general/guides/drift_guide.html#text-nlp-checks
    print("Running Prediction Drift...")

    # Run model predictions
    test_preds = run_predictions(classifier, test_texts)
    stream_preds = run_predictions(classifier, stream_texts)

    # Attach predictions to datasets
    test_dataset.predictions = test_preds
    stream_dataset.predictions = stream_preds

    prediction_drift_check = PredictionDrift()
    prediction_drift_result = prediction_drift_check.run(
        train_dataset=test_dataset,
        test_dataset=stream_dataset
    )

    prediction_drift_score = prediction_drift_result.value["drift_score"]

    print(f"Prediction Drift Score: {prediction_drift_score:.4f}")

    if prediction_drift_score > prediction_drift_threshold:
        print(
            f"ERROR: Prediction drift ({prediction_drift_score:.4f}) "
            f"exceeds threshold ({prediction_drift_threshold})"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
