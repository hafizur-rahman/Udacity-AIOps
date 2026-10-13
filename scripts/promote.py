"""
Promote the latest version of the registered model to Production
if the weighted F1 score from the most recent evaluation run exceeds
the required threshold.

Run:
    python scripts/promote.py
"""

import os

import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["promote"]


def get_latest_f1(client: MlflowClient, experiment_name: str) -> tuple[str, float]:
    """
    Return the run_id and f1_weighted of the most recent evaluation run.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No evaluation runs found.")

    run = runs[0]
    run_id = run.info.run_id
    f1 = run.data.metrics.get("f1_weighted")

    if f1 is None:
        raise ValueError("Metric 'f1_weighted' not found in the latest run.")

    return run_id, f1


def promote(client: MlflowClient, model_name: str):
    """
    # Fetch all registered versions of the model, identify the latest one,
    # and assign it the "production" alias using the MLflow client.
    # Documentation: https://mlflow.org/docs/latest/ml/model-registry/

    # get the latest version of the model
    """
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No registered versions found for model '{model_name}'.")

    # Find the highest version number
    latest_version = max(versions, key=lambda v: int(v.version))

    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=latest_version.version,
    )

    print(
        f"Model '{model_name}' version {latest_version.version} "
        f"promoted to 'production'."
    )


def main():
    params = load_params()
    threshold = params["f1_threshold"]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    model_name = os.getenv("MODEL_NAME")

    if not tracking_uri or not experiment_name or not model_name:
        raise ValueError(
            "Environment variables MLFLOW_TRACKING_URI, "
            "MLFLOW_EXPERIMENT_NAME, and MODEL_NAME must be set."
        )

    client = MlflowClient(tracking_uri)

    run_id, latest_f1 = get_latest_f1(client, experiment_name)

    print(f"Latest evaluation run: {run_id}")
    print(f"Weighted F1 score: {latest_f1:.4f}")
    print(f"Required threshold: {threshold:.4f}")

    if latest_f1 >= threshold:
        promote(client, model_name)
    else:
        print(
            f"Model NOT promoted. F1 score {latest_f1:.4f} "
            f"did not meet threshold {threshold:.4f}."
        )


if __name__ == "__main__":
    main()
