from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator

from airflow import DAG

# Ensure project root is importable
_project_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _project_root)

# Ensure dags folder is importable (for callbacks)
_dags_dir = os.path.dirname(__file__)
if _dags_dir not in sys.path:
    sys.path.insert(0, _dags_dir)

from callbacks import build_failure_callback  # noqa: E402

logger = logging.getLogger(__name__)

ALERT_EMAIL = os.getenv("ALERT_EMAIL", "team@example.com")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "hydraulic-anomaly-detector"
EXPERIMENT_NAME = "hydraulic-condition-monitoring"

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": build_failure_callback(ALERT_EMAIL),
}


def _compute_f1_macro(run) -> float:
    """Compute the average of per-target F1 metrics from an MLflow run."""
    from src.train import TARGETS

    f1_values = []
    for target in TARGETS:
        val = run.data.metrics.get(f"f1_macro_{target}")
        if val is not None:
            f1_values.append(float(val))
    return sum(f1_values) / len(f1_values) if f1_values else 0.0


def _get_production_f1() -> float | None:
    """Return the mean F1 of the current Production model, or None if absent."""
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            return None
        run = client.get_run(versions[0].run_id)
        return _compute_f1_macro(run)
    except MlflowException as exc:
        if "RESOURCE_DOES_NOT_EXIST" in str(exc):
            return None
        logger.exception("MLflow error while fetching Production F1 for %s", MODEL_NAME)
        raise


def train_and_log(**context) -> str:
    """Delegate training to src.train.train(), then register model in MLflow Registry.

    src/train.py handles: data loading, train/test split, model fitting,
    MLflow run creation, param/metric logging, and artifact saving.

    This task adds: Model Registry registration (needed for promote/reject).
    Returns the run_id via XCom.
    """
    from mlflow.tracking import MlflowClient

    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    from src.train import TARGETS, train

    # src/train.py creates an MLflow run, logs params/metrics/artifact
    train()

    # Retrieve the run that train() just created
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found after training.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No MLflow run found after training.")

    run = runs[0]
    run_id = run.info.run_id

    # Register the model artifact in the Model Registry for promote/reject
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, MODEL_NAME)

    # Log the overall f1_macro for easier comparison in promote_or_reject
    f1_macro = _compute_f1_macro(run)
    client.log_metric(run_id, "f1_macro", f1_macro)

    logger.info(
        "Run %s | F1 macro: %.4f | Per-target: %s",
        run_id,
        f1_macro,
        {t: f"{run.data.metrics.get(f'f1_macro_{t}', 0.0):.4f}" for t in TARGETS},
    )
    return run_id


def promote_or_reject(**context) -> None:
    """Compare new model F1 vs Production. Promote if better, archive otherwise."""
    from mlflow.tracking import MlflowClient

    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    run_id: str = context["ti"].xcom_pull(task_ids="train_model")

    if not run_id:
        raise ValueError("No run_id received from train_model task.")

    run = client.get_run(run_id)
    new_f1 = _compute_f1_macro(run)

    versions = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging"])
    if not versions:
        raise ValueError("No new model version found in registry.")
    new_version = sorted(versions, key=lambda v: int(v.version))[-1]

    prod_f1 = _get_production_f1()
    logger.info("New model F1 macro: %.4f | Production F1 macro: %s", new_f1, prod_f1)

    if prod_f1 is None or new_f1 > prod_f1:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info("Model v%s promoted to Production.", new_version.version)
    else:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version.version,
            stage="Archived",
        )
        logger.info("Model v%s archived (F1 did not improve).", new_version.version)


with DAG(
    dag_id="training_pipeline",
    default_args=default_args,
    description="Retraining triggered by data_pipeline: train, evaluate, promote/reject",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["training", "mlflow"],
) as dag:
    t_train = PythonOperator(
        task_id="train_model",
        python_callable=train_and_log,
    )

    t_promote = PythonOperator(
        task_id="promote_or_reject",
        python_callable=promote_or_reject,
    )

    t_train >> t_promote
