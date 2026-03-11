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
FEATURES_PATH = os.getenv("FEATURES_PATH", "data/processed/hydraulic_sample.csv")

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": build_failure_callback(ALERT_EMAIL),
}


def _get_production_f1() -> float | None:
    """Return the F1 metric of the current Production model, or None if no model exists."""
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            return None
        run_id = versions[0].run_id
        run = client.get_run(run_id)
        return float(run.data.metrics.get("f1_macro", 0.0))
    except MlflowException as exc:
        if "RESOURCE_DOES_NOT_EXIST" in str(exc):
            # Model not registered yet — treat as no Production model
            return None
        logger.exception("MLflow error while fetching Production F1 for %s", MODEL_NAME)
        raise


def train_and_log(**context) -> str:
    """Train multi-output model, log to MLflow, register in Model Registry.

    Uses the same FEATURES/TARGETS as src/train.py but wraps training
    with MLflow tracking and model registry.  Returns run_id via XCom.
    """
    import mlflow
    import mlflow.sklearn
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.multioutput import MultiOutputClassifier

    from src.train import FEATURES, TARGETS

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    df = pd.read_csv(FEATURES_PATH)

    X = df[FEATURES].values
    y = df[TARGETS].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y[:, 0],
    )

    model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, random_state=42),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute per-target macro F1 and overall macro average
    per_target_f1 = {}
    for i, target in enumerate(TARGETS):
        per_target_f1[target] = float(
            f1_score(y_test[:, i], y_pred[:, i], average="macro", zero_division=0)
        )

    f1_macro = float(np.mean(list(per_target_f1.values())))

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "MultiOutputClassifier(RandomForestClassifier)")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("targets", TARGETS)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        mlflow.log_metric("f1_macro", f1_macro)
        for target, f1_val in per_target_f1.items():
            mlflow.log_metric(f"f1_{target}", f1_val)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        run_id = run.info.run_id

    logger.info(
        "Run %s | F1 macro: %.4f | Per-target: %s",
        run_id,
        f1_macro,
        {k: f"{v:.4f}" for k, v in per_target_f1.items()},
    )
    return run_id


def promote_or_reject(**context) -> None:
    """Compare new model F1 vs Production. Promote if better, archive otherwise."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    run_id: str = context["ti"].xcom_pull(task_ids="train_model")

    if not run_id:
        raise ValueError("No run_id received from train_model task.")

    run = client.get_run(run_id)
    new_f1 = float(run.data.metrics.get("f1_macro", 0.0))

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
            to_stage="Production",
            archive_existing_versions=True,
        )
        logger.info("Model v%s promoted to Production.", new_version.version)
    else:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version.version,
            to_stage="Archived",
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
