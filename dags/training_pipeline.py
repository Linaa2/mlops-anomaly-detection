from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Ensure project root is importable
_project_root = os.path.join(os.path.dirname(__file__), "..")
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

FEATURES = ["PS1", "PS2", "PS3", "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP"]

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
        return float(run.data.metrics.get("f1_score", 0.0))
    except MlflowException as exc:
        if "RESOURCE_DOES_NOT_EXIST" in str(exc):
            # Model not registered yet — treat as no Production model
            return None
        logger.exception("MLflow error while fetching Production F1 for %s", MODEL_NAME)
        raise


def train_and_log(**context) -> str:
    """Train model, log to MLflow, register in Model Registry. Returns run_id via XCom."""
    import mlflow
    import mlflow.sklearn
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    df = pd.read_csv(FEATURES_PATH)
    X = df[FEATURES].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_scaled)

    preds = model.predict(X_scaled)
    preds_bin = [0 if p == 1 else 1 for p in preds]
    # Placeholder labels — Personne A will replace with profile.txt labels
    y_placeholder = [0] * len(X)
    f1 = float(f1_score(y_placeholder, preds_bin, average="macro", zero_division=0))
    anomaly_ratio = sum(preds_bin) / len(preds_bin)

    with mlflow.start_run() as run:
        mlflow.log_param("contamination", 0.05)
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("anomaly_ratio", anomaly_ratio)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        run_id = run.info.run_id

    logger.info("Run id: %s | F1: %.4f | Anomaly ratio: %.4f", run_id, f1, anomaly_ratio)
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
    new_f1 = float(run.data.metrics.get("f1_score", 0.0))

    versions = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging"])
    if not versions:
        raise ValueError("No new model version found in registry.")
    new_version = sorted(versions, key=lambda v: int(v.version))[-1]

    prod_f1 = _get_production_f1()
    logger.info("New model F1: %.4f | Production F1: %s", new_f1, prod_f1)

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
