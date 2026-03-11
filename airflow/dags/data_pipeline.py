from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from airflow import DAG

# Ensure project root is importable
_project_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _project_root)

# Ensure dags folder is importable (for callbacks)
_dags_dir = os.path.dirname(__file__)
if _dags_dir not in sys.path:
    sys.path.insert(0, _dags_dir)

from callbacks import build_failure_callback  # noqa: E402

from src.data_ingestion import download_dataset, merge_sensors, unzip_dataset  # noqa: E402
from src.preprocess import preprocess  # noqa: E402

ALERT_EMAIL = os.getenv("ALERT_EMAIL", "team@example.com")
SAMPLE_FRAC = float(os.getenv("SAMPLE_FRAC", "0.8"))
CLEAN_PATH = "data/processed/hydraulic_clean.csv"
SAMPLE_PATH = "data/processed/hydraulic_sample.csv"

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": build_failure_callback(ALERT_EMAIL),
}


def sample_data(**context) -> None:
    """Draw a random subset from the clean dataset to simulate new data arrival."""
    import pandas as pd

    df = pd.read_csv(CLEAN_PATH)
    sample = df.sample(frac=SAMPLE_FRAC, random_state=None)  # truly random each run
    sample.to_csv(SAMPLE_PATH, index=False)

    n_total = len(df)
    n_sample = len(sample)
    print(f"Sampled {n_sample}/{n_total} rows ({SAMPLE_FRAC:.0%}) -> {SAMPLE_PATH}")


with DAG(
    dag_id="data_pipeline",
    default_args=default_args,
    description="Download, extract, preprocess and sample hydraulic sensor data",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["data", "ingestion"],
) as dag:
    t_download = PythonOperator(
        task_id="download_dataset",
        python_callable=download_dataset,
    )

    t_unzip = PythonOperator(
        task_id="unzip_dataset",
        python_callable=unzip_dataset,
    )

    t_merge = PythonOperator(
        task_id="merge_sensors",
        python_callable=merge_sensors,
    )

    t_preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess,
    )

    t_sample = PythonOperator(
        task_id="sample_data",
        python_callable=sample_data,
    )

    t_trigger_training = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="training_pipeline",
        wait_for_completion=False,
    )

    t_download >> t_unzip >> t_merge >> t_preprocess >> t_sample >> t_trigger_training
