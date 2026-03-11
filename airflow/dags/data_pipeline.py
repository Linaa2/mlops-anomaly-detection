from __future__ import annotations

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

from src.data_ingestion import download_dataset, merge_sensors, unzip_dataset  # noqa: E402
from src.preprocess import preprocess  # noqa: E402

ALERT_EMAIL = os.getenv("ALERT_EMAIL", "team@example.com")

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": build_failure_callback(ALERT_EMAIL),
}

with DAG(
    dag_id="data_pipeline",
    default_args=default_args,
    description="Download, extract and preprocess hydraulic sensor data",
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

    t_download >> t_unzip >> t_merge >> t_preprocess
