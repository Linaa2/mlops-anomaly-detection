# Airflow Pipelines Implementation Plan

**Goal:** Implement two Airflow DAGs — `data_pipeline` (daily ingestion + preprocessing + random sampling) and `training_pipeline` (event-driven retraining with MLflow comparison and automatic model promotion) — with failure alerting by email.

**Architecture:** Each DAG is a standalone Python file in `airflow/dags/`. Tasks call functions from `src/` directly (no SubDagOperator). Data flows through a shared Docker volume mounted at `/opt/airflow/data/`. The training DAG compares the new model's F1 score against the current Production model in MLflow Registry before promoting.

**Tech Stack:** Apache Airflow 2.x, MLflow, scikit-learn, Python 3.10, pytest, uv

---

## Context

### Existing files to reuse
- `src/data_ingestion.py` — `download_dataset()`, `unzip_dataset()`, `merge_sensors()`
- `src/preprocess.py` — `preprocess()`
- `src/train.py` — `train()` (currently no MLflow — Personne A will add it; we stub it for now)

### File paths convention (via env vars)
```
DATA_DIR   = /opt/airflow/data        (default for Airflow container)
MODEL_DIR  = /opt/airflow/models
MLFLOW_TRACKING_URI = http://mlflow:5000
```

### DAG summary
| DAG id | Schedule | Trigger | Key output |
|--------|----------|---------|------------|
| `data_pipeline` | `@daily` | time | `hydraulic_clean.csv`, `hydraulic_sample.csv` |
| `training_pipeline` | `None` (event-driven) | `TriggerDagRunOperator` | model promoted in MLflow Registry |

---

## Task 1: Project structure

**Files:**
- Create: `airflow/dags/__init__.py`
- Create: `airflow/dags/data_pipeline.py`
- Create: `airflow/dags/training_pipeline.py`
- Create: `airflow/dags/callbacks.py`
- Create: `tests/test_dags.py`

**Step 1: Create the directory structure**

```bash
mkdir -p airflow/dags
touch airflow/dags/__init__.py
touch airflow/dags/callbacks.py
touch airflow/dags/data_pipeline.py
touch airflow/dags/training_pipeline.py
touch tests/test_dags.py
```

**Step 2: Verify**

```bash
find airflow/ tests/test_dags.py
```
Expected: all 5 files listed.

**Step 3: Commit**

```bash
git add airflow/ tests/test_dags.py
git commit -m "feat(airflow): scaffold DAG directory structure"
```

---

## Task 2: Email failure callback

**Files:**
- Modify: `airflow/dags/callbacks.py`

**Step 1: Write the failing test**

In `tests/test_dags.py`:
```python
from callbacks import build_failure_callback


def test_build_failure_callback_returns_callable():
    cb = build_failure_callback(email="test@example.com")
    assert callable(cb)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_dags.py::test_build_failure_callback_returns_callable -v
```
Expected: FAIL — `ImportError` or `ModuleNotFoundError`.

**Step 3: Write implementation in `airflow/dags/callbacks.py`**

```python
from __future__ import annotations

from typing import Callable

from airflow.utils.email import send_email


def build_failure_callback(email: str) -> Callable:
    """Return an Airflow on_failure_callback that sends an alert email."""

    def on_failure(context: dict) -> None:
        dag_id = context["dag"].dag_id
        task_id = context["task_instance"].task_id
        execution_date = context["execution_date"]
        log_url = context["task_instance"].log_url

        subject = f"[Airflow] DAG {dag_id} — task {task_id} FAILED"
        body = f"""
        <h3>Task failure alert</h3>
        <p><b>DAG:</b> {dag_id}</p>
        <p><b>Task:</b> {task_id}</p>
        <p><b>Execution date:</b> {execution_date}</p>
        <p><b>Logs:</b> <a href="{log_url}">{log_url}</a></p>
        """
        send_email(to=email, subject=subject, html_content=body)

    return on_failure
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_dags.py::test_build_failure_callback_returns_callable -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add airflow/dags/callbacks.py tests/test_dags.py
git commit -m "feat(airflow): add email failure callback builder"
```

---

## Task 3: DAG `data_pipeline`

**Files:**
- Modify: `airflow/dags/data_pipeline.py`
- Modify: `tests/test_dags.py`

**Step 1: Write the failing test**

Append to `tests/test_dags.py`:
```python
from airflow.models import DagBag


def test_data_pipeline_dag_loads():
    dagbag = DagBag(dag_folder="airflow/dags", include_examples=False)
    assert "data_pipeline" in dagbag.dags
    assert len(dagbag.import_errors) == 0


def test_data_pipeline_task_ids():
    dagbag = DagBag(dag_folder="airflow/dags", include_examples=False)
    dag = dagbag.dags["data_pipeline"]
    task_ids = {t.task_id for t in dag.tasks}
    assert task_ids == {
        "download_dataset",
        "unzip_dataset",
        "merge_sensors",
        "preprocess",
    }


def test_data_pipeline_schedule():
    dagbag = DagBag(dag_folder="airflow/dags", include_examples=False)
    dag = dagbag.dags["data_pipeline"]
    assert str(dag.schedule_interval) == "@daily"
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_dags.py::test_data_pipeline_dag_loads -v
```
Expected: FAIL — DAG not found.

**Step 3: Write `airflow/dags/data_pipeline.py`**

```python
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from callbacks import build_failure_callback
from src.data_ingestion import download_dataset, merge_sensors, unzip_dataset
from src.preprocess import preprocess

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
    schedule_interval="@daily",
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
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_dags.py::test_data_pipeline_dag_loads \
              tests/test_dags.py::test_data_pipeline_task_ids \
              tests/test_dags.py::test_data_pipeline_schedule -v
```
Expected: 3 PASS.

**Step 5: Commit**

```bash
git add airflow/dags/data_pipeline.py tests/test_dags.py
git commit -m "feat(airflow): add data_pipeline DAG (download → unzip → merge → preprocess)"
```

---

## Task 4: DAG `training_pipeline`

**Files:**
- Modify: `airflow/dags/training_pipeline.py`
- Modify: `tests/test_dags.py`

### Sub-task 4a: Training and evaluation functions

**Step 1: Write the failing test**

Append to `tests/test_dags.py`:
```python
from airflow.dags.training_pipeline import evaluate_model, promote_or_reject


def test_evaluate_model_returns_float():
    """evaluate_model must return a float F1 score."""
    import numpy as np

    score = evaluate_model.__wrapped__() if hasattr(evaluate_model, "__wrapped__") else None
    # We test the helper directly
    from airflow.dags.training_pipeline import _compute_f1_on_test_set
    import pandas as pd

    # Minimal smoke test with fake data
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    y = [0, 0, 1, 0, 1]
    score = _compute_f1_on_test_set(X, y)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_dags.py::test_evaluate_model_returns_float -v
```
Expected: FAIL — import error.

**Step 3: Write `airflow/dags/training_pipeline.py`**

```python
from __future__ import annotations

import os
from datetime import datetime, timedelta

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from airflow import DAG
from airflow.operators.python import PythonOperator

from callbacks import build_failure_callback

ALERT_EMAIL = os.getenv("ALERT_EMAIL", "team@example.com")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "hydraulic-anomaly-detector"
FEATURES_PATH = os.getenv("FEATURES_PATH", "data/processed/hydraulic_clean.csv")
F1_THRESHOLD = float(os.getenv("F1_MIN_THRESHOLD", "0.0"))  # always beat prod

FEATURES = ["PS1", "PS2", "PS3", "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": build_failure_callback(ALERT_EMAIL),
}


# ---------------------------------------------------------------------------
# Helpers (kept public for unit testing)
# ---------------------------------------------------------------------------

def _compute_f1_on_test_set(X: pd.DataFrame, y: list) -> float:
    """Train a minimal model on X/y and return macro F1 on a hold-out set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(X_train_s)
    preds = clf.predict(X_test_s)
    # IsolationForest returns 1 (normal) / -1 (anomaly) — map to 0/1
    preds_bin = [0 if p == 1 else 1 for p in preds]
    y_test_bin = [0 if v == 1 else 1 for v in y_test]
    return float(f1_score(y_test_bin, preds_bin, zero_division=0))


def _get_production_f1() -> float | None:
    """Return the F1 metric of the current Production model, or None."""
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            return None
        run_id = versions[0].run_id
        run = client.get_run(run_id)
        return float(run.data.metrics.get("f1_score", 0.0))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def train_and_log() -> str:
    """Train model, log to MLflow, register in Model Registry. Returns run_id."""
    df = pd.read_csv(FEATURES_PATH)
    X = df[FEATURES].dropna()
    # Use IsolationForest predictions as pseudo-labels for F1 computation
    y = [1] * len(X)  # placeholder — Personne A will replace with profile.txt labels

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_scaled)

    preds = model.predict(X_scaled)
    preds_bin = [0 if p == 1 else 1 for p in preds]
    f1 = float(f1_score(y, preds_bin, average="macro", zero_division=0))

    with mlflow.start_run() as run:
        mlflow.log_param("contamination", 0.05)
        mlflow.log_param("features", FEATURES)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("anomaly_ratio", sum(preds_bin) / len(preds_bin))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        run_id = run.info.run_id

    print(f"Run id: {run_id} | F1: {f1:.4f}")
    return run_id


def promote_or_reject(**context) -> None:
    """Compare new model F1 vs Production. Promote if better, archive otherwise."""
    client = MlflowClient()
    run_id: str = context["ti"].xcom_pull(task_ids="train_model")

    # Get F1 of the newly registered model
    run = client.get_run(run_id)
    new_f1 = float(run.data.metrics.get("f1_score", 0.0))

    # Get latest version just registered (stage = None / Staging)
    versions = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging"])
    if not versions:
        raise ValueError("No new model version found in registry.")
    new_version = sorted(versions, key=lambda v: int(v.version))[-1]

    prod_f1 = _get_production_f1()
    print(f"New model F1: {new_f1:.4f} | Production F1: {prod_f1}")

    if prod_f1 is None or new_f1 > prod_f1:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version.version,
            to_stage="Production",
            archive_existing_versions=True,
        )
        print(f"Model v{new_version.version} promoted to Production.")
    else:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version.version,
            to_stage="Archived",
        )
        print(f"Model v{new_version.version} archived (F1 did not improve).")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="training_pipeline",
    default_args=default_args,
    description="Weekly retraining: train → evaluate → compare → promote/reject",
    schedule_interval="@weekly",
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
        provide_context=True,
    )

    t_train >> t_promote
```

**Step 4: Run the test**

```bash
uv run pytest tests/test_dags.py::test_evaluate_model_returns_float -v
```
Expected: PASS.

### Sub-task 4b: DAG structure tests

**Step 5: Append DAG structure tests to `tests/test_dags.py`**

```python
def test_training_pipeline_dag_loads():
    dagbag = DagBag(dag_folder="airflow/dags", include_examples=False)
    assert "training_pipeline" in dagbag.dags
    assert len(dagbag.import_errors) == 0


def test_training_pipeline_task_ids():
    dagbag = DagBag(dag_folder="airflow/dags", include_examples=False)
    dag = dagbag.dags["training_pipeline"]
    task_ids = {t.task_id for t in dag.tasks}
    assert task_ids == {"train_model", "promote_or_reject"}


def test_training_pipeline_schedule():
    dagbag = DagBag(dag_folder="airflow/dags", include_examples=False)
    dag = dagbag.dags["training_pipeline"]
    assert str(dag.schedule_interval) == "@weekly"


def test_training_pipeline_task_order():
    dagbag = DagBag(dag_folder="airflow/dags", include_examples=False)
    dag = dagbag.dags["training_pipeline"]
    train_task = dag.get_task("train_model")
    assert "promote_or_reject" in {t.task_id for t in train_task.downstream_list}
```

**Step 6: Run all DAG tests**

```bash
uv run pytest tests/test_dags.py -v
```
Expected: all tests PASS.

**Step 7: Commit**

```bash
git add airflow/dags/training_pipeline.py tests/test_dags.py
git commit -m "feat(airflow): add training_pipeline DAG with MLflow promotion logic"
```

---

## Task 5: Add `apache-airflow` and `mlflow` to dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add dependencies**

In `pyproject.toml`, add to `dependencies`:
```toml
"apache-airflow>=2.8,<3.0",
"mlflow>=2.11",
```

**Step 2: Sync**

```bash
uv sync
```
Expected: packages installed without errors.

**Step 3: Re-run all tests**

```bash
uv run pytest tests/test_dags.py -v
```
Expected: all PASS.

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add apache-airflow and mlflow dependencies"
```

---

## Task 6: Full test suite pass + push

**Step 1: Run all project tests**

```bash
uv run pytest -v
```
Expected: all tests PASS (existing `test_model.py` + new `test_dags.py`).

**Step 2: Run linter**

```bash
uv run ruff check .
```
Expected: no errors.

**Step 3: Push branch**

```bash
git push origin feature/airflow-pipelines
```

---

## Notes for integration with other team members

- **Personne A** (`src/train.py`) : remplacer le placeholder `y = [1] * len(X)` dans `train_and_log()` par le chargement réel de `profile.txt` une fois disponible.
- **Personne C** (`api/app.py`) : le modèle en Production est accessible via `mlflow.sklearn.load_model("models:/hydraulic-anomaly-detector/Production")`.
- **Personne D** (`docker-compose.yml`) : monter le volume `./data:/opt/airflow/data` et `./models:/opt/airflow/models`; exposer le service `mlflow` sur le port 5000.
