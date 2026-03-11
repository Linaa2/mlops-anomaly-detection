# Hydraulic System Condition Monitoring — MLOps Pipeline

Multi-output classification of hydraulic system component health using sensor data, with full MLOps infrastructure: Airflow orchestration, MLflow tracking, FastAPI serving, Streamlit UI, and Prometheus/Grafana monitoring.

## Dataset

**[UCI Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems)**

- 2205 cycles × 17 sensors (pressure, temperature, vibration, flow, efficiency)
- Labels from `profile.txt`: condition of 4 components per cycle
- Unstable cycles filtered out (`stable_flag == 0`)

| Target | Component | Classes |
|--------|-----------|---------|
| `cooler_condition` | Cooler | 3 (failure), 20 (reduced), 100 (ok) |
| `valve_condition` | Valve | 73 (failure), 80 (severe lag), 90 (slight lag), 100 (ok) |
| `pump_leakage` | Pump | 0 (none), 1 (weak), 2 (severe) |
| `accumulator_pressure` | Accumulator | 90 (failure), 100 (severe), 115 (reduced), 130 (ok) |

## Model

`MultiOutputClassifier(RandomForestClassifier(n_estimators=100))`

- **17 features**: PS1–PS6, EPS1, FS1–FS2, TS1–TS4, VS1, CE, CP, SE (mean per cycle)
- **4 targets**: multi-class classification per component
- **Metrics**: F1 macro per target + overall F1 macro average
- **Tracking**: MLflow (params, metrics, model artifact)

## Architecture

```
[UCI Dataset]
    ↓  DAG: data_pipeline (@daily)
[Download → Unzip → Merge → Preprocess → Sample 80% → Trigger training]
    ↓  DAG: training_pipeline (event-driven)
[Train via src/train.py → Register in MLflow → Promote/Reject (F1 comparison)]
    ↓
[FastAPI] ← model from MLflow Registry
    ↓
[Streamlit WebApp] → FastAPI → predictions → UI
    ↓
[Prometheus + Grafana] → API metrics + model performance
```

## Services

| Service | Port | Role |
|---------|------|------|
| Airflow | 8080 | Pipeline orchestration (admin/admin) |
| MLflow | 5000 | Experiment tracking & model registry |
| FastAPI | 8000 | Prediction API |
| Streamlit | 8501 | User interface |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Dashboards (admin/admin) |
| PostgreSQL | 5432 | Airflow metadata backend |

## Quick Start

```bash
# Clone and install
git clone git@github.com:Linaa2/mlops-anomaly-detection.git
cd mlops-anomaly-detection
pip install uv && uv sync

# Run tests
uv run pytest tests/ -v

# Launch dev environment (all services)
docker compose up --build

# Stop
docker compose down
```

## Project Structure

```
mlops-anomaly-detection/
├── .github/workflows/
│   ├── ci.yaml                    # CI: pytest + ruff on push/PR
│   └── cd.yml                     # CD: test → build DockerHub → deploy K8s
├── airflow/dags/
│   ├── callbacks.py               # Email failure alerting
│   ├── data_pipeline.py           # DAG: ingestion + preprocessing + sampling
│   └── training_pipeline.py       # DAG: train + MLflow register + promote/reject
├── api/
│   ├── Dockerfile
│   └── app.py                     # FastAPI prediction service
├── webapp/
│   ├── Dockerfile
│   └── app.py                     # Streamlit UI
├── src/
│   ├── data_ingestion.py          # Download UCI + unzip + merge sensors
│   ├── preprocess.py              # Filter, clean, select features+targets
│   └── train.py                   # MultiOutput RF training + MLflow tracking
├── k8s/
│   ├── api-deployment.yaml        # K8s deployment + service
│   └── webapp-deployment.yaml     # K8s deployment + service
├── monitoring/
│   ├── prometheus.yml             # Prometheus scrape config
│   └── grafana/                   # Grafana provisioning + dashboards
├── tests/
│   ├── test_dags.py               # Airflow DAG structure tests (9 tests)
│   └── test_model.py              # Model training + prediction tests (7 tests)
├── envs/.env.example              # Environment variable template
├── docker-compose.yml             # Dev environment (all services)
├── pyproject.toml                 # Dependencies (uv)
└── ORGANISATION.md                # Project plan & task tracking
```

## CI/CD

**CI** (`.github/workflows/ci.yaml`): runs on every push/PR
- `uv run pytest` — unit + DAG tests
- `uv run ruff check .` — linting

**CD** (`.github/workflows/cd.yml`): runs on push to `main`
1. Run tests
2. Build & push Docker images to DockerHub (API + Webapp)
3. Deploy to Kubernetes (conditional: skipped if `KUBECONFIG` secret not set)

## Airflow Pipelines

### `data_pipeline` — `@daily`
```
download_dataset → unzip_dataset → merge_sensors → preprocess → sample_data (80%) → trigger_training
```
Random 80% sampling on each run simulates new data arrival on a static dataset, giving purpose to the champion/challenger model comparison.

### `training_pipeline` — event-driven (triggered by data_pipeline)
```
train_model → promote_or_reject
```
- Delegates training to `src/train.py` (single source of truth)
- Registers model in MLflow Model Registry
- Compares new model F1 macro vs current Production model
- Promotes if better, archives otherwise

## Team

| Person | Scope |
|--------|-------|
| A | Data & ML pipeline (`src/`) |
| B | Airflow DAGs, continuous training, CI/CD, alerting |
| C | FastAPI API, Streamlit webapp |
| D | Docker, K8s, monitoring infra |

See `ORGANISATION.md` for detailed task tracking.

## Stack

Python 3.11 · scikit-learn · MLflow · Apache Airflow · FastAPI · Streamlit · Docker · Kubernetes · Prometheus · Grafana · GitHub Actions · uv
