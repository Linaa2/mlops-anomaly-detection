# Hydraulic System Condition Monitoring — MLOps Pipeline

Multi-output classification of hydraulic system component health using sensor data, with full MLOps infrastructure: Airflow orchestration, MLflow tracking, FastAPI serving, Streamlit UI, and Prometheus/Grafana monitoring.

## Dataset

**[UCI Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems)**

- 2205 cycles x 17 sensors (pressure, temperature, vibration, flow, efficiency)
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

- **17 features**: PS1-PS6, EPS1, FS1-FS2, TS1-TS4, VS1, CE, CP, SE (mean per cycle)
- **4 targets**: multi-class classification per component
- **Metrics**: F1 macro, accuracy, precision, recall per target
- **Tracking**: MLflow (params, metrics, model artifact, JSON report)

**Results (test set 20%):**

| Target | Accuracy | F1 macro | F1 weighted |
|--------|----------|----------|-------------|
| cooler_condition | 1.000 | 1.000 | 1.000 |
| valve_condition | 0.948 | 0.947 | 0.948 |
| pump_leakage | 0.993 | 0.993 | 0.993 |
| accumulator_pressure | 0.990 | 0.990 | 0.990 |

## Architecture

```
[UCI Dataset]
    |  DAG: data_pipeline (@daily)
[Download -> Unzip -> Merge -> Preprocess -> Sample 80% -> Trigger training]
    |  DAG: training_pipeline (event-driven)
[Train via src/train.py -> Register in MLflow -> Promote/Reject (F1 comparison)]
    |
[FastAPI] <- model served from models/model.pkl
    |
[Streamlit WebApp] -> FastAPI -> predictions -> UI
    |
[Prometheus + Grafana] -> API metrics (/metrics) + system metrics (node-exporter)
```

## Services (Docker Compose)

| Service | Port | Role |
|---------|------|------|
| Airflow (webserver + scheduler + init) | 8080 | Pipeline orchestration (admin/admin) |
| MLflow | 5000 | Experiment tracking & model registry |
| FastAPI | 8000 | Prediction API (`/predict`, `/health`, `/metrics`) |
| Streamlit | 8501 | User interface (2 tabs: prediction + model evaluation) |
| Prometheus | 9090 | Metrics collection (scrapes API + node-exporter) |
| Grafana | 3000 | Dashboards, auto-provisioned (admin/admin) |
| PostgreSQL | 5432 | Airflow metadata backend |
| Node Exporter | 9100 | System metrics for Prometheus |

## Quick Start

```bash
# Clone and install
git clone git@github.com:Linaa2/mlops-anomaly-detection.git
cd mlops-anomaly-detection
pip install uv && uv sync

# Run tests (26 tests)
uv run pytest tests/ -v

# Lint
uv run ruff check .

# Launch dev environment (all 9 services)
docker compose up --build

# Stop
docker compose down
```

## Project Structure

```
mlops-anomaly-detection/
├── .github/workflows/
│   ├── ci.yaml                    # CI: pytest + ruff on push/PR
│   └── cd.yml                     # CD: test -> build GHCR+DockerHub -> deploy K8s
├── airflow/dags/
│   ├── callbacks.py               # Email failure alerting (bonus)
│   ├── data_pipeline.py           # DAG: ingestion + preprocessing + sampling
│   └── training_pipeline.py       # DAG: train + MLflow register + promote/reject
├── api/
│   ├── Dockerfile
│   └── app.py                     # FastAPI prediction service + Prometheus /metrics
├── webapp/
│   ├── Dockerfile
│   └── app.py                     # Streamlit UI (prediction + model evaluation)
├── src/
│   ├── data_ingestion.py          # Download UCI + unzip + merge sensors
│   ├── preprocess.py              # Filter, clean, select features+targets
│   └── train.py                   # MultiOutput RF training + MLflow tracking + JSON export
├── k8s/
│   ├── api-deployment.yaml        # K8s deployment + service
│   └── webapp-deployment.yaml     # K8s deployment + service
├── monitoring/
│   ├── prometheus.yml             # Prometheus scrape config (API, node-exporter)
│   └── grafana/                   # Grafana provisioning + dashboards (auto-provisioned)
├── tests/
│   ├── test_dags.py               # Airflow DAG structure tests (9 tests)
│   ├── test_model.py              # Model training + prediction tests (7 tests)
│   └── test_preprocessing.py      # Preprocessing integration tests (10 tests)
├── docs/
│   ├── model_card.md              # Model card (intended use, limitations, ethics)
│   └── ml_pipeline.md             # ML pipeline design rationale
├── envs/.env.example              # Environment variable template
├── docker-compose.yml             # Dev environment (9 services)
├── pyproject.toml                 # Dependencies (uv)
└── ORGANISATION.md                # Project plan & task tracking
```

## CI/CD

**CI** (`.github/workflows/ci.yaml`): runs on every push/PR
- Python 3.11
- `uv run pytest tests/ -v` — 26 tests (DAGs + model + preprocessing)
- `uv run ruff check .` — linting

**CD** (`.github/workflows/cd.yml`): runs on push to `main`
1. Run tests
2. Build & push Docker images to GHCR + DockerHub (API + Webapp), tagged `latest` + git SHA
3. Deploy to Kubernetes (conditional: skipped if `KUBECONFIG` secret not set)

## Airflow Pipelines

### `data_pipeline` — `@daily`
```
download_dataset -> unzip_dataset -> merge_sensors -> preprocess -> sample_data (80%) -> trigger_training
```
Random 80% sampling on each run simulates new data arrival on a static dataset, giving purpose to the champion/challenger model comparison.

### `training_pipeline` — event-driven (triggered by data_pipeline)
```
train_model -> promote_or_reject
```
- Delegates training to `src/train.py` (separation of concerns)
- Registers model in MLflow Model Registry
- Compares new model F1 macro vs current Production model
- Promotes if better, archives otherwise (versioning/rollback via MLflow stages)

## Objectives Coverage

**10/10 required objectives completed** + 3 bonus:

| # | Objective | Status |
|---|-----------|--------|
| 1 | Data extraction & preprocessing | Done — `data_ingestion.py` + `preprocess.py` + DAG |
| 2 | ML model | Done — MultiOutputClassifier(RF), F1 > 0.94 on all targets |
| 3 | Model registry | Done — MLflow Model Registry |
| 4 | Airflow retraining pipeline | Done — 2 DAGs (data + training) |
| 5 | MLflow experiment tracking | Done — params, metrics, artifacts |
| 6 | API | Done — FastAPI, Pydantic, Swagger, `/predict`, `/health`, `/metrics` |
| 7 | WebApp | Done — Streamlit, 2 tabs (prediction + model evaluation) |
| 8 | Continuous Training | Done — daily sampling + champion/challenger F1 comparison |
| 9 | Docker + K8s + CI/CD | Done — Docker Compose (9 services), K8s manifests, GitHub Actions CI/CD |
| 10 | GitHub versioning & docs | Done — 20+ PRs, conventional commits, README, model card |
| **12** | **Monitoring (bonus)** | Done — Prometheus + Grafana + Node Exporter + auto-provisioned dashboard |
| **14** | **Model versioning/rollback (bonus)** | Done — MLflow Registry stages (None -> Production -> Archived) |
| **17** | **Email alerting (bonus)** | Done — `on_failure_callback` on all DAGs |

## Team

| Person | Scope |
|--------|-------|
| A | Data & ML pipeline (`src/train.py`, MLflow integration, metrics export) |
| B | Airflow DAGs, continuous training, CI/CD, tests (26), alerting, docs |
| C | FastAPI API (Pydantic), Streamlit webapp (2 tabs) |
| D | Docker Compose (9 services), K8s manifests, CD pipeline, monitoring |

See `ORGANISATION.md` for detailed task tracking.

## Stack

Python 3.11 · scikit-learn · MLflow · Apache Airflow · FastAPI · Streamlit · Docker · Kubernetes · Prometheus · Grafana · GitHub Actions · uv
