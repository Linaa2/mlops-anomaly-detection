<h1 align="center">Hydraulic System Condition Monitoring</h1>

<p align="center"><strong>Détection de pannes multi-composants sur systèmes hydrauliques par classification supervisée</strong></p>

<p align="center">
  <a href="https://github.com/Linaa2/mlops-anomaly-detection/actions/workflows/ci.yaml">
    <img src="https://github.com/Linaa2/mlops-anomaly-detection/actions/workflows/ci.yaml/badge.svg?branch=main" alt="CI Pipeline">
  </a>
  <a href="https://github.com/Linaa2/mlops-anomaly-detection/pkgs/container/mlops-anomaly-detection">
    <img src="https://img.shields.io/badge/Docker-GHCR.io-0db7ed?logo=docker&logoColor=white" alt="Images Docker">
  </a>
  <a href="https://linaa2.github.io/mlops-anomaly-detection/">
    <img src="https://img.shields.io/badge/Docs-GitHub%20Pages-327FC7?logo=github" alt="Documentation GitHub Pages">
  </a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white" alt="MLflow">
  <a href="https://github.com/Linaa2/mlops-anomaly-detection/actions/workflows/ci.yaml">
    <img src="https://img.shields.io/endpoint?url=https://linaa2.github.io/mlops-anomaly-detection/badges/tests.json&logo=pytest&logoColor=white" alt="Tests">
  </a>
  <a href="https://github.com/Linaa2/mlops-anomaly-detection/actions/workflows/ci.yaml">
    <img src="https://img.shields.io/endpoint?url=https://linaa2.github.io/mlops-anomaly-detection/badges/coverage.json&logo=codecov&logoColor=white" alt="Couverture">
  </a>
</p>

---

## 📊 Aperçu

- Pipeline MLOps complet pour la **surveillance de l'état de composants hydrauliques** à partir de données capteurs (UCI dataset — 2205 cycles × 17 capteurs).
- Modèle `MultiOutputClassifier(RandomForestClassifier)` classifiant simultanément l'état de 4 composants (refroidisseur, valve, pompe, accumulateur).
- Infrastructure complète : orchestration Airflow, tracking MLflow, API FastAPI, interface Streamlit, monitoring Prometheus/Grafana, CI/CD GitHub Actions.

---

## 🔬 Dataset

**[UCI Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems)**

- 2205 cycles × 17 capteurs (pression, température, vibration, débit, efficacité)
- Labels issus de `profile.txt` : état de 4 composants par cycle
- Cycles instables filtrés (`stable_flag == 0`)

| Target | Composant | Classes |
|--------|-----------|---------|
| `cooler_condition` | Refroidisseur | 3 (panne), 20 (réduit), 100 (ok) |
| `valve_condition` | Valve | 73 (panne), 80 (lag sévère), 90 (lag léger), 100 (ok) |
| `pump_leakage` | Pompe | 0 (aucune), 1 (faible), 2 (sévère) |
| `accumulator_pressure` | Accumulateur | 90 (panne), 100 (sévère), 115 (réduit), 130 (ok) |

---

## 🤖 Modèle

`MultiOutputClassifier(RandomForestClassifier(n_estimators=100))`

- **17 features** : PS1–PS6, EPS1, FS1–FS2, TS1–TS4, VS1, CE, CP, SE (moyenne par cycle)
- **4 targets** : classification multi-classe par composant
- **Métriques** : F1 macro, accuracy, precision, recall par target
- **Tracking** : MLflow (params, métriques, artefacts modèle, rapport JSON)

**Résultats (test set 20%) :**

| Target | Accuracy | F1 macro | F1 weighted |
|--------|----------|----------|-------------|
| `cooler_condition` | 1.000 | 1.000 | 1.000 |
| `valve_condition` | 0.948 | 0.947 | 0.948 |
| `pump_leakage` | 0.993 | 0.993 | 0.993 |
| `accumulator_pressure` | 0.990 | 0.990 | 0.990 |

---

## 🏗️ Architecture

```
[UCI Dataset]
    │  DAG: data_pipeline (@daily)
[Download → Unzip → Merge → Preprocess → Sample 80% → Trigger training]
    │  DAG: training_pipeline (event-driven)
[Train via src/train.py → Register in MLflow → Promote/Reject (F1 comparison)]
    │
[FastAPI] ← model served from models/model.pkl
    │
[Streamlit WebApp] → FastAPI → predictions → UI
    │
[Prometheus + Grafana] → API metrics (/metrics) + system metrics (node-exporter)
```

## 🐳 Services (Docker Compose)

| Service | Port | Rôle |
|---------|------|------|
| Airflow (webserver + scheduler + init) | 8080 | Orchestration (admin/admin) |
| MLflow | 5000 | Experiment tracking & model registry |
| FastAPI | 8000 | API de prédiction (`/predict`, `/health`, `/metrics`) |
| Streamlit | 8501 | Interface utilisateur (prédiction + évaluation) |
| Prometheus | 9090 | Collecte de métriques |
| Grafana | 3000 | Dashboards auto-provisionnés (admin/admin) |
| PostgreSQL | 5432 | Backend metadata Airflow |
| Node Exporter | 9100 | Métriques système |

---

## 🚀 Démarrage rapide

```bash
# Cloner et installer
git clone git@github.com:Linaa2/mlops-anomaly-detection.git
cd mlops-anomaly-detection
pip install uv && uv sync

# Tests
uv run pytest tests/ -v

# Linting
uv run ruff check .

# Lancer l'environnement dev (tous les services)
docker compose up --build

# Arrêter
docker compose down
```

**Variables d'environnement** — copier et adapter :
```bash
cp envs/.env.example envs/.env.dev
# Modifier MLFLOW_TRACKING_URI selon l'environnement
```

---

## ⚙️ CI/CD

**CI** (`.github/workflows/ci.yaml`) — déclenché à chaque push/PR :
- Python 3.11 · `uv run pytest tests/ -v` · `uv run ruff check .`
- Audit sécurité : Bandit (SAST) · Safety (dépendances) · Trivy (SARIF → onglet Security)

**CD** (`.github/workflows/cd.yml`) — déclenché sur push `main` :
1. Tests complets
2. Build & push images Docker → GHCR + DockerHub (`api` + `webapp`), tags `latest` + git SHA
3. Déploiement Kubernetes (conditionnel : ignoré si secret `KUBECONFIG` absent)

**Docs** (`.github/workflows/docs.yml`) — build Sphinx + déploiement GitHub Pages automatique.

**Dependabot** — mises à jour automatiques pip (hebdo) + Actions/Docker (mensuel).

---

## 🔄 Pipelines Airflow

### `data_pipeline` — `@daily`
```
download_dataset → unzip_dataset → merge_sensors → preprocess → sample_data (80%) → trigger_training
```
L'échantillonnage aléatoire à 80% à chaque run simule l'arrivée de nouvelles données sur un dataset statique, justifiant la comparaison champion/challenger.

### `training_pipeline` — event-driven (déclenché par `data_pipeline`)
```
train_model → promote_or_reject
```
- Délègue l'entraînement à `src/train.py` (séparation des responsabilités)
- Enregistre le modèle dans le MLflow Model Registry
- Compare le F1 macro du nouveau modèle vs le modèle Production actuel
- Promeut si meilleur, archive sinon (versioning/rollback via MLflow stages)

---

## 📚 Documentation

Documentation Sphinx déployée automatiquement sur GitHub Pages :

```bash
# Build local
uv run sphinx-apidoc -o docs/source src --force
uv run sphinx-build -b html docs/source docs/build/html
xdg-open docs/build/html/index.html
```

---

## 🧪 Tests & qualité

```bash
uv run pytest tests/ -v                              # Tests unitaires & intégration
uv run pytest --cov=src --cov-report=term-missing    # Avec couverture
uv run ruff check .                                  # Linting
uv run bandit -r src -ll                             # Sécurité SAST (optionnel)
uv run safety scan                                   # Audit dépendances (optionnel)
```

---

## 📁 Structure du projet

```
mlops-anomaly-detection/
├── .github/
│   ├── workflows/
│   │   ├── ci.yaml              # CI : pytest + ruff + sécurité
│   │   ├── cd.yml               # CD : build GHCR+DockerHub + deploy K8s
│   │   └── docs.yml             # Docs : Sphinx → GitHub Pages
│   └── dependabot.yml           # Mises à jour automatiques
├── airflow/dags/
│   ├── callbacks.py             # Alerting email sur échec (bonus)
│   ├── data_pipeline.py         # DAG : ingestion + prétraitement + sampling
│   └── training_pipeline.py     # DAG : entraînement + MLflow register + promote/reject
├── api/
│   ├── Dockerfile
│   └── app.py                   # FastAPI prediction service + Prometheus /metrics
├── webapp/
│   ├── Dockerfile
│   └── app.py                   # Streamlit UI (prédiction + évaluation modèle)
├── src/
│   ├── data_ingestion.py        # Téléchargement UCI + unzip + fusion capteurs
│   ├── preprocess.py            # Filtrage, nettoyage, features + targets
│   └── train.py                 # MultiOutput RF + MLflow tracking + export JSON
├── k8s/
│   ├── api-deployment.yaml
│   └── webapp-deployment.yaml
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/                 # Provisioning + dashboards (auto-provisionnés)
├── tests/
│   ├── test_dags.py             # Tests structure DAGs Airflow (9 tests)
│   ├── test_model.py            # Tests entraînement + prédiction (7 tests)
│   └── test_preprocessing.py   # Tests intégration preprocessing (10 tests)
├── docs/
│   ├── source/                  # Documentation Sphinx
│   ├── model_card.md            # Model card (usage, limites, éthique)
│   └── ml_pipeline.md          # Conception du pipeline ML
├── envs/
│   └── .env.example             # Template variables d'environnement
├── docker-compose.yml           # Environnement dev (8 services)
├── pyproject.toml               # Dépendances (uv)
└── ORGANISATION.md              # Plan & suivi des tâches
```

---

## ✅ Couverture des objectifs

**10/10 objectifs requis complétés** + 3 bonus :

| # | Objectif | Statut |
|---|----------|--------|
| 1 | Extraction & prétraitement des données | ✅ `data_ingestion.py` + `preprocess.py` + DAG |
| 2 | Modèle ML | ✅ MultiOutputClassifier(RF), F1 > 0.94 sur tous les targets |
| 3 | Model registry | ✅ MLflow Model Registry |
| 4 | Pipeline de réentraînement Airflow | ✅ 2 DAGs (data + training) |
| 5 | MLflow experiment tracking | ✅ params, métriques, artefacts |
| 6 | API | ✅ FastAPI, Pydantic, Swagger, `/predict`, `/health`, `/metrics` |
| 7 | WebApp | ✅ Streamlit, 2 onglets (prédiction + évaluation modèle) |
| 8 | Entraînement continu | ✅ sampling quotidien + comparaison champion/challenger F1 |
| 9 | Docker + K8s + CI/CD | ✅ Docker Compose (8 services), K8s manifests, GitHub Actions |
| 10 | Versioning GitHub & docs | ✅ 20+ PRs, commits conventionnels, README, model card |
| **12** | **Monitoring (bonus)** | ✅ Prometheus + Grafana + Node Exporter + dashboard auto-provisionné |
| **14** | **Versioning modèle/rollback (bonus)** | ✅ MLflow Registry stages (None → Production → Archived) |
| **17** | **Alerting email (bonus)** | ✅ `on_failure_callback` sur tous les DAGs |

---

## 👥 Équipe

| Membre | Périmètre |
|--------|-----------|
| A | Pipeline data & ML (`src/train.py`, intégration MLflow, export métriques) |
| B | DAGs Airflow, entraînement continu, CI/CD, tests (26), alerting, docs |
| C | API FastAPI (Pydantic), webapp Streamlit (2 onglets) |
| D | Docker Compose (8 services), K8s manifests, pipeline CD, monitoring |

Voir `ORGANISATION.md` pour le suivi détaillé des tâches.

---

## 🛠️ Stack

Python 3.11 · scikit-learn · MLflow · Apache Airflow · FastAPI · Streamlit · Docker · Kubernetes · Prometheus · Grafana · GitHub Actions · uv
