# Projet MLOps - Organisation du Groupe

**Use case :** Condition monitoring de systèmes hydrauliques — classification de l'état des composants  
**Dataset :** [UCI Hydraulic Systems](https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems)  
**Groupe :** 4 personnes  
**Livrable :** Repo GitHub + présentation PDF | Présentation 6 min + 4 min Q&A

---

## Stack Technique Retenue

| Composant | Techno |
|-----------|--------|
| Orchestration pipelines | Apache Airflow |
| ML framework | Scikit-learn — MultiOutputClassifier(RandomForestClassifier) |
| Model Registry | MLflow Model Registry |
| ML Metadata Store | MLflow |
| Feature store / BDD | Volume Docker local (simple) |
| API Serving | FastAPI |
| WebApp | Streamlit |
| Conteneurisation | Docker / Docker Compose |
| CI/CD | GitHub Actions |
| Déploiement | Kubernetes (local Docker Desktop / Minikube) |
| Source repository | GitHub |
| Monitoring (bonus) | Prometheus + Grafana |
| Alerting (bonus) | SMTP (alerting mail via Airflow callbacks) |

## Dataset — Ce qu'il faut savoir

**Structure :** 2205 cycles × 17 capteurs → chaque cycle = 60 secondes de mesure  
**Données brutes :** matrices tab-delimited, une par capteur (ex: `PS1.txt` = 2205 lignes × 6000 points à 100Hz)  
**Target :** `profile.txt` — 5 colonnes par cycle :

| Colonne | Composant | Valeurs |
|---------|-----------|---------|
| 1 | Cooler condition | 3 (panne), 20 (réduit), 100 (ok) |
| 2 | Valve condition | 73 (panne), 80 (lag sévère), 90 (lag léger), 100 (ok) |
| 3 | Pump leakage | 0 (aucune), 1 (faible), 2 (sévère) |
| 4 | Accumulator pressure | 90 (panne), 100 (sévère), 115 (réduit), 130 (ok) |
| 5 | Stable flag | 0 (stable), 1 (instable — à filtrer) |

**Preprocessing clé :** filtrer les cycles instables (`stable flag = 1`) avant tout entraînement.

### Approche ML retenue

**Classification multi-classes multi-output supervisée :**
- `MultiOutputClassifier(RandomForestClassifier(n_estimators=100))`
- 17 features capteurs : `PS1-PS6, EPS1, FS1-FS2, TS1-TS4, VS1, CE, CP, SE`
- 4 targets : `cooler_condition, valve_condition, pump_leakage, accumulator_pressure`
- Labels depuis `profile.txt`
- Métriques : F1 macro par target + accuracy, precision, recall, F1 weighted
- Feature engineering : moyenne par cycle de chaque capteur (agrégation temporelle)

**Résultats obtenus (sur test set 20%) :**

| Target | Accuracy | F1 macro | F1 weighted |
|--------|----------|----------|-------------|
| cooler_condition | 1.000 | 1.000 | 1.000 |
| valve_condition | 0.948 | 0.947 | 0.948 |
| pump_leakage | 0.993 | 0.993 | 0.993 |
| accumulator_pressure | 0.990 | 0.990 | 0.990 |

---

## Architecture

```
[Dataset UCI / capteurs] 
        ↓ DAG Airflow (extraction + preprocessing)
[Volume Docker local] ── raw data ──> processed data
        ↓ DAG Airflow (training)
[MLflow Tracking] ←── métriques ──── [Training script]
[MLflow Model Registry] ←── artefacts ── [Training script]
        ↓ DAG Airflow (promote si meilleur modèle)
[FastAPI] ← modèle chargé depuis models/model.pkl
        ↓ 
[Streamlit WebApp] ── requêtes ──> FastAPI ── prédictions ──> UI
        ↓
[Prometheus + Grafana] ── métriques API + système
```

**Environnements :**
- `dev` : Docker Compose (Airflow + MLflow + FastAPI + Streamlit + Prometheus + Grafana)
- `prod` : Kubernetes local (Docker Desktop / Minikube) — deploy conditionnel si KUBECONFIG configuré

---

## État du Projet (mis à jour 11 mars 2026 — post merge PR #20)

### Composants Fonctionnels ✅

| Composant | Fichier(s) | État |
|-----------|-----------|------|
| Data ingestion | `src/data_ingestion.py` | ✅ Download UCI + unzip + merge 17 capteurs + profile.txt |
| Preprocessing | `src/preprocess.py` | ✅ Filtre stable_flag, sélection features+targets, dropna |
| Training + MLflow | `src/train.py` | ✅ MultiOutput RF, MLflow tracking (params, métriques, artifact), export JSON métriques |
| API | `api/app.py` | ✅ FastAPI, Pydantic body, 17 features, 4 targets, `/metrics` Prometheus |
| WebApp | `webapp/app.py` | ✅ Streamlit, capteurs hydrauliques, 2 onglets (prediction + model evaluation), API_URL configurable via env var |
| DAG data | `airflow/dags/data_pipeline.py` | ✅ 6 tâches : download→unzip→merge→preprocess→sample(80%)→trigger training |
| DAG training | `airflow/dags/training_pipeline.py` | ✅ 2 tâches : train_model→promote_or_reject (délègue à src/train.py) |
| Alerting | `airflow/dags/callbacks.py` | ✅ Email failure callback sur tous les DAGs |
| Tests DAGs | `tests/test_dags.py` | ✅ 9 tests (chargement, task IDs, ordre, schedules) |
| Tests modèle | `tests/test_model.py` | ✅ 7 tests (training, predictions, F1, feature/target validation) |
| Tests preprocessing | `tests/test_preprocessing.py` | ✅ 10 tests (merge, preprocess, NaN, cross-module consistency) |
| CI | `.github/workflows/ci.yaml` | ✅ Python 3.11, pytest + ruff sur push/PR |
| CD | `.github/workflows/cd.yml` | ✅ test→build-push GHCR+DockerHub→deploy K8s (conditionnel via check-deploy job) |
| K8s manifests | `k8s/` | ✅ api + webapp deployments |
| Dockerfiles | `api/Dockerfile`, `webapp/Dockerfile` | ✅ |
| Docker Compose | `docker-compose.yml` | ✅ 9 services, volumes corrigés (DAGs + Grafana) |
| Monitoring configs | `monitoring/` | ✅ Prometheus scrape (API /metrics + node-exporter) + Grafana provisioning + dashboard JSON |
| Model metrics report | `reports/model_metrics.json` | ✅ Métriques détaillées exportées par train.py (gitignored) |
| Documentation | `README.md`, `docs/model_card.md`, `docs/ml_pipeline.md` | ✅ Architecture, setup, model card, ML rationale |
| Env example | `envs/.env.example` | ✅ Variables documentées |
| Python harmonisé | `.python-version`, CI, pyproject.toml | ✅ Tout en 3.11 |

### Problèmes Résolus ✅ (historique)

| # | Problème | Résolution |
|---|----------|------------|
| 1 | `mlflow.db` (612K SQLite) commité dans le repo | `git rm --cached mlflow.db` (PR #17) |
| 2 | WebApp hardcodait `API_URL = "http://127.0.0.1:8000"` | Remplacé par `os.getenv("API_URL", "http://api:8000")` (PR #17) |
| 3 | `reports/model_metrics.json` commité | Ajouté `reports/` à .gitignore, `git rm --cached` (PR #17) |
| 4 | Pas d'endpoint `/metrics` Prometheus dans l'API | Initialisé `prometheus-fastapi-instrumentator` (PR #17) |
| 5 | CD `secrets.KUBECONFIG` invalide dans `if:` job-level | Ajouté job `check-deploy` intermédiaire (PR #19) |

### Problèmes Restants 🟡

| # | Sévérité | Problème | Notes |
|---|----------|----------|-------|
| 1 | 🟡 | API charge `models/model.pkl` en local au lieu du MLflow Registry | Acceptable pour la démo — en prod, chargerait depuis MLflow Registry |
| 2 | 🟡 | `.coverage` commité dans le repo (PR #21) | Artefact pytest, devrait être dans .gitignore |

---

## Responsabilités par Personne

### Personne A — Data & ML Pipeline
- [x] Intégrer `profile.txt` (labels des 4 composants)
- [x] Feature engineering : agréger capteurs par cycle (mean)
- [x] Filtrer les cycles instables (`stable flag = 1`)
- [x] Choix définitif modèle : `MultiOutputClassifier(RandomForestClassifier)`
- [x] Intégrer MLflow : tracking expériences + log params/métriques/artefact
- [x] Enrichir train.py : accuracy, precision, recall, F1 weighted + export JSON métriques

### Personne B — Airflow & Continuous Training
- [x] DAG `data_pipeline` : ingestion + preprocessing + random sampling → trigger training
- [x] DAG `training_pipeline` : délègue à `src/train.py`, registre modèle MLflow, promote/reject
- [x] Trigger CT : data_pipeline `@daily` déclenche training_pipeline via `TriggerDagRunOperator`
- [x] Check : nouveau modèle > ancien avant promotion Production (champion/challenger F1)
- [x] Alerting mail : `on_failure_callback` sur tous les DAGs
- [x] Tests DAGs (9) + Tests modèle (7) + Tests preprocessing (10) = 26 tests
- [x] CD pipeline : restauration build-push DockerHub + deploy K8s conditionnel
- [x] Refactoring DAG training → délègue à `src/train.py` (source unique de vérité)
- [x] Fix docker-compose volumes (DAGs + Grafana)
- [x] Harmonisation Python 3.11
- [x] Nettoyage pyproject.toml
- [x] Fix bug preprocess.py mkdir
- [x] README réécrit + Model Card

### Personne C — API & WebApp
- [x] Réécrire `api/app.py` : Pydantic body, 17 features, 4 targets, MultiOutputClassifier
- [x] Réécrire `webapp/app.py` : capteurs hydrauliques, 2 onglets (prediction + model evaluation)
- [x] Fix API_URL hardcodé dans webapp (127.0.0.1 → env var) — PR #17
- [x] Endpoint `/metrics` : initialiser `prometheus-fastapi-instrumentator` — PR #17
- [ ] Tests unitaires API (endpoint `/predict`, `/health`)

### Personne D — Infra & DevOps
- [x] `docker-compose.yml` : Airflow + MLflow + FastAPI + Streamlit + Prometheus + Grafana
- [x] Manifests K8s : `api-deployment.yaml`, `webapp-deployment.yaml`
- [x] GitHub Actions `cd.yml` : build → push GHCR + DockerHub → deploy K8s (conditionnel)
- [x] Fix CD : job `check-deploy` pour secrets dans `if:` — PR #19
- [x] CD : dual registry GHCR + DockerHub — PR #18
- [x] `Dockerfile` API + WebApp
- [x] `envs/.env.example`
- [x] Monitoring Prometheus + Grafana (dashboards + provisioning)
- [x] `.gitignore`

---

## TODO Restante Priorisée

### Priorité 1 — Avant la présentation 🔴

1. **Valider end-to-end `docker-compose up`** : tous les services doivent être healthy
2. **Prendre les screenshots** pour la présentation (Airflow UI, MLflow UI, Swagger, Streamlit, Grafana, GitHub CI/CD)
3. **Créer la présentation PDF** (basée sur `docs/presentation_prep.md`)

### Priorité 2 — Nice to have 🟡

4. **Tests unitaires API** : `/predict`, `/health`
5. **Supprimer `.coverage` du repo** : `git rm --cached .coverage`, ajouter à .gitignore

### Priorité 3 — Bonus non réalisés

6. Tests de charge Locust
7. Rollback modèle via MLflow stages demo

---

## Décisions Tranchées

- **Approche ML** : Classification supervisée multi-output avec Random Forest — plus de IsolationForest
- **Feature engineering** : moyenne par cycle de chaque capteur (agrégation temporelle)
- **Labels** : `profile.txt` — 4 targets multi-classes
- **Stockage données** : volume Docker local (pas MinIO)
- **Airflow schedules** : `data_pipeline` @daily avec random sampling 80% → trigger `training_pipeline` (schedule=None, event-driven)
- **MLflow** : tracking intégré dans `src/train.py`, Model Registry géré par le DAG training
- **Airflow imports** : pas de `airflow/__init__.py` (shadow le package installé), heavy imports dans les fonctions de tâche
- **API charge model.pkl local** : acceptable pour la démo, idéalement chargerait depuis MLflow Registry

---

## Liens Utiles

- [Dataset UCI Hydraulic Systems](https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems)
- [MLflow Docs](https://mlflow.org/docs/latest/index.html)
- [Airflow Docs](https://airflow.apache.org/docs/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
