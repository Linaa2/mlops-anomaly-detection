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
- Métriques : F1 macro par target + F1 macro globale (moyenne des 4)
- Feature engineering : moyenne par cycle de chaque capteur (agrégation temporelle)

---

## Architecture

```
[Dataset UCI / capteurs] 
        ↓ DAG Airflow (extraction + preprocessing)
[Volume Docker local] ── raw data ──> processed data
        ↓ DAG Airflow (training)
[MLflow Tracking] ←── métriques ──── [Training script]
[MLflow Model Registry] ←── artefacts ── [Training script]
        ↓ DAG Airflow (deploy si meilleur modèle)
[FastAPI] ← modèle chargé depuis MLflow Registry
        ↓ 
[Streamlit WebApp] ── requêtes ──> FastAPI ── prédictions ──> UI
        ↓
[Prometheus + Grafana] ── métriques API + modèle
```

**Environnements :**
- `dev` : Docker Compose (Airflow + MLflow + FastAPI + Streamlit + Prometheus + Grafana)
- `prod` : Kubernetes local (Docker Desktop / Minikube) — deploy conditionnel si KUBECONFIG configuré

---

## État du Projet (mis à jour 11 mars 2026)

### Composants Fonctionnels ✅

| Composant | Fichier(s) | État |
|-----------|-----------|------|
| Data ingestion | `src/data_ingestion.py` | ✅ Download UCI + unzip + merge 17 capteurs + profile.txt |
| Preprocessing | `src/preprocess.py` | ✅ Filtre stable_flag, sélection features+targets, dropna |
| Training + MLflow | `src/train.py` | ✅ MultiOutput RF, MLflow tracking intégré (params, métriques F1, artifact) |
| DAG data | `airflow/dags/data_pipeline.py` | ✅ 6 tâches : download→unzip→merge→preprocess→sample(80%)→trigger training |
| DAG training | `airflow/dags/training_pipeline.py` | ✅ Délègue à src/train.py, registre modèle, promote/reject champion/challenger |
| Alerting | `airflow/dags/callbacks.py` | ✅ Email failure callback sur tous les DAGs |
| Tests DAGs | `tests/test_dags.py` | ✅ 9 tests (chargement, task IDs, ordre, schedules) |
| CI | `.github/workflows/ci.yaml` | ✅ pytest + ruff sur push/PR |
| CD | `.github/workflows/cd.yml` | ✅ test→build-push DockerHub→deploy K8s (conditionnel) |
| K8s manifests | `k8s/` | ✅ api + webapp deployments avec probes et resource limits |
| Dockerfiles | `api/Dockerfile`, `webapp/Dockerfile` | ✅ Multi-stage avec uv |
| Monitoring configs | `monitoring/` | ✅ Prometheus scrape config + Grafana provisioning + dashboard JSON |
| Docker Compose | `docker-compose.yml` | ⚠️ Fonctionnel mais bug volume DAGs (voir ci-dessous) |
| Env example | `envs/.env.example` | ✅ Variables documentées |

### Problèmes Critiques à Résoudre 🔴

| # | Problème | Fichier | Propriétaire | Détail |
|---|----------|---------|-------------|--------|
| 1 | **API cassée** : charge IsolationForest + scaler.pkl (modèle obsolète) | `api/app.py` | **Personne C** | Utilise 10 features, `decision_function()`, `scaler.pkl` — tout est incompatible avec le nouveau `MultiOutputClassifier` (17 features, 4 targets, pas de scaler) |
| 2 | **WebApp cassée** : champs météo au lieu de capteurs hydrauliques | `webapp/app.py` | **Personne C** | Envoie `temperature, humidity, wind_speed, pressure, precipitation` — aucun rapport avec les capteurs PS1-PS6, TS1-TS4, etc. |
| 3 | **Docker Compose volume DAGs** : monte `./dags` au lieu de `./airflow/dags` | `docker-compose.yml` | **Personne D** | Airflow ne verra aucun DAG en mode Docker |
| 4 | **API ne charge pas depuis MLflow** | `api/app.py` | **Personne C** | Charge `models/model.pkl` en local au lieu du modèle Production depuis MLflow Registry |

### Problèmes Secondaires 🟡

| # | Problème | Fichier | Propriétaire |
|---|----------|---------|-------------|
| 5 | Grafana dashboard JSON non monté dans le container | `docker-compose.yml` | Personne D |
| 6 | `prometheus-fastapi-instrumentator` jamais initialisé → pas de `/metrics` | `api/app.py` | Personne C |
| 7 | `tests/test_model.py` teste IsolationForest (obsolète) | `tests/test_model.py` | Personne A |
| 8 | Python version incohérente : `.python-version`=3.12, CI=3.10, CD=3.11 | Divers | Tous |
| 9 | README obsolète (IsolationForest, chemins Dockerfile faux) | `README.md` | Tous |

---

## Responsabilités par Personne

### Personne A — Data & ML Pipeline
- [x] Intégrer `profile.txt` (labels des 4 composants)
- [x] Feature engineering : agréger capteurs par cycle (mean)
- [x] Filtrer les cycles instables (`stable flag = 1`)
- [x] Choix définitif modèle : `MultiOutputClassifier(RandomForestClassifier)`
- [x] Intégrer MLflow : tracking expériences + log params/métriques/artefact
- [ ] **Mettre à jour `tests/test_model.py`** (teste encore IsolationForest)

### Personne B — Airflow & Continuous Training
- [x] DAG `data_pipeline` : ingestion + preprocessing + random sampling → trigger training
- [x] DAG `training_pipeline` : délègue à `src/train.py`, registre modèle MLflow, promote/reject
- [x] Trigger CT : data_pipeline `@daily` déclenche training_pipeline via `TriggerDagRunOperator`
- [x] Check : nouveau modèle > ancien avant promotion Production (champion/challenger F1)
- [x] Alerting mail : `on_failure_callback` sur tous les DAGs
- [x] Tests DAGs : 9 tests couvrant chargement, structure, dépendances, schedules
- [x] CD pipeline : restauration build-push DockerHub + deploy K8s conditionnel
- [x] Refactoring DAG training → délègue à `src/train.py` (source unique de vérité)

### Personne C — API & WebApp
- [ ] **🔴 CRITIQUE : Réécrire `api/app.py`** : charger MultiOutputClassifier depuis MLflow Registry, 17 features, 4 targets en sortie, body JSON Pydantic
- [ ] **🔴 CRITIQUE : Réécrire `webapp/app.py`** : champs capteurs hydrauliques (PS1-PS6, TS1-TS4, etc.) + affichage état des 4 composants
- [ ] Endpoint `/metrics` : initialiser `prometheus-fastapi-instrumentator`
- [ ] Tests unitaires API (endpoint `/predict`, `/health`)

### Personne D — Infra & DevOps
- [x] `docker-compose.yml` : Airflow + MLflow + FastAPI + Streamlit + Prometheus + Grafana
- [x] Manifests K8s : `api-deployment.yaml`, `webapp-deployment.yaml`
- [x] GitHub Actions `cd.yml` : build → push DockerHub → deploy K8s
- [x] `Dockerfile` API + WebApp
- [x] `envs/.env.example`
- [x] Monitoring Prometheus + Grafana (dashboards + provisioning)
- [ ] **🔴 Fix docker-compose.yml** : volume DAGs `./dags` → `./airflow/dags`
- [ ] **🟡 Fix Grafana dashboard mount** : monter `./monitoring/grafana/dashboards` dans le container
- [ ] **🟡 Harmoniser Python version** : choisir 3.11 ou 3.12 partout (CI, CD, Dockerfiles, .python-version)

---

## TODO Restante Priorisée

### Priorité 1 — Bloquant pour la démo 🔴

1. **Personne C** : Réécrire `api/app.py`
   - Charger le modèle Production depuis MLflow Registry (pas fichier local)
   - 17 features en entrée (body JSON Pydantic)
   - 4 targets en sortie (cooler, valve, pump, accumulator)
   - Supprimer IsolationForest / scaler / decision_function
2. **Personne C** : Réécrire `webapp/app.py`
   - Remplacer champs météo par capteurs hydrauliques
   - Afficher l'état des 4 composants
3. **Personne D** : Fix volume DAGs dans docker-compose (`./dags` → `./airflow/dags`)

### Priorité 2 — Important 🟡

4. **Personne C** : Initialiser `prometheus-fastapi-instrumentator` dans `api/app.py`
5. **Personne D** : Fix mount dashboard Grafana
6. **Personne A** : Mettre à jour `tests/test_model.py` pour le nouveau modèle
7. **Tous** : Harmoniser Python version (3.11 recommandé)

### Priorité 3 — Finitions

8. **Tous** : Mettre à jour README.md (architecture réelle, setup, screenshots)
9. **Tous** : Ajouter tests d'intégration
10. **Tous** : Model Card / documentation ML
11. **Personne B** : Vérifier que le pipeline end-to-end tourne dans Docker Compose (après fixes C+D)

---

## Décisions Tranchées

- **Approche ML** : Classification supervisée multi-output avec Random Forest — plus de IsolationForest
- **Feature engineering** : moyenne par cycle de chaque capteur (agrégation temporelle)
- **Labels** : `profile.txt` — 4 targets multi-classes
- **Stockage données** : volume Docker local (pas MinIO)
- **Airflow schedules** : `data_pipeline` @daily avec random sampling 80% → trigger `training_pipeline` (schedule=None, event-driven)
- **MLflow** : tracking intégré dans `src/train.py`, Model Registry géré par le DAG training
- **Airflow imports** : pas de `airflow/__init__.py` (shadow le package installé), heavy imports dans les fonctions de tâche

---

## Liens Utiles

- [Dataset UCI Hydraulic Systems](https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems)
- [MLflow Docs](https://mlflow.org/docs/latest/index.html)
- [Airflow Docs](https://airflow.apache.org/docs/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
