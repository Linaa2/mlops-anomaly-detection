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
| ML framework | Scikit-learn (v1) → extension multi-output (v2) |
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
| Alerting (bonus) | SMTP (alerting mail) |

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

### Stratégie ML en 2 versions

**V1 — Classification binaire (priorité, pour avoir quelque chose de fonctionnel) :**
- Binariser chaque composant : `état dégradé/panne` vs `ok`
- 1 classificateur par composant (ou multi-output) avec Random Forest / Gradient Boosting
- Feature engineering : agréger chaque capteur par cycle (mean, std, min, max)
- Métriques : F1-score, precision, recall (focus sur le rappel des pannes)

**V2 — Classification multi-classes multi-output (enrichissement si le temps le permet) :**
- Prédire le niveau exact de dégradation pour chacun des 4 composants
- `MultiOutputClassifier(RandomForestClassifier())` scikit-learn
- Métriques : accuracy par composant + macro F1

**Feature engineering (commun aux 2 versions) :**
```
Pour chaque capteur et chaque cycle → extraire :
- mean, std, min, max, median
- percentile 5%, 95%
→ vecteur de ~100 features par cycle (au lieu de 43680 points bruts)
```

---



```
[Dataset UCI / capteurs] 
        ↓ DAG Airflow (extraction + preprocessing)
[S3/MinIO ou Volume local] ── raw data ──> processed data
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
- `dev` : Docker Compose (Airflow + MLflow + FastAPI + Streamlit)
- `prod` : Kubernetes local (Docker Desktop / Minikube) via Helm Charts

---

## État du Repo (audit initial)

### Ce qui existe et fonctionne ✅
| Fichier | État | Notes |
|---------|------|-------|
| `src/data_ingestion.py` | ✅ Fonctionnel | Download UCI + unzip + merge capteurs (mean par cycle) |
| `src/preprocess.py` | ✅ Fonctionnel | Nettoyage basique, sélection features |
| `src/train.py` | ⚠️ Incomplet | Isolation Forest OK, mais **pas de MLflow**, pas de labels profile.txt |
| `api/app.py` | ⚠️ À corriger | FastAPI OK, mais `/predict` utilise query params (pas body JSON) |
| `webapp/app.py` | ❌ À réécrire | Champs météo au lieu des capteurs hydrauliques |
| `tests/test_model.py` | ⚠️ Trivial | Test minimal, à enrichir |
| `.github/workflows/ci.yaml` | ✅ Fonctionnel | ruff + pytest sur push/PR |
| `pyproject.toml` | ✅ OK | uv, dépendances correctes |

### Ce qui manque entièrement ❌
- MLflow tracking + model registry (mentionné dans README mais absent du code)
- `profile.txt` non utilisé → les labels de condition existent mais on fait de l'unsupervised
- Airflow (aucun DAG)
- Docker Compose
- Kubernetes manifests
- Tests substantiels (unitaires, intégration, e2e)
- `.env.example`

---



### Personne A — Data & ML Pipeline
- [ ] Intégrer `profile.txt` (labels des 4 composants)
- [ ] Feature engineering : agréger capteurs par cycle (mean/std/min/max)
- [ ] Filtrer les cycles instables (`stable flag = 1`)
- [ ] Choix définitif modèle + entraînement
- [ ] **Intégrer MLflow** : tracking expériences + model registry
- [ ] DAG Airflow : extraction + preprocessing

### Personne B — Airflow & Continuous Training
- [ ] Setup Airflow (Docker Compose)
- [ ] DAG `data_pipeline` : ingestion + preprocessing → stockage
- [ ] DAG `training_pipeline` : entraînement + comparaison modèles + promotion
- [ ] Trigger CT : schedule OU dégradation performance
- [ ] Check : nouveau modèle > ancien avant promotion Production
- [ ] Alerting mail (bonus) : `on_failure_callback` + `on_success_callback`

### Personne C — API & WebApp
- [ ] **Corriger `/predict`** : body JSON (Pydantic) au lieu de query params
- [ ] Endpoint `/metrics` (Prometheus format)
- [ ] Charger modèle depuis MLflow Registry (pas fichier local)
- [ ] **Réécrire `webapp/app.py`** : champs capteurs hydrauliques + état des 4 composants
- [ ] Dockerisation API + WebApp
- [ ] Tests unitaires + intégration API

### Personne D — Infra & DevOps
- [ ] `docker-compose.yml` : Airflow + MLflow + FastAPI + Streamlit
- [ ] Manifests K8s : `api-deployment.yaml`, `webapp-deployment.yaml`
- [ ] GitHub Actions `cd.yml` : build → push DockerHub → deploy K8s
- [ ] Séparation env `dev` (Docker Compose) vs `prod` (K8s)
- [ ] Monitoring Prometheus + Grafana (bonus)

---

## TODO Globale Priorisée

### Phase 0 — Setup (30 min)
- [ ] Créer le repo GitHub (organisation + branches `main`, `dev`)
- [ ] Définir la structure des dossiers (voir ci-dessous)
- [ ] Créer le `.env.example` (variables d'env)
- [ ] Choisir l'approche ML (Isolation Forest, LOF, ou Autoencoder)
- [ ] Télécharger le dataset UCI

### Phase 0 — Setup (30 min)
- [x] Créer le repo GitHub
- [x] Structure dossiers de base (`src/`, `api/`, `webapp/`, `tests/`)
- [x] CI GitHub Actions (ruff + pytest)
- [ ] Créer branche `dev` pour le développement quotidien
- [ ] Créer `.env.example` (MLFLOW_TRACKING_URI, API_URL, etc.)
- [ ] Décider : garder Isolation Forest (non-supervisé) OU passer à RF supervisé avec `profile.txt`

### Phase 1 — Core ML (objectifs 1-3)
- [x] Ingestion données UCI (download + unzip + merge)
- [x] Preprocessing basique (dropna, sélection features)
- [ ] **Intégrer `profile.txt`** : charger les labels de condition des 4 composants
- [ ] Filtrer les cycles instables (`stable flag = 1`)
- [ ] Choix définitif approche : RF supervisé (V1 binaire) ou garder Isolation Forest
- [ ] **Ajouter MLflow** dans `train.py` : `mlflow.log_params`, `mlflow.log_metrics`, `mlflow.sklearn.log_model`
- [ ] Pusher modèle dans MLflow Model Registry (stage Staging → Production)
- [ ] Métriques : F1, precision, recall (si supervisé) ou anomaly ratio (si Isolation Forest)

### Phase 2 — Airflow Pipelines (objectifs 4 + 8)
- [ ] Installer Airflow en local (via Docker Compose)
- [ ] DAG `data_pipeline` : ingestion + preprocessing → stockage
- [ ] DAG `training_pipeline` : entraînement + log MLflow + promotion modèle
- [ ] Trigger CT : schedule hebdo OU dégradation performance
- [ ] Check : nouveau modèle > ancien avant promotion en Production

### Phase 3 — API + WebApp (objectifs 6-7)
- [x] FastAPI structure de base
- [ ] **Corriger `/predict`** : passer à un body JSON (Pydantic model) au lieu de query params
- [ ] Ajouter endpoint `/metrics` (Prometheus format)
- [ ] Charger le modèle **depuis MLflow** (pas depuis fichier local hardcodé)
- [ ] **Réécrire `webapp/app.py`** : remplacer les champs météo par les capteurs hydrauliques (PS1-PS6, TS1-TS4, etc.)
- [ ] Streamlit : afficher l'état des 4 composants (cooler, valve, pompe, accumulateur)
- [ ] Tests unitaires API (endpoint `/predict`, `/health`)
- [ ] Tests d'intégration (API + modèle)

### Phase 4 — Infra & CI/CD (objectifs 9)
- [ ] `docker-compose.yml` : Airflow + MLflow + FastAPI + Streamlit
- [ ] `Dockerfile.api` + `Dockerfile.webapp`
- [ ] Manifests K8s : `api-deployment.yaml`, `webapp-deployment.yaml`
- [ ] GitHub Actions `cd.yml` : build image → push DockerHub → deploy K8s
- [ ] Séparer env `dev` (Docker Compose) vs `prod` (K8s) dans les workflows

### Phase 5 — Documentation & Tests (objectif 10)
- [ ] Compléter le README (architecture, setup local, screenshots)
- [ ] Enrichir les tests (`tests/unit/`, `tests/integration/`)
- [ ] Model Card dans le README ou doc dédiée
- [x] Structure projet documentée

### Phase 6 — Bonus (si le temps le permet)
- [ ] **Monitoring** : Prometheus scrape sur `/metrics` de FastAPI + dashboard Grafana
- [ ] **Alerting mail** : callback Airflow `on_failure_callback` + `on_success_callback` sur réentraînement
- [ ] Rollback modèle via MLflow stages (Staging / Production / Archived)
- [ ] Tests de charge Locust sur `/predict`

---

## Structure du Repo GitHub

```
mlops-hydraulic-anomaly/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Tests + lint + build
│       └── cd.yml              # Deploy Kubernetes
├── airflow/
│   └── dags/
│       ├── data_pipeline.py
│       └── training_pipeline.py
├── data/
│   ├── raw/                    # READ ONLY
│   └── processed/
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── api/                    # FastAPI
│   └── webapp/                 # Streamlit
├── models/                     # Artefacts locaux (git-ignored)
├── kubernetes/
│   ├── api-deployment.yaml
│   └── webapp-deployment.yaml
├── monitoring/                 # Prometheus + Grafana configs (bonus)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docker-compose.yml          # Dev environment
├── .env.example
├── requirements.txt
└── README.md
```

---

## Points d'Attention

### Critères d'évaluation clés
1. **Qualité du code** - modularité, pas de notebooks en prod, type hints
2. **Bonne utilisation de l'IA Gen** - comprendre ce qu'on code, pas de code superflu
3. **Airflow + CI/CD** - pipelines fonctionnels, automatisation prouvée
4. **MLflow** - tracking + model registry bien utilisé
5. **API + WebApp** - Swagger visible, Streamlit fonctionnel

### Pièges à éviter
- Ne pas laisser des secrets dans le code (utiliser `.env`)
- Bien distinguer les exécutions `dev` vs `prod` dans la CI/CD
- S'assurer que le modèle **se charge depuis MLflow** dans l'API (pas depuis un fichier local hardcodé)
- Tester le DAG de CT : il faut que le check "nouveau modèle > ancien" soit visible

### Décisions tranchées
- **Approche ML** : V1 classification binaire (Random Forest scikit-learn) → V2 multi-output si temps
- **Feature engineering** : agrégation statistique par cycle (mean/std/min/max/percentiles) — pas de deep learning sur séries brutes
- **Stockage données** : volume Docker local (pas MinIO, trop complexe pour une journée)
- **Labels** : utiliser les labels `profile.txt` → classification supervisée (pas d'unsupervised)

---

## Liens Utiles

- [Dataset UCI Hydraulic Systems](https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems)
- [MLflow Docs](https://mlflow.org/docs/latest/index.html)
- [Airflow Docs](https://airflow.apache.org/docs/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Helm Charts Airflow](https://airflow.apache.org/docs/helm-chart/stable/index.html)
- [Helm Charts MLflow](https://github.com/community-charts/helm-charts)
