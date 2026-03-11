# Presentation Prep — Hydraulic Condition Monitoring (MLOps)

> DATA713 — 6 min presentation + 4 min Q&A
> Focus: system design justification, stack choices, architecture coherence

---

## Slide Plan (6 min budget)

### Slide 1 — Title + Context (30s)

**Content:**
- Project: Condition Monitoring of Hydraulic Systems
- Use case: multi-class multi-output classification of 4 component states
- Dataset: UCI — 2205 cycles, 17 sensors, 4 targets
- Team: 4 people

**Speaker notes:**
- Industrial predictive maintenance scenario
- Static dataset but CT pipeline designed for production-readiness
- Key point: we chose supervised classification over anomaly detection (why: labeled data available via `profile.txt`, more actionable predictions per component)

---

### Slide 2 — Architecture Overview (1m30)

**Content:** Architecture diagram showing the full system

```
                     GitHub (source)
                          |
                   CI (pytest + ruff)
                          |
                    CD (build + push)
                          |
           +--------------+--------------+
           |                             |
     Docker Compose (dev)          K8s (prod)
           |
    +------+------+------+------+------+------+
    |      |      |      |      |      |      |
  Postgres Airflow MLflow  API  Webapp Prom  Grafana
           |      |       |      |      |      |
           |      +---<---+      |      +--<---+
           |              |      |
      DAGs:               FastAPI Streamlit
      - data_pipeline     (serve) (UI: prediction
      - training_pipeline          + model eval)
```

**Key points to justify:**
1. **Docker Compose for dev** — single `docker-compose up` brings up 9 services (Postgres, Airflow webserver+scheduler+init, MLflow, FastAPI, Streamlit, Prometheus, Grafana, Node Exporter). No manual setup.
2. **K8s for prod** — manifests ready, CD pipeline conditional on `KUBECONFIG` secret (honest: no cluster for school project, but pipeline is ready)
3. **Airflow as orchestrator** — required by course, but also the right choice for DAG-based batch pipelines
4. **MLflow as metadata store + registry** — tracks experiments, stores model artifacts, enables promote/reject logic

---

### Slide 3 — ML Pipeline + Continuous Training (1m30)

**Content:**

**Model:** `MultiOutputClassifier(RandomForestClassifier)` — scikit-learn
- 17 sensor features (PS1-PS6, EPS1, FS1-FS2, TS1-TS4, VS1, CE, CP, SE)
- 4 targets: cooler condition, valve condition, pump leakage, accumulator pressure
- Metrics: F1 macro, accuracy, precision, recall per target

**Results (test set 20%):**

| Target | Accuracy | F1 macro |
|--------|----------|----------|
| Cooler condition | 1.000 | 1.000 |
| Valve condition | 0.948 | 0.947 |
| Pump leakage | 0.993 | 0.993 |
| Accumulator pressure | 0.990 | 0.990 |

**CT Logic (2 Airflow DAGs):**

```
data_pipeline (@daily)                training_pipeline (event-driven)
  download_data                         train_model (delegates to src/train.py)
  preprocess_data                       promote_or_reject (champion/challenger)
  sample_training_data (80%)
  trigger_training ──────────────────>  (schedule=None)
```

**Design decisions to justify:**

| Decision | Why |
|----------|-----|
| Random 80% sampling | Static dataset — sampling simulates data variability so promote/reject has purpose |
| MultiOutput over 4 separate models | Single training run, consistent feature set, simpler deployment |
| F1 macro comparison | Handles class imbalance (e.g., pump leakage has 3 unequal classes) |
| Promote/reject in DAG | New model promoted to Production only if F1 > current Production model |
| DAG delegates to `src/train.py` | Separation of concerns: DAG = orchestration, src/ = business logic |

**MLflow integration:**
- `src/train.py` logs params + metrics + artifact + JSON report to MLflow
- DAG's `train_and_log()` registers model in MLflow Model Registry
- `promote_or_reject()` compares new model F1 vs Production F1, promotes or archives

---

### Slide 4 — CI/CD Pipeline (45s)

**Content:**

```
PR / push
   |
   v
CI: ruff + pytest (26 tests)
   |
   v (merge to main)
CD: test -> build API + Webapp images -> push DockerHub -> deploy K8s
```

**Key points:**
- **CI triggers on every PR + push to main** — catches regressions before merge
- **CD triggers on push to main only** — test first, then build & push to DockerHub
- **K8s deploy is conditional** — `if: secrets.KUBECONFIG != ''` — graceful skip for environments without cluster
- **Image tagging**: `latest` + `git short SHA` — enables rollback
- **Package manager**: `uv` (faster than pip, lockfile for reproducibility)

**Test coverage (26 tests):**
| Suite | Count | What it tests |
|-------|-------|---------------|
| `test_dags.py` | 9 | DAG structure, task IDs, dependencies, schedules |
| `test_model.py` | 7 | Model training, predictions, F1, feature/target validation |
| `test_preprocessing.py` | 10 | Merge sensors, preprocess filtering, NaN handling, cross-module consistency |

---

### Slide 5 — API + WebApp + Monitoring (1m)

**Content:**

**API (FastAPI):**
- `/predict` — POST with JSON body (Pydantic), 17 sensor values -> 4 component states
- `/health` — readiness probe
- Swagger auto-generated at `/docs`
- Model loaded from `models/model.pkl` (trained by Airflow pipeline)

**WebApp (Streamlit) — 2 tabs:**
- **Tab 1 "Prediction"**: 17 sensor input fields -> calls API -> displays 4 predicted component states
- **Tab 2 "Model evaluation"**: displays accuracy, precision, recall, F1, confusion matrices per target from `reports/model_metrics.json`

**Monitoring (Prometheus + Grafana):**
- Prometheus scrapes node-exporter (system metrics)
- Grafana dashboard provisioned automatically (JSON + provisioning config)
- Dashboard auto-provisioned on `docker-compose up`

**Screenshot suggestions:**
- [ ] Airflow UI showing both DAGs with task graph
- [ ] MLflow UI showing experiment runs with F1 metrics
- [ ] Swagger `/docs` page with Pydantic schema
- [ ] Streamlit webapp — Tab 1 with prediction result
- [ ] Streamlit webapp — Tab 2 with model evaluation metrics + confusion matrices
- [ ] Grafana dashboard
- [ ] GitHub Actions CI/CD green checks
- [ ] GitHub PR with review + checks

---

### Slide 6 — Collaboration + Lessons Learned (45s)

**Content:**

**Team organization:**
| Person | Scope | Key deliverables |
|--------|-------|-------------------|
| A | Data + ML | `src/train.py`, MLflow integration, metrics export |
| B | Airflow + CT + CI | DAGs, CT logic, 26 tests, CI/CD quality, docs |
| C | API + WebApp | FastAPI (Pydantic), Streamlit (2 tabs), model evaluation display |
| D | Infra + DevOps | Docker Compose (9 services), K8s manifests, CD pipeline, monitoring |

**Git workflow:**
- Feature branches (`feature/`, `fix/`, `refactor/`)
- PRs to `main` with CI checks (ruff + pytest must pass)
- 16 PRs merged, atomic commits, conventional commit messages

**Honest lessons / limitations:**
- Dataset is static — CT pipeline is designed for production but trained on sampled data
- No K8s cluster available — CD pipeline is ready but deploy step is skipped
- API loads model from local file, not from MLflow Registry (acceptable for dev, would need change for prod)
- Heavy imports in Airflow DAGs cause DagBag parsing timeouts — learned to defer imports inside task functions
- Initial API/WebApp scaffold was for weather data (not hydraulic) — required full rewrite

---

## Anticipated Q&A

### "Why supervised classification instead of anomaly detection?"
The dataset comes with labeled conditions (`profile.txt`) for all 4 components. Supervised classification gives:
- Actionable predictions per component (not just "anomaly yes/no")
- Measurable metrics (F1 per target) for the promote/reject mechanism
- More meaningful CT: we can compare new vs old model quantitatively

### "Why random sampling on a static dataset?"
The dataset is fixed (2205 cycles). Random 80% sampling at each `data_pipeline` run simulates data variability. This gives purpose to the promote/reject mechanism — different samples can yield different model performance, so the comparison is meaningful.

### "How does the promote/reject mechanism work?"
1. `train_and_log()` trains on sampled data, logs to MLflow, registers model in the Registry
2. `promote_or_reject()` compares new model's mean F1 macro vs current Production model's F1
3. If new F1 > production F1: promote to Production (archive old). Otherwise: archive new model.

### "Why Airflow and not Prefect/Dagster?"
Course requirement. But also appropriate: batch DAGs with clear dependencies, mature ecosystem, widely used in industry for data/ML pipelines.

### "Why Docker Compose for dev instead of running services natively?"
- 9 services with consistent configuration in one file
- Reproducible across team members' machines
- Same service names as production (Airflow, MLflow, API, etc.)
- `docker-compose up` vs manually installing PostgreSQL + Airflow + MLflow + ...

### "Why `uv` instead of `pip`?"
- 10-100x faster dependency resolution
- Lockfile (`uv.lock`) for reproducible builds
- Drop-in replacement for pip/virtualenv
- Native Python version management

### "How do you handle model versioning / rollback?"
MLflow Model Registry with stages: None -> Staging -> Production -> Archived. Old models stay in registry with their metrics. Rollback = transition previous version back to Production stage.

### "What about data drift monitoring?"
Not implemented (static dataset). In production: would add statistical tests (KS test, PSI) on incoming sensor distributions vs training distribution, with Airflow DAG triggering retraining on drift detection.

### "What tests do you have?"
26 tests in 3 suites:
- DAG structure tests (no actual execution, just task graph validation)
- Model tests (synthetic data, shape/class/F1 validation)
- Preprocessing integration tests (merge, filter, NaN, cross-module consistency)
All run in CI on every PR.

### "Why not use MLflow's `mlflow.sklearn.log_model()` directly?"
`src/train.py` uses `mlflow.log_artifact()` for the model pickle. The DAG's `train_and_log()` then registers it in the Model Registry via `mlflow.register_model()`. This separation lets the training script stay independent of the registry workflow.

### "Why does the API load from a local file instead of MLflow Registry?"
Pragmatic choice for dev: `models/model.pkl` is simpler to load at startup. In production, the API should load from MLflow Registry to get the latest promoted model. The DAG already manages the registry — the API just needs to be updated to query it.

### "What are your model's performance results?"
F1 macro per target: cooler=1.000, valve=0.947, pump=0.993, accumulator=0.990. These are on a 20% test split with stratified sampling. The model exports a detailed JSON report with confusion matrices and classification reports per target.

---

## Status Checklist (for screenshots / demo)

- [ ] `docker-compose up` works with all services healthy
- [ ] Airflow UI: both DAGs visible, task graph correct
- [ ] MLflow UI: experiment with logged runs, metrics, artifacts
- [ ] API `/docs` (Swagger) accessible with Pydantic schema visible
- [ ] WebApp Tab 1: prediction form works end-to-end (17 sensors -> 4 states)
- [ ] WebApp Tab 2: model evaluation metrics + confusion matrices displayed
- [ ] Grafana: dashboard loads with metrics
- [ ] GitHub: CI green on main, PRs with checks
- [ ] GitHub: CD pipeline visible (even if K8s deploy skips)

---

## Open Issues to Address Before Presentation

| Priority | Issue | Owner | Status |
|----------|-------|-------|--------|
| P1 | `mlflow.db` committed to repo (612K SQLite binary) — needs `git rm --cached` | All | DONE |
| P1 | Validate `docker-compose up` end-to-end (all services healthy) | All | TODO |
| P1 | Take screenshots for slides | All | After E2E validation |
| P2 | `webapp/app.py` hardcodes `API_URL=http://127.0.0.1:8000` — won't work in Docker | C | DONE |
| P2 | No `/metrics` Prometheus endpoint in API | C | DONE |
| P3 | Add `reports/` to .gitignore (generated files) | Any | DONE |

---

## Slide Design Notes

- **Keep slides visual** — architecture diagrams > text
- **One key message per slide** — 6 min = ~1 min/slide
- **Justify, don't describe** — "we chose X because Y" not "we used X"
- **Show results** — the metrics table is impressive (F1 > 0.94 on all targets), put it on a slide
- **Be honest about limitations** — evaluators appreciate awareness over pretending everything is perfect
- **Code snippets only if asked** — no code on slides, explain concepts
- **WebApp Tab 2 is a strong visual** — shows confusion matrices and classification reports, good for a screenshot
