# Préparation de la présentation — Hydraulic Condition Monitoring (MLOps)

> DATA713 — 6 min de présentation + 4 min de Q&A
> Axe : justification des choix de conception, cohérence de l'architecture, retour honnête sur les limites
> Règle : affirmer les choix, expliquer ce que chaque screenshot prouve, pas de formulations vagues

---

## Structure des slides (budget 6 min)

| # | Slide | Durée | Qui parle |
|---|-------|-------|-----------|
| 1 | Titre | 15s | — |
| 2 | Contexte & Problématique | 30s | |
| 3 | Vue d'ensemble de l'architecture | 45s | |
| 4 | Pipeline ML & Continuous Training | 1m | |
| 5 | Intégration Continue (CI) | 45s | |
| 6 | API & WebApp | 45s | |
| 7 | Déploiement | 45s | |
| 8 | Monitoring & Alerting | 45s | |
| 9 | Couverture des objectifs | 30s | |

---

## Texte oral — slide par slide

### Slide 1 — Titre (15s)

> Bonjour, nous sommes Lina, Mohammed, Benjamin et Grégoire. Notre projet porte sur le monitoring de systèmes hydrauliques : un pipeline MLOps complet, de l'entraînement à la production.

---

### Slide 2 — Contexte & Problématique (30s)

**Ce qui est affiché :** Tableau des 4 cibles avec le nombre de classes, liste des 17 capteurs, formulation du problème.

> Le cas d'usage, c'est de la maintenance prédictive industrielle. On utilise le dataset UCI "Condition Monitoring of Hydraulic Systems" — 2205 cycles de mesures. Chaque cycle contient 17 capteurs échantillonnés à des fréquences différentes : de 1 Hz à 100 Hz, soit entre 60 et 6000 mesures par cycle selon le capteur. On agrège tout ça en une moyenne par cycle et par capteur — ça donne un vecteur de 17 features par observation.
>
> On a 4 composants à diagnostiquer simultanément : refroidisseur (3 classes), vanne (4 classes), pompe (3 classes), accumulateur (4 classes). Comme vous le voyez dans le tableau, les labels viennent de `profile.txt` — c'est un problème supervisé.
>
> On a formulé ça comme une classification multi-classe multi-output. Pas de la détection d'anomalies — les données sont labellisées, donc on peut faire un diagnostic précis par composant. C'est plus actionnable qu'un simple « anomalie oui/non », et ça nous donne des métriques quantitatives F1 par cible pour alimenter la logique de Continuous Training.

---

### Slide 3 — Vue d'ensemble de l'architecture (45s)

**Ce qui est affiché :** Diagramme TikZ de l'architecture complète — UCI Dataset → data_pipeline → training_pipeline → MLflow, FastAPI → Streamlit, Prometheus → Grafana. Cadre Docker Compose englobant les 9 services.

> Voici l'architecture globale. Tout tourne dans Docker Compose avec 10 services — on utilise des ancres YAML pour factoriser la config des 3 services Airflow.
>
> Le flux est le suivant. Le dataset UCI est ingéré par un premier DAG Airflow, le `data_pipeline`, qui tourne en cron `@daily`. Ses 6 tâches s'enchaînent linéairement : download, unzip, merge des 17 capteurs, preprocessing, échantillonnage aléatoire à 80%, puis un `TriggerDagRunOperator` qui déclenche un second DAG — le `training_pipeline` — de manière event-driven.
>
> Le `training_pipeline` a 2 tâches : `train_model` qui délègue à `src/train.py`, et `promote_or_reject` qui fait la comparaison champion/challenger dans le MLflow Model Registry.
>
> Le modèle entraîné est sérialisé en `model.pkl` et servi par une API FastAPI, elle-même consommée par une webapp Streamlit à 2 onglets. Côté observabilité, Prometheus scrape 5 cibles — l'API FastAPI, Airflow, MLflow, lui-même et le Node Exporter — et Grafana affiche un dashboard auto-provisionné avec 7 panels.
>
> Les manifests Kubernetes sont prêts pour la production. Le pipeline CD est conditionnel : un job `check-deploy` vérifie la présence du secret `KUBECONFIG` avant tout déploiement.

---

### Slide 4 — Pipeline ML & Continuous Training (1m)

**Ce qui est affiché :**
- **Colonne gauche :** Screenshot MLflow (`mlflow.jpeg`) montrant un run avec les 4 métriques F1 macro loguées. En dessous, les choix de conception (F1 macro, séparation DAG/train.py, promote seulement si F1_new > F1_prod).
- **Colonne droite :** Diagramme TikZ des 2 DAGs (6 + 2 tâches), et screenshot Airflow (`airflow1.jpeg`) montrant le graphe réel du `data_pipeline`.

> Le modèle, c'est un `MultiOutputClassifier` wrappant un `RandomForestClassifier` avec 100 estimators et `random_state=42`. Un seul fit pour les 4 cibles — pas 4 modèles séparés. Le split train/test est à 80/20 avec stratification sur la première cible.
>
> **Sur le screenshot MLflow**, vous voyez un run typique. On logue 20 métriques par run — 5 par cible : accuracy, precision weighted, recall weighted, F1 macro et F1 weighted. Plus 7 paramètres : n_estimators, test_size, random_state, n_features, n_targets, nombre d'échantillons train et test. Et 2 artefacts : le `model.pkl` et un `model_metrics.json` contenant les matrices de confusion et rapports de classification complets par cible. C'est ce JSON qui alimente l'onglet évaluation de la webapp.
>
> Les résultats : F1 macro de 1.0 sur le refroidisseur, 0.947 sur la vanne, 0.993 sur la pompe, 0.990 sur l'accumulateur. On utilise le F1 macro — pas le weighted — pour la comparaison champion/challenger, parce que le macro traite toutes les classes de manière égale, même les minoritaires.
>
> **Sur le screenshot Airflow**, vous voyez le graphe du `data_pipeline` : 6 tâches chaînées linéairement. Le `sample_data` utilise `random_state=None` — pas de seed fixe — donc chaque run quotidien voit un sous-ensemble différent du dataset. C'est ce qui donne un sens au Continuous Training sur un dataset statique : des échantillons différents produisent des modèles différents, et la comparaison F1 est pertinente.
>
> La logique promote/reject fonctionne ainsi : après l'entraînement, le DAG compare le F1 macro moyen du nouveau modèle avec celui du modèle actuellement en stage `Production` dans le MLflow Registry. S'il n'y a pas de modèle en production, le nouveau est automatiquement promu. Sinon, promotion seulement si F1_new > F1_prod. Les modèles non retenus sont archivés, pas supprimés — on garde l'historique complet.

---

### Slide 5 — Intégration Continue (CI) (45s)

**Ce qui est affiché :**
- **Colonne gauche :** Diagramme TikZ du pipeline CI (ruff → pytest → Trivy), tableau des 26 tests en 3 suites, outils utilisés.
- **Colonne droite :** Screenshot CI (`ci.png`) montrant les checks GitHub Actions verts. Screenshot Dependabot (`dependabot.png`) montrant la configuration des mises à jour automatiques.

> La CI se déclenche à chaque push sur main et à chaque pull request. **Sur le screenshot**, vous voyez les checks GitHub Actions — tout est vert.
>
> Le pipeline a en réalité 4 couches de sécurité. D'abord, ruff pour le linting. Ensuite pytest avec 26 tests et coverage par branche activé. Puis Bandit pour l'analyse statique de sécurité sur `src/`. Et enfin Trivy en scan filesystem avec severité HIGH et CRITICAL — les résultats sont uploadés en SARIF dans l'onglet Security de GitHub. On utilise aussi Safety pour scanner les CVEs de nos dépendances.
>
> Les 26 tests couvrent 3 suites. 9 tests de structure des DAGs : on vérifie les task IDs, les dépendances, les schedules — sans exécuter les DAGs. 7 tests modèle : on génère des données synthétiques avec les bonnes distributions de classes et on valide shape, classes prédites, et F1 > random. 10 tests preprocessing : merge capteurs, filtrage des cycles instables via `stable_flag`, suppression NaN, cohérence cross-module.
>
> Ce dernier point est important : on a des tests de contrat qui vérifient que les constantes `SENSORS`, `TARGETS` et `FEATURES` sont identiques entre `data_ingestion.py`, `preprocess.py` et `train.py`. Ça empêche le schema drift silencieux entre les étapes du pipeline.
>
> **Sur le screenshot Dependabot**, vous voyez la config : mises à jour automatiques des dépendances pip et des GitHub Actions. La CI est une gate obligatoire — la branch protection empêche le merge si la CI est rouge. On a plus de 30 PRs mergées sur le projet.

---

### Slide 6 — API & WebApp (45s)

**Ce qui est affiché :**
- **Colonne gauche :** Screenshot API (`api.jpeg`) montrant le Swagger UI avec les 4 endpoints documentés. Liste des endpoints et note sur Pydantic/async.
- **Colonne droite :** Screenshot Streamlit (`webapp_final.jpeg`) montrant l'onglet évaluation du modèle — métriques par cible et matrice de confusion. Description des 2 onglets.

> L'API est en FastAPI. **Sur le screenshot Swagger**, vous voyez les 4 endpoints. L'endpoint principal `POST /predict` reçoit les 17 valeurs capteurs — chaque champ est un float validé par Pydantic. En interne, ça crée un DataFrame pandas avec les bonnes colonnes et appelle `model.predict()`. Le modèle est chargé une seule fois au démarrage du serveur — pas à chaque requête. On a `/health` pour les probes K8s, et `/metrics` exposé automatiquement par `prometheus-fastapi-instrumentator` — ça génère des métriques HTTP standard : `http_requests_total`, `http_request_duration_seconds` en histogramme, taille des requêtes/réponses. Zéro config manuelle côté Prometheus.
>
> La webapp Streamlit a 2 onglets. L'onglet prédiction propose 17 `number_input` — un par capteur — et envoie un POST à l'API. L'onglet évaluation, que vous voyez **sur le screenshot**, lit le fichier `reports/model_metrics.json` généré par `src/train.py` à chaque entraînement. Pour chaque cible, il affiche 4 métriques (accuracy, precision, recall, F1 weighted) en cartes, la matrice de confusion en DataFrame, et le rapport de classification complet dans un expander. La webapp est totalement découplée du modèle — elle ne communique qu'avec l'API REST pour les prédictions et lit un JSON statique pour l'évaluation.

---

### Slide 7 — Déploiement (45s)

**Ce qui est affiché :**
- **Colonne gauche :** Liste des 9 services Docker Compose, images optimisées (multi-stage, uv, .dockerignore).
- **Colonne droite :** Manifests K8s (API 2 replicas, webapp 1 replica), description du pipeline CD. Screenshot CD (`cd.png`) montrant le pipeline GitHub Actions avec build, push, check-deploy.

> En développement, tout tourne dans Docker Compose. 10 services dont 3 pour Airflow (scheduler, webserver, triggerer), PostgreSQL 15, MLflow v3.1 avec `--serve-artifacts`, l'API, la webapp, Prometheus v2.51, Grafana v10.4, et un Node Exporter pour les métriques système. Les images Docker sont optimisées : multi-stage build pour réduire la taille finale, `uv` pour l'installation rapide des dépendances, et `.dockerignore` pour réduire le contexte de build.
>
> **Sur le screenshot CD**, vous voyez le pipeline GitHub Actions. Il se déclenche à chaque push sur main. D'abord les tests, puis le build des images Docker avec Docker Buildx et cache GitHub Actions (`type=gha,mode=max`). Chaque image reçoit 4 tags : `latest` et le SHA court du commit, en double sur GHCR et DockerHub — double registry pour la redondance.
>
> Le job `check-deploy` vérifie si le secret `KUBECONFIG` est configuré. S'il existe, le job `deploy` applique les manifests K8s : il crée le namespace `mlops`, injecte les secrets, remplace les placeholders d'image par `sed`, et lance les deployments avec `kubectl rollout status --timeout=120s`. L'API a 2 replicas avec liveness et readiness probes sur `/health`, des resource limits CPU/mémoire, et des annotations Prometheus pour le scraping automatique. La webapp a 1 replica, un service `LoadBalancer` pour l'accès externe, et elle utilise l'endpoint interne K8s `http://api-service:80` pour joindre l'API.

---

### Slide 8 — Monitoring & Alerting (45s)

**Ce qui est affiché :**
- **Colonne gauche :** Screenshot Grafana (`grafana.jpeg`) montrant le dashboard MLOps Overview avec les panels — requests/s, latence p95, error rate, predictions/min, anomaly score distribution, DAG success rate, MLflow active runs.
- **Colonne droite :** Description de Prometheus (scrape 15s, 5 cibles) et de l'alerting (callbacks email Airflow).

> Côté monitoring, Prometheus scrape 5 cibles toutes les 15 secondes : l'API FastAPI, Airflow, MLflow, lui-même et le Node Exporter pour les métriques système. Tout est configuré dans `prometheus.yml`.
>
> **Sur le screenshot Grafana**, vous voyez le dashboard "MLOps Overview" — il contient 7 panels auto-provisionnés, c'est de l'infrastructure-as-code, zéro configuration manuelle. Les requêtes PromQL sont précises : `rate(http_requests_total{job="fastapi"}[1m])` pour le RPS, `histogram_quantile(0.95, rate(..._bucket[5m])) * 1000` pour la latence p95 en millisecondes — c'est un SLI de niveau production. On a aussi le taux d'erreurs HTTP 5xx en pourcentage, le nombre de prédictions par minute, la distribution des anomaly scores, le taux de succès des DAGs Airflow, et le nombre de runs MLflow actifs. La datasource Prometheus est aussi auto-provisionnée au `docker compose up`.
>
> Pour l'alerting, on utilise les callbacks Airflow — c'est un factory pattern. La fonction `build_failure_callback(email)` retourne une closure qui, en cas d'échec d'une tâche, envoie un email HTML avec le DAG ID, la tâche en erreur, la date d'exécution et un lien vers les logs. Le tout via `airflow.utils.email.send_email`. Si l'envoi échoue, ça log l'exception mais ne crashe pas le DAG — dégradation gracieuse.

---

### Slide 9 — Couverture des objectifs (30s)

**Ce qui est affiché :** Tableau des 10 objectifs requis (tous verts), tableau des bonus (3 réalisés sur 8), screenshot Sphinx (`sphinx_docs.png`) montrant la documentation technique auto-générée.

> Pour conclure, on a couvert les 10 objectifs requis du barème — extraction et prétraitement des données, modèle ML, model registry MLflow, pipeline Airflow de réentraînement, experiment tracking, API FastAPI avec Swagger, webapp Streamlit à deux onglets, Continuous Training, Docker plus K8s plus CI/CD, et versionnage GitHub avec documentation.
>
> **Sur le screenshot Sphinx**, vous voyez la documentation technique auto-générée depuis les docstrings du code via `sphinx-build`. Elle est déployée automatiquement sur GitHub Pages par un workflow dédié.
>
> En bonus, on en a réalisé 3 sur 8 : le monitoring avec Prometheus et Grafana (bonus #12), le versionnage de modèle avec rollback via le MLflow Registry avec les stages None/Staging/Production/Archived (bonus #14), et l'alerting email via les callbacks factory Airflow (bonus #17). Merci, on est disponibles pour vos questions.

---

## Q&A anticipées (en français)

### « Pourquoi classification supervisée plutôt que détection d'anomalies ? »

Le dataset est labellisé via `profile.txt` pour les 4 composants avec des classes explicites (ex : vanne à 73%, 80%, 90%, 100% d'ouverture). La classification supervisée donne un diagnostic actionnable par composant — pas juste « anomalie oui/non ». Et on a des métriques quantitatives (F1 macro par cible) pour alimenter le mécanisme promote/reject du Continuous Training. En production, si on recevait des données sans labels, on passerait à de la détection d'anomalies — mais ce n'est pas le cas ici.

### « Pourquoi échantillonner aléatoirement 80% sur un dataset statique ? »

Le dataset fait 2205 cycles et ne change pas. L'échantillonnage à 80% avec `random_state=None` — pas de seed fixe — à chaque run du `data_pipeline` simule la variabilité des données en production. Chaque DAG run voit un sous-ensemble différent, donc le modèle entraîné est différent, et la comparaison champion/challenger via le F1 macro moyen est pertinente. C'est un compromis pragmatique pour démontrer le Continuous Training sur un dataset statique.

### « Comment fonctionne le mécanisme promote/reject ? »

1. `train_and_log()` entraîne sur les données échantillonnées, logue 20 métriques + 7 paramètres + 2 artefacts dans MLflow, et enregistre le modèle dans le Model Registry avec `mlflow.register_model()`.
2. `promote_or_reject()` récupère le `run_id` via XCom, calcule le F1 macro moyen (moyenne des 4 `f1_macro_{target}`), et le compare au F1 du modèle en stage `Production`.
3. S'il n'y a pas encore de modèle en production → promotion automatique. Si `F1_new > F1_prod` → promotion avec `archive_existing_versions=True` (l'ancien passe en Archived). Sinon → le nouveau est archivé. On gère aussi l'exception `RESOURCE_DOES_NOT_EXIST` de MLflow quand le modèle n'existe pas encore dans le Registry.

### « Pourquoi Airflow et pas Prefect ou Dagster ? »

C'est un requirement du cours, mais c'est un choix cohérent pour notre cas d'usage : orchestration de pipelines batch avec des dépendances explicites. On utilise le `LocalExecutor` avec PostgreSQL comme metadata DB, le `TriggerDagRunOperator` pour le couplage event-driven entre les 2 DAGs, et les callbacks factory pour l'alerting. L'écosystème est mature et très utilisé en industrie.

### « Pourquoi Docker Compose plutôt que de lancer les services nativement ? »

10 services configurés de manière cohérente dans un seul fichier avec des ancres YAML pour la factorisation. Reproductible sur les machines de chaque membre de l'équipe. Chaîne de dépendances correcte : postgres → airflow, mlflow → api → webapp, prometheus → grafana. Et `depends_on` avec `condition: service_healthy` pour PostgreSQL, donc Airflow ne démarre pas tant que la DB n'est pas ready. Un `docker compose up` vs installer manuellement PostgreSQL 15, Airflow 2.9, MLflow 3.1, etc.

### « Pourquoi `uv` plutôt que `pip` ? »

10 à 100x plus rapide pour la résolution de dépendances. Lockfile (`uv.lock`) pour des builds reproductibles. Drop-in replacement pour pip/virtualenv. On l'utilise aussi dans la CI (`astral-sh/setup-uv@v5` avec version épinglée `0.10.9`) et dans les Dockerfiles pour l'installation des dépendances. C'est cohérent du dev à la CI à la prod.

### « Comment gérez-vous le versionnage / rollback des modèles ? »

Via le MLflow Model Registry avec les stages : None → Staging → Production → Archived. Chaque modèle enregistré garde ses 20 métriques, 7 paramètres, et les artefacts complets (model.pkl + model_metrics.json). Rollback = remettre la version précédente au stage Production via l'API MLflow. Les anciens modèles ne sont jamais supprimés — `archive_existing_versions=True` lors de la promotion, pas delete.

### « Et le drift monitoring ? »

Pas implémenté — dataset statique, pas de sens dans notre contexte. En production, on ajouterait des tests statistiques (KS test, PSI) sur les distributions des 17 capteurs entrants vs les distributions d'entraînement. Ça pourrait être un DAG Airflow supplémentaire qui scrute les distributions et déclenche le réentraînement en cas de drift détecté. Côté concept drift, on surveillerait les métriques de performance du modèle dans Grafana (le panel "predictions/min" est déjà là comme base).

### « Quels tests avez-vous ? »

En réalité on a ~41 tests répartis en 7 fichiers, pas 26. Les 3 suites principales : tests DAGs (structure, task IDs, dépendances, schedules — sans exécuter les DAGs), tests modèle (données synthétiques avec les bonnes distributions de classes, validation shape/classes/F1), et tests preprocessing (merge, filtre `stable_flag`, NaN, création de répertoires). En plus : tests d'ingestion (idempotence du download/unzip, recherche de fichiers), tests de train.py (avec mock complet de MLflow), tests API (avec `TestClient` et modèle mocké via `importlib`), et surtout des tests de contrat cross-module qui vérifient que SENSORS, TARGETS et FEATURES sont identiques entre les 3 modules du pipeline. Tous les tests utilisent `tmp_path` et `unittest.mock.patch` pour l'isolation complète.

### « Pourquoi l'API charge un fichier local au lieu du MLflow Registry ? »

Choix pragmatique pour le développement : `models/model.pkl` est chargé une seule fois à l'import du module (`joblib.load()` au top-level) — pas de latence réseau au démarrage. En production, l'API devrait charger depuis le MLflow Registry pour avoir le dernier modèle promu. Le DAG gère déjà le Registry avec les stages — il suffirait de modifier l'API pour interroger `client.get_latest_versions("hydraulic-anomaly-detector", stages=["Production"])`.

### « Quelles sont les performances du modèle ? »

F1 macro par cible : cooler = 1.000, valve = 0.947, pump = 0.993, accumulator = 0.990. C'est sur un split de test à 20% avec stratification sur `cooler_condition`. Le modèle exporte un `model_metrics.json` contenant accuracy, precision weighted, recall weighted, F1 macro, F1 weighted, matrices de confusion et rapports de classification sklearn complets par cible. Tout ça est visible dans l'onglet évaluation de la webapp et dans les artefacts MLflow.

### « Pourquoi pas de cluster K8s ? »

Projet scolaire — pas de budget. Mais le pipeline CD est prêt et le déploiement est conditionnel : le job `check-deploy` teste la présence du secret `KUBECONFIG`, et le job `deploy` ne s'exécute que si `has_kubeconfig == 'true'`. Les manifests K8s sont écrits avec des resource limits (256Mi/250m requests, 512Mi/500m limits), des probes liveness/readiness, et des annotations Prometheus pour le scraping auto. L'API a 2 replicas, la webapp 1. En ajoutant le secret, le déploiement se fait automatiquement.

### « Comment est séparé le code ? »

Séparation stricte des responsabilités : `src/` = logique métier (train, preprocess, data_ingestion), `airflow/dags/` = orchestration uniquement (les DAGs appellent les fonctions de `src/`, ils n'implémentent pas la logique). `api/` = serving (FastAPI), `webapp/` = interface utilisateur (Streamlit), `monitoring/` = configs Prometheus/Grafana, `k8s/` = manifests Kubernetes, `tests/` = 7 fichiers de tests isolés. Les tests de contrat entre modules empêchent le schema drift.

### « Comment fonctionne le provisioning automatique Grafana ? »

Grafana est configuré en infrastructure-as-code. Au `docker compose up`, le répertoire `monitoring/grafana/provisioning/` est monté dans le container. Il contient : un fichier datasource qui configure Prometheus comme source par défaut (`access: proxy`, `isDefault: true`, `editable: false`), un fichier dashboard provider qui pointe vers `/etc/grafana/dashboards`, et le JSON du dashboard "MLOps Overview" avec 7 panels et les requêtes PromQL pré-configurées. Tout est déclaratif, reproductible, versionné dans git.

### « Quelles métriques Prometheus expose l'API exactement ? »

La librairie `prometheus-fastapi-instrumentator` génère automatiquement : `http_requests_total` (compteur par method/status/path), `http_request_duration_seconds` (histogramme avec buckets pour les percentiles), `http_request_size_bytes` et `http_response_size_bytes`. On a aussi `model_predictions_total` visible dans le dashboard Grafana. Tout ça sans une seule ligne de code d'instrumentation manuelle dans l'API — c'est le middleware qui s'en charge.

### « Pourquoi MultiOutputClassifier plutôt que 4 modèles séparés ? »

Un seul objet sklearn à sérialisé, un seul `joblib.dump()`, un seul artefact dans MLflow, un seul `model.predict()` dans l'API. Le `MultiOutputClassifier` wrapper un `RandomForestClassifier` par cible en interne, mais ça reste un seul pipeline d'entraînement et un seul point de versionnage. En contrepartie, on perd la possibilité d'optimiser les hyperparamètres indépendamment par cible — mais pour 100 estimators sur ce dataset, la performance est déjà quasi-optimale.

---

## Checklist avant la présentation

- [ ] `docker compose up` fonctionne, tous les services healthy
- [ ] Airflow UI : les 2 DAGs visibles, graphe des tâches correct
- [ ] MLflow UI : expérience avec des runs loggés, métriques, artefacts
- [ ] API `/docs` (Swagger) accessible avec le schéma Pydantic
- [ ] WebApp onglet 1 : formulaire de prédiction fonctionne end-to-end
- [ ] WebApp onglet 2 : métriques + matrices de confusion affichées
- [ ] Grafana : dashboard auto-provisionné avec les 7 panels
- [ ] GitHub Actions : CI verte sur main
- [ ] GitHub Actions : pipeline CD visible (build → push → check-deploy)
- [ ] PDF compilé et envoyé à `prillard.martin@gmail.com`
- [ ] Répartir les slides par personne (colonne "Qui parle" ci-dessus)
- [ ] Chronomètre : vérifier que ça tient en 6 minutes
