# Préparation de la présentation — Hydraulic Condition Monitoring (MLOps)

> DATA713 — 6 min de présentation + 4 min de Q&A
> Axe : justification des choix de conception, cohérence de l'architecture, retour honnête sur les limites

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

**Slides backup (pour le Q&A) :**
- Backup 1 : Airflow UI (screenshots)
- Backup 2 : MLflow UI (screenshots)
- Backup 3 : API & WebApp (screenshots)
- Backup 4 : Monitoring & GitHub (screenshots)
- Backup 5 : Collaboration & Leçons apprises

---

## Texte oral — slide par slide

### Slide 1 — Titre (15s)

> Bonjour, nous sommes Lina, Mohammed, Benjamin et Grégoire. Notre projet porte sur le monitoring de systèmes hydrauliques : un pipeline MLOps complet, de l'entraînement à la production.

---

### Slide 2 — Contexte & Problématique (30s)

> Le cas d'usage, c'est de la maintenance prédictive industrielle. On utilise le dataset UCI "Condition Monitoring of Hydraulic Systems" : 2205 cycles de mesures, 17 capteurs, et 4 composants à diagnostiquer simultanément — refroidisseur, vanne, pompe et accumulateur.
>
> On a formulé le problème comme une classification multi-classe multi-output. Pourquoi pas de la détection d'anomalies ? Parce que les données sont labellisées via `profile.txt` — on peut donc faire un diagnostic précis par composant, ce qui est beaucoup plus actionnable qu'un simple « anomalie oui/non ». Et ça nous donne des métriques quantitatives pour la logique de Continuous Training.

---

### Slide 3 — Vue d'ensemble de l'architecture (45s)

> Voici l'architecture globale. Tout tourne dans Docker Compose — 9 services lancés en un seul `docker compose up`, zéro installation manuelle.
>
> Le flux est le suivant : le dataset UCI est ingéré par un premier DAG Airflow, le `data_pipeline`, qui tourne quotidiennement. Il déclenche un second DAG, le `training_pipeline`, qui entraîne le modèle et le logue dans MLflow. Le modèle entraîné est servi par une API FastAPI, consommée par une webapp Streamlit à deux onglets. Côté observabilité, Prometheus scrape les métriques de l'API et Grafana les affiche dans un dashboard auto-provisionné.
>
> On a aussi des manifests Kubernetes prêts pour la production. Le pipeline CD est conditionnel : il vérifie la présence du secret `KUBECONFIG` avant de déployer. Pas de cluster pour un projet scolaire, mais le pipeline est prêt.

---

### Slide 4 — Pipeline ML & Continuous Training (1m)

> Le modèle, c'est un `MultiOutputClassifier` avec un Random Forest en dessous. Pourquoi ? Un seul entraînement pour les 4 cibles, des features cohérentes, et un déploiement plus simple que 4 modèles séparés.
>
> Les résultats sont très bons : F1 macro de 1.0 sur le refroidisseur, 0.947 sur la vanne, 0.993 sur la pompe, et 0.990 sur l'accumulateur. On utilise le F1 macro pour la comparaison champion/challenger, parce qu'il gère bien le déséquilibre de classes.
>
> La logique de Continuous Training repose sur deux DAGs. Le `data_pipeline` tourne tous les jours : il télécharge le dataset, le dézippe, fusionne les 17 capteurs, prétraite, échantillonne 80% des données, puis déclenche le `training_pipeline`. Celui-ci entraîne le modèle — en déléguant à `src/train.py` pour séparer orchestration et logique métier — puis compare le F1 du nouveau modèle avec celui en production dans le MLflow Registry. Si le nouveau est meilleur, il est promu ; sinon, il est archivé.
>
> Pourquoi échantillonner 80% sur un dataset statique ? C'est pour simuler la variabilité des données : chaque run voit un échantillon différent, ce qui donne un sens à la comparaison champion/challenger.

---

### Slide 5 — Intégration Continue (CI) (45s)

> La CI se déclenche à chaque push et chaque pull request. Trois étapes : linting avec ruff, exécution des 26 tests avec pytest, et scan de vulnérabilités avec Trivy.
>
> Les 26 tests couvrent trois suites : 9 tests sur la structure des DAGs — task IDs, dépendances, schedules — 7 tests sur le modèle — entraînement, prédictions, validation du F1 — et 10 tests de preprocessing — fusion des capteurs, filtrage, gestion des NaN.
>
> Côté outils, on utilise `uv` comme gestionnaire de paquets — beaucoup plus rapide que pip, avec un lockfile pour la reproductibilité. On a aussi Dependabot activé pour les mises à jour de dépendances. Et la CI est une gate obligatoire : la branch protection empêche le merge si la CI est rouge.
>
> Le workflow Git, c'est feature branches vers main via PR, avec des conventional commits. On a plus de 30 PRs mergées sur le projet, avec des reviews Copilot.

---

### Slide 6 — API & WebApp (45s)

> L'API est en FastAPI. L'endpoint principal, c'est `POST /predict` : il reçoit les 17 valeurs capteurs validées par Pydantic, et retourne les 4 états des composants. On a aussi `/health` pour les sondes readiness K8s, `/metrics` pour le scraping Prometheus, et `/docs` pour le Swagger auto-généré. Pourquoi FastAPI ? Le typage natif, l'async, et Swagger inclus sans config.
>
> La webapp Streamlit a deux onglets. Le premier, c'est la prédiction : 17 sliders pour les capteurs, appel à l'API, et affichage du diagnostic des 4 composants. Le second onglet affiche l'évaluation du modèle : métriques par cible — F1, accuracy, precision, recall — matrices de confusion interactives, et rapports de classification complets. Tout vient du fichier `model_metrics.json` généré par `src/train.py` à chaque entraînement.

---

### Slide 7 — Déploiement (45s)

> En développement et pour la démo, tout tourne dans Docker Compose — 9 services : Airflow avec scheduler, webserver et triggerer, PostgreSQL pour les métadonnées Airflow, le serveur MLflow, l'API FastAPI, la webapp Streamlit, Prometheus et Grafana. Un seul `docker compose up` et tout est lancé. Les images Docker sont optimisées : multi-stage build, `uv` pour les dépendances, `.dockerignore` pour réduire le contexte.
>
> Pour la production, on a des manifests Kubernetes prêts pour l'API et la webapp. Le déploiement est conditionnel : un job `check-deploy` dans le pipeline CD vérifie que le secret `KUBECONFIG` existe avant de lancer `kubectl apply`. Les images sont taguées avec `latest` plus le SHA du commit Git, ce qui permet un rollback immédiat.
>
> Le pipeline CD se déclenche à chaque push sur main : build des images Docker, push vers GHCR et DockerHub en double registry pour la redondance, puis check-deploy et potentiellement le déploiement K8s.

---

### Slide 8 — Monitoring & Alerting (45s)

> Côté monitoring, Prometheus scrape l'endpoint `/metrics` de l'API toutes les 15 secondes. On collecte le nombre de requêtes, la latence par endpoint, les erreurs HTTP et les compteurs de prédictions.
>
> Grafana affiche ces métriques dans un dashboard auto-provisionné — il se configure tout seul au `docker compose up`, zéro intervention manuelle. Les panels montrent le taux de requêtes, les percentiles de latence (p50, p95, p99), le taux d'erreurs et l'état des services. La datasource Prometheus est aussi configurée automatiquement.
>
> Pour l'alerting, on a mis en place des callbacks email dans Airflow. Quand un DAG échoue, le `on_failure_callback` envoie un email avec le nom du DAG, la task en erreur et la date. C'est configurable par DAG, avec le SMTP paramétrable dans `airflow.cfg`. C'est le bonus numéro 17 du barème.

---

### Slide 9 — Couverture des objectifs (30s)

> Pour conclure, on a couvert les 10 objectifs requis du barème : extraction et prétraitement des données, modèle ML, model registry MLflow, pipeline Airflow de réentraînement, experiment tracking, API FastAPI avec Swagger, webapp Streamlit à deux onglets, Continuous Training, Docker plus K8s plus CI/CD, et versionnage GitHub avec documentation.
>
> En bonus, on en a réalisé 3 sur 8 : le monitoring avec Prometheus et Grafana, le versionnage de modèle avec rollback via le MLflow Registry, et l'alerting email via les callbacks Airflow. Merci, on est disponibles pour vos questions.

---

## Q&A anticipées (en français)

### « Pourquoi classification supervisée plutôt que détection d'anomalies ? »

Le dataset est labellisé via `profile.txt` pour les 4 composants. La classification supervisée donne un diagnostic actionnable par composant — pas juste « anomalie oui/non ». Et on a des métriques quantitatives (F1 par cible) pour alimenter le mécanisme promote/reject du Continuous Training.

### « Pourquoi échantillonner aléatoirement 80% sur un dataset statique ? »

Le dataset fait 2205 cycles et ne change pas. L'échantillonnage à 80% à chaque run du `data_pipeline` simule la variabilité des données. Ça donne un sens à la comparaison champion/challenger : des échantillons différents peuvent donner des performances différentes, donc la comparaison F1 est pertinente.

### « Comment fonctionne le mécanisme promote/reject ? »

1. `train_and_log()` entraîne sur les données échantillonnées, logue dans MLflow, enregistre le modèle dans le Registry
2. `promote_or_reject()` compare le F1 macro moyen du nouveau modèle vs celui du modèle en Production
3. Si F1 nouveau > F1 production : promotion en Production, l'ancien est archivé. Sinon : le nouveau est archivé.

### « Pourquoi Airflow et pas Prefect ou Dagster ? »

C'est un requirement du cours. Mais c'est aussi un bon choix : DAGs avec dépendances claires, écosystème mature, très utilisé en industrie pour les pipelines batch data/ML.

### « Pourquoi Docker Compose plutôt que de lancer les services nativement ? »

9 services configurés de manière cohérente dans un seul fichier. Reproductible sur les machines de chaque membre de l'équipe. Les mêmes noms de services qu'en production. Un `docker compose up` vs installer manuellement PostgreSQL + Airflow + MLflow + ...

### « Pourquoi `uv` plutôt que `pip` ? »

10 à 100x plus rapide pour la résolution de dépendances. Lockfile (`uv.lock`) pour des builds reproductibles. Drop-in replacement pour pip/virtualenv. Gestion native des versions Python.

### « Comment gérez-vous le versionnage / rollback des modèles ? »

Via le MLflow Model Registry avec les stages : None → Staging → Production → Archived. Les anciens modèles restent dans le registry avec leurs métriques. Rollback = remettre la version précédente au stage Production.

### « Et le drift monitoring ? »

Pas implémenté — dataset statique. En production, on ajouterait des tests statistiques (KS test, PSI) sur les distributions des capteurs entrants vs les distributions d'entraînement, avec un DAG Airflow qui déclenche le réentraînement en cas de drift détecté.

### « Quels tests avez-vous ? »

26 tests en 3 suites : tests de structure des DAGs (pas d'exécution réelle, juste validation du graphe de tâches), tests du modèle (données synthétiques, validation shape/classes/F1), et tests de preprocessing (merge, filtre, NaN, cohérence cross-module). Tous lancés dans la CI à chaque PR.

### « Pourquoi l'API charge un fichier local au lieu du MLflow Registry ? »

Choix pragmatique pour le développement : `models/model.pkl` est plus simple à charger au démarrage. En production, l'API devrait charger depuis le MLflow Registry pour avoir le dernier modèle promu. Le DAG gère déjà le registry — il suffirait de modifier l'API pour l'interroger.

### « Quelles sont les performances du modèle ? »

F1 macro par cible : cooler = 1.000, valve = 0.947, pump = 0.993, accumulator = 0.990. C'est sur un split de test à 20% avec échantillonnage stratifié. Le modèle exporte un rapport JSON détaillé avec les matrices de confusion et rapports de classification par cible.

### « Pourquoi pas de cluster K8s ? »

Projet scolaire — pas de budget pour un cluster. Mais le pipeline CD est prêt : manifests K8s écrits, job `check-deploy` conditionnel au secret `KUBECONFIG`. En ajoutant le secret, le déploiement se fait automatiquement.

### « Comment est séparé le code ? »

Séparation des responsabilités : `src/` = logique métier (train, preprocess, ingestion), `airflow/dags/` = orchestration, `api/` = serving, `webapp/` = interface utilisateur. Le DAG délègue à `src/train.py`, il n'implémente pas la logique d'entraînement lui-même.

---

## Checklist avant la présentation

- [ ] `docker compose up` fonctionne, tous les services healthy
- [ ] Airflow UI : les 2 DAGs visibles, graphe des tâches correct
- [ ] MLflow UI : expérience avec des runs loggés, métriques, artefacts
- [ ] API `/docs` (Swagger) accessible avec le schéma Pydantic
- [ ] WebApp onglet 1 : formulaire de prédiction fonctionne end-to-end
- [ ] WebApp onglet 2 : métriques + matrices de confusion affichées
- [ ] Grafana : dashboard avec les métriques
- [ ] GitHub Actions : CI verte sur main
- [ ] GitHub Actions : pipeline CD visible
- [ ] Screenshots pris et insérés dans les backup slides
- [ ] PDF compilé et envoyé à `prillard.martin@gmail.com`
