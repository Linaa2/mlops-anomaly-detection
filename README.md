# Détection d’anomalies dans un système hydraulique

## Objectif

Ce projet vise à détecter des comportements anormaux dans un système hydraulique à partir de données issues de capteurs industriels.  
Nous mettons en place un pipeline de machine learning permettant de :

- récupérer et préparer les données,
- entraîner un modèle de détection d’anomalies,
- exposer le modèle via une API,
- préparer une intégration MLOps avec suivi, tests et conteneurisation.

## Jeu de données

Le projet utilise le dataset **Condition Monitoring of Hydraulic Systems** de l’UCI Machine Learning Repository.

Les données proviennent de plusieurs capteurs industriels, notamment :

- capteurs de pression (`PS1`, `PS2`, `PS3`)
- capteurs de température (`TS1`, `TS2`, `TS3`, `TS4`)
- capteur de vibration (`VS1`)
- variables d’efficacité et de puissance (`CE`, `CP`)

Chaque fichier capteur est traité puis fusionné dans un CSV unique.

## Pipeline du projet

Le pipeline suit les étapes suivantes :

1. **Ingestion des données**
   - téléchargement du dataset
   - décompression
   - lecture des fichiers capteurs
   - fusion en un CSV unique

2. **Prétraitement**
   - sélection des variables utiles
   - suppression des valeurs manquantes
   - sauvegarde des données nettoyées

3. **Entraînement**
   - standardisation des données avec `StandardScaler`
   - entraînement d’un modèle `IsolationForest`
   - calcul des prédictions et scores d’anomalie
   - sauvegarde du modèle et des résultats

4. **Exposition via API**
   - déploiement d’un service FastAPI
   - prédiction sur de nouvelles mesures capteurs

## Modèle utilisé

Le modèle choisi est **Isolation Forest**.

Ce choix est motivé par le fait qu’il s’agit d’un algorithme bien adapté à la détection d’anomalies dans des données tabulaires multivariées, notamment dans un contexte de surveillance industrielle.

## Stack technique

- Python
- pandas
- scikit-learn
- FastAPI
- uv
- GitHub Actions
- Ruff
- Pytest
- MLflow
- Docker

## Structure du projet

```text
mlops-anomaly-detection/
│
├── api/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── data_ingestion.py
│   ├── preprocess.py
│   └── train.py
├── tests/
├── webapp/
├── .github/workflows/
├── pyproject.toml
├── Dockerfile.api
├── Dockerfile.webapp
├── docker-compose.yml
└── README.md