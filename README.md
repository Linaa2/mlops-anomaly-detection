# MLOps Anomaly Detection

## Objective
Detect anomalies in system/server metrics using a simple ML pipeline.

## Stack
- Python
- scikit-learn
- MLflow
- FastAPI
- uv
- GitHub Actions
- Ruff
- Pytest

## Project structure
...

## Run locally
```bash
uv sync
uv run python src/train.py
uv run pytest
uv run uvicorn api.app:app --reload

---

## 10. Préparer la démo de demain
Il faut déjà savoir ce que tu montreras.

### Démo simple
1. montrer le repo
2. montrer le CI GitHub Actions
3. lancer l’entraînement
4. montrer MLflow
5. lancer l’API
6. faire une prédiction

Ça suffit largement pour un POC d’une journée.

---

# Ordre optimal à partir de maintenant

Je te conseille cet ordre exact :

## Bloc 1 — indispensable
1. choisir le dataset
2. mettre les données dans `data/raw`
3. coder `preprocess.py`
4. coder `train.py`

## Bloc 2 — très important
5. sauvegarder le modèle
6. coder `api/app.py`
7. coder un test simple

## Bloc 3 — finition MLOps
8. ajouter MLflow
9. finaliser README
10. push GitHub + vérifier CI

---

# Ce qui n’est pas prioritaire
Tu peux le laisser pour plus tard si tu manques de temps :

- Docker
- DVC
- déploiement cloud
- monitoring avancé
- gros notebook exploratoire

Pour demain, ce n’est pas ça qui te sauvera.  
Ce qui compte, c’est :

- repo propre
- CI vert
- modèle qui tourne
- API qui répond
- explication claire

---

# En vrai, ton minimum viable pour demain
Si tu arrives avec ça, c’est déjà très bien :

- un repo GitHub propre
- un dataset prêt
- un script d’entraînement
- un modèle sauvegardé
- une API FastAPI
- un CI GitHub Actions qui passe
- un README correct

Ça, c’est déjà un **vrai mini projet MLOps**.

Si tu veux, on peut maintenant faire la suite dans l’ordre le plus utile : **choisir tout de suite la thématique + le dataset + la structure exacte des scripts**.