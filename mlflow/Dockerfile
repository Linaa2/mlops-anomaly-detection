FROM ghcr.io/mlflow/mlflow:v3.1.0

# Pre-install dependencies so the container starts instantly in prod
# (no pip install at runtime = deterministic, fast, PyPI-independent).
RUN pip install --no-cache-dir \
    boto3==1.38.0 \
    psycopg2-binary==2.9.10
