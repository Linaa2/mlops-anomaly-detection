FROM apache/airflow:2.9.1

# Install project dependencies needed by training DAGs
# (mlflow, scikit-learn, pandas, joblib, boto3 for MinIO S3-compatible storage)
RUN pip install --no-cache-dir \
    mlflow>=2.11 \
    scikit-learn \
    pandas \
    numpy \
    joblib \
    boto3

# Copy DAGs and source code into the image (used in prod, overridden by
# bind-mounts in dev docker-compose.yml)
COPY --chown=airflow:root airflow/dags/ /opt/airflow/dags/
COPY --chown=airflow:root src/ /opt/airflow/src/

# Pre-create data directories so named volumes inherit correct ownership.
# When Docker mounts an empty named volume over these dirs, it copies
# the ownership and permissions from the image.
RUN mkdir -p /opt/airflow/data/raw /opt/airflow/data/processed \
             /opt/airflow/models \
             /opt/airflow/reports
