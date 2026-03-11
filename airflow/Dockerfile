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
