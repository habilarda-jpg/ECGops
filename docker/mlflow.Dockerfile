# MLflow tracking server with PostgreSQL (psycopg2) and S3/MinIO (boto3) support.
# Base version matches the mlflow pinned in pyproject.toml / uv.lock.
FROM ghcr.io/mlflow/mlflow:v3.17.0

RUN pip install --no-cache-dir psycopg2-binary boto3

EXPOSE 5000
