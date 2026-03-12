#!/bin/bash
# ============================================================
# Create additional databases required by the stack.
# This script runs ONCE on first PostgreSQL start (empty data dir).
#
# The default database ($POSTGRES_DB = airflow) is created by
# the official postgres image entrypoint.
# We only need to create the MLflow metadata database.
# ============================================================

set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE mlflow'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
EOSQL

echo ">>> init-databases.sh: 'mlflow' database ready."
