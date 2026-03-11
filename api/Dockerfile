# ============================================================
# Dockerfile — FastAPI
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Créer l'utilisateur non-root avant uv sync pour que le .venv lui appartienne
RUN useradd --create-home appuser && chown appuser /app
USER appuser

# Installer uv et les dépendances
COPY --chown=appuser pyproject.toml uv.lock ./
RUN pip install uv --user && /home/appuser/.local/bin/uv sync --frozen --no-dev

# Copier le code
COPY --chown=appuser api/ ./api/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/.venv/bin/uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
