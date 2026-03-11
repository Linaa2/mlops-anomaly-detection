"""Tests for the FastAPI prediction API."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

mock_model = MagicMock()
mock_model.predict.return_value = np.array([[3, 100, 0, 130]])

with patch("joblib.load", return_value=mock_model):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "api.app", os.path.join(PROJECT_ROOT, "api", "app.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    app = module.app

client = TestClient(app)

# Payload compatible avec les deux versions (10 ou 17 features)
FEATURES = getattr(module, "FEATURES", [
    "PS1","PS2","PS3","PS4","PS5","PS6",
    "EPS1","FS1","FS2","TS1","TS2","TS3","TS4",
    "VS1","CE","CP","SE"
])

VALID_PAYLOAD = {f: 1.0 for f in FEATURES}

# ── GET / ──────────────────────────────────────────────────────
def test_home_returns_200():
    assert client.get("/").status_code == 200

def test_home_contains_message():
    assert "message" in client.get("/").json()

# ── GET /health ────────────────────────────────────────────────
def test_health_returns_200():
    assert client.get("/health").status_code == 200

def test_health_returns_ok():
    assert client.get("/health").json() == {"status": "ok"}

# ── POST /predict ──────────────────────────────────────────────
def test_predict_returns_200():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200, response.json()

def test_predict_has_response():
    data = client.post("/predict", json=VALID_PAYLOAD).json()
    assert isinstance(data, dict)
    assert len(data) > 0

def test_predict_missing_field_returns_422():
    payload = VALID_PAYLOAD.copy()
    del payload[FEATURES[0]]
    assert client.post("/predict", json=payload).status_code == 422

def test_predict_invalid_type_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload[FEATURES[0]] = "not_a_float"
    assert client.post("/predict", json=payload).status_code == 422

def test_predict_calls_model():
    mock_model.predict.reset_mock()
    client.post("/predict", json=VALID_PAYLOAD)
    assert mock_model.predict.call_count >= 1
