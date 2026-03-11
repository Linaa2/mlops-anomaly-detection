"""Tests for the FastAPI anomaly detection API."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Mock modèle + scaler ───────────────────────────────────────
mock_model = MagicMock()
mock_model.predict.return_value = np.array([1])
mock_model.decision_function.return_value = np.array([-0.35])

mock_scaler = MagicMock()
mock_scaler.transform.return_value = np.zeros((1, 10))

with patch("joblib.load", side_effect=[mock_model, mock_scaler]):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "api.app", os.path.join(PROJECT_ROOT, "api", "app.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    app = module.app

client = TestClient(app)

VALID_PARAMS = {
    "PS1": 155.0, "PS2": 104.9, "PS3": 30.5,
    "TS1": 35.4,  "TS2": 40.8,  "TS3": 38.5,
    "TS4": 30.5,  "VS1": 0.52,
    "CE": 28.5,   "CP": 2.1,
}

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
    assert client.post("/predict", params=VALID_PARAMS).status_code == 200

def test_predict_returns_prediction_key():
    data = client.post("/predict", params=VALID_PARAMS).json()
    assert "prediction" in data

def test_predict_returns_anomaly_score():
    data = client.post("/predict", params=VALID_PARAMS).json()
    assert "anomaly_score" in data

def test_predict_prediction_is_int():
    data = client.post("/predict", params=VALID_PARAMS).json()
    assert isinstance(data["prediction"], int)

def test_predict_anomaly_score_is_float():
    data = client.post("/predict", params=VALID_PARAMS).json()
    assert isinstance(data["anomaly_score"], float)

def test_predict_missing_field_returns_422():
    params = VALID_PARAMS.copy()
    del params["PS1"]
    assert client.post("/predict", params=params).status_code == 422

def test_predict_calls_scaler():
    mock_scaler.transform.reset_mock()
    client.post("/predict", params=VALID_PARAMS)
    mock_scaler.transform.assert_called_once()

def test_predict_calls_model():
    mock_model.predict.reset_mock()
    client.post("/predict", params=VALID_PARAMS)
    mock_model.predict.assert_called_once()

def test_predict_calls_decision_function():
    mock_model.decision_function.reset_mock()
    client.post("/predict", params=VALID_PARAMS)
    mock_model.decision_function.assert_called_once()
