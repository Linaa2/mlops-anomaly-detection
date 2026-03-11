"""Tests for the training pipeline in src/train.py."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.train import FEATURES, INPUT_PATH, MODEL_PATH, OUTPUT_PATH, SCALER_PATH


def _make_synthetic_df(n_samples: int = 200) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame({f: rng.rand(n_samples) for f in FEATURES})


def test_features_count():
    assert len(FEATURES) == 10

def test_features_are_strings():
    assert all(isinstance(f, str) for f in FEATURES)

def test_expected_sensors_present():
    expected = {"PS1", "PS2", "PS3", "TS1", "VS1", "CE", "CP"}
    assert expected.issubset(set(FEATURES))

def test_train_saves_model(tmp_path):
    df = _make_synthetic_df()
    with (
        patch("src.train.INPUT_PATH", str(tmp_path / "data.csv")),
        patch("src.train.MODEL_PATH", str(tmp_path / "model.pkl")),
        patch("src.train.SCALER_PATH", str(tmp_path / "scaler.pkl")),
        patch("src.train.OUTPUT_PATH", str(tmp_path / "output.csv")),
        patch("pandas.read_csv", return_value=df),
    ):
        from src.train import train
        train()
    assert (tmp_path / "model.pkl").exists()

def test_train_saves_scaler(tmp_path):
    df = _make_synthetic_df()
    with (
        patch("src.train.INPUT_PATH", str(tmp_path / "data.csv")),
        patch("src.train.MODEL_PATH", str(tmp_path / "model.pkl")),
        patch("src.train.SCALER_PATH", str(tmp_path / "scaler.pkl")),
        patch("src.train.OUTPUT_PATH", str(tmp_path / "output.csv")),
        patch("pandas.read_csv", return_value=df),
    ):
        from src.train import train
        train()
    assert (tmp_path / "scaler.pkl").exists()

def test_train_saves_predictions_csv(tmp_path):
    df = _make_synthetic_df()
    with (
        patch("src.train.INPUT_PATH", str(tmp_path / "data.csv")),
        patch("src.train.MODEL_PATH", str(tmp_path / "model.pkl")),
        patch("src.train.SCALER_PATH", str(tmp_path / "scaler.pkl")),
        patch("src.train.OUTPUT_PATH", str(tmp_path / "output.csv")),
        patch("pandas.read_csv", return_value=df),
    ):
        from src.train import train
        train()
    assert (tmp_path / "output.csv").exists()

def test_train_output_has_prediction_column(tmp_path):
    df = _make_synthetic_df()
    output = tmp_path / "output.csv"
    with (
        patch("src.train.INPUT_PATH", str(tmp_path / "data.csv")),
        patch("src.train.MODEL_PATH", str(tmp_path / "model.pkl")),
        patch("src.train.SCALER_PATH", str(tmp_path / "scaler.pkl")),
        patch("src.train.OUTPUT_PATH", str(output)),
        patch("pandas.read_csv", return_value=df),
    ):
        from src.train import train
        train()
    result = pd.read_csv(output)
    assert "prediction" in result.columns
    assert "anomaly_score" in result.columns

def test_train_predictions_are_valid(tmp_path):
    df = _make_synthetic_df()
    output = tmp_path / "output.csv"
    with (
        patch("src.train.INPUT_PATH", str(tmp_path / "data.csv")),
        patch("src.train.MODEL_PATH", str(tmp_path / "model.pkl")),
        patch("src.train.SCALER_PATH", str(tmp_path / "scaler.pkl")),
        patch("src.train.OUTPUT_PATH", str(output)),
        patch("pandas.read_csv", return_value=df),
    ):
        from src.train import train
        train()
    result = pd.read_csv(output)
    assert set(result["prediction"].unique()).issubset({1, -1})

def test_train_anomaly_ratio_is_near_contamination(tmp_path):
    df = _make_synthetic_df(n_samples=500)
    output = tmp_path / "output.csv"
    with (
        patch("src.train.INPUT_PATH", str(tmp_path / "data.csv")),
        patch("src.train.MODEL_PATH", str(tmp_path / "model.pkl")),
        patch("src.train.SCALER_PATH", str(tmp_path / "scaler.pkl")),
        patch("src.train.OUTPUT_PATH", str(output)),
        patch("pandas.read_csv", return_value=df),
    ):
        from src.train import train
        train()
    result = pd.read_csv(output)
    ratio = (result["prediction"] == -1).sum() / len(result)
    assert abs(ratio - 0.05) < 0.02, f"Anomaly ratio {ratio:.2%} too far from 5%"
