"""Tests for src/train.py."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import src.train as train_module

FEATURES = train_module.FEATURES
HAS_MLFLOW = hasattr(train_module, "METRICS_PATH")
HAS_SCALER = hasattr(train_module, "SCALER_PATH")


def _make_df(n=200):
    rng = np.random.RandomState(42)
    data = {f: rng.rand(n) for f in FEATURES}
    if HAS_MLFLOW:
        TARGETS = train_module.TARGETS
        data["cooler_condition"] = rng.choice([3, 20, 100], n)
        data["valve_condition"] = rng.choice([73, 80, 90, 100], n)
        data["pump_leakage"] = rng.choice([0, 1, 2], n)
        data["accumulator_pressure"] = rng.choice([90, 100, 115, 130], n)
    return pd.DataFrame(data)


def test_features_exist():
    assert len(FEATURES) > 0
    assert all(isinstance(f, str) for f in FEATURES)


def test_train_saves_model(tmp_path):
    df = _make_df()
    model_path = tmp_path / "model.pkl"

    patches = {
        "src.train.INPUT_PATH": str(tmp_path / "data.csv"),
        "src.train.MODEL_PATH": str(model_path),
    }
    if HAS_MLFLOW:
        patches["src.train.METRICS_PATH"] = str(tmp_path / "metrics.json")
    if HAS_SCALER:
        patches["src.train.SCALER_PATH"] = str(tmp_path / "scaler.pkl")
        patches["src.train.OUTPUT_PATH"] = str(tmp_path / "output.csv")

    with patch("pandas.read_csv", return_value=df):
        if HAS_MLFLOW:
            with (
                patch("mlflow.set_tracking_uri"),
                patch("mlflow.set_experiment"),
                patch("mlflow.start_run") as mock_run,
                patch("mlflow.log_params"),
                patch("mlflow.log_metric"),
                patch("mlflow.log_artifact"),
            ):
                mock_run.return_value.__enter__ = MagicMock(return_value=None)
                mock_run.return_value.__exit__ = MagicMock(return_value=False)
                with patch.multiple("src.train", **{k.replace("src.train.", ""): v for k, v in patches.items()}):
                    train_module.train()
        else:
            with patch.multiple("src.train", **{k.replace("src.train.", ""): v for k, v in patches.items()}):
                train_module.train()

    assert model_path.exists()


def test_features_are_sensor_names():
    expected = {"PS1", "PS2", "PS3"}
    assert expected.issubset(set(FEATURES))
