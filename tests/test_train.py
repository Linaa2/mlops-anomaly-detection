"""Tests for src/train.py."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import src.train as train_module  # noqa: E402

FEATURES = train_module.FEATURES
HAS_MLFLOW = hasattr(train_module, "METRICS_PATH")
HAS_SCALER = hasattr(train_module, "SCALER_PATH")


def _make_df(n=200):
    rng = np.random.RandomState(42)
    data = {f: rng.rand(n) for f in FEATURES}
    if HAS_MLFLOW:
        data["cooler_condition"] = rng.choice([3, 20, 100], n)
        data["valve_condition"] = rng.choice([73, 80, 90, 100], n)
        data["pump_leakage"] = rng.choice([0, 1, 2], n)
        data["accumulator_pressure"] = rng.choice([90, 100, 115, 130], n)
    return pd.DataFrame(data)


def _get_patches(tmp_path):
    p = {
        "INPUT_PATH": str(tmp_path / "data.csv"),
        "MODEL_PATH": str(tmp_path / "model.pkl"),
    }
    if HAS_MLFLOW:
        p["METRICS_PATH"] = str(tmp_path / "metrics.json")
    if HAS_SCALER:
        p["SCALER_PATH"] = str(tmp_path / "scaler.pkl")
        p["OUTPUT_PATH"] = str(tmp_path / "output.csv")
    return p


def test_features_exist():
    assert len(FEATURES) > 0
    assert all(isinstance(f, str) for f in FEATURES)


def test_features_are_sensor_names():
    assert {"PS1", "PS2", "PS3"}.issubset(set(FEATURES))


def test_train_saves_model(tmp_path):
    df = _make_df()
    patches = _get_patches(tmp_path)

    with patch("pandas.read_csv", return_value=df):
        if HAS_MLFLOW:
            with (
                patch("mlflow.set_tracking_uri"),
                patch("mlflow.set_experiment"),
                patch("mlflow.start_run") as mock_run,
                patch("mlflow.log_params"),
                patch("mlflow.log_metric"),
                patch("mlflow.log_artifact"),
                patch.multiple("src.train", **patches),
            ):
                mock_run.return_value.__enter__ = MagicMock(return_value=None)
                mock_run.return_value.__exit__ = MagicMock(return_value=False)
                train_module.train()
        else:
            with patch.multiple("src.train", **patches):
                train_module.train()

    assert (tmp_path / "model.pkl").exists()
