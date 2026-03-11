"""Tests for src/preprocess.py."""

import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import src.preprocess as preprocess_module

# Compatible avec les deux versions du module
FEATURES = getattr(preprocess_module, "FEATURES", None) or getattr(preprocess_module, "SENSORS", None)


def _make_df(n=50, with_nan=False):
    rng = np.random.RandomState(0)
    df = pd.DataFrame({f: rng.rand(n) for f in FEATURES})
    if with_nan:
        df.loc[:4, FEATURES[0]] = np.nan
    return df


def test_features_count():
    assert len(FEATURES) > 0


def test_features_are_strings():
    assert all(isinstance(f, str) for f in FEATURES)


def test_preprocess_creates_output(tmp_path):
    df = _make_df()
    output = tmp_path / "clean.csv"
    with (
        patch("pandas.read_csv", return_value=df),
        patch("src.preprocess.OUTPUT_PATH", str(output)),
    ):
        preprocess_module.preprocess()
    assert output.exists()


def test_preprocess_drops_nan(tmp_path):
    df = _make_df(with_nan=True)
    output = tmp_path / "clean.csv"
    with (
        patch("pandas.read_csv", return_value=df),
        patch("src.preprocess.OUTPUT_PATH", str(output)),
    ):
        preprocess_module.preprocess()
    result = pd.read_csv(output)
    assert not result.isnull().any().any()


def test_preprocess_output_has_correct_rows(tmp_path):
    df = _make_df(n=100)
    output = tmp_path / "clean.csv"
    with (
        patch("pandas.read_csv", return_value=df),
        patch("src.preprocess.OUTPUT_PATH", str(output)),
    ):
        preprocess_module.preprocess()
    result = pd.read_csv(output)
    assert len(result) == 100
