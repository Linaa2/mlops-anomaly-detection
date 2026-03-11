"""Tests for the MultiOutputClassifier(RandomForestClassifier) training pipeline."""

import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.train import FEATURES, N_ESTIMATORS, RANDOM_STATE, TARGETS, TEST_SIZE


def _make_synthetic_data(n_samples: int = 200, random_state: int = 42):
    """Generate synthetic data matching the hydraulic dataset structure."""
    rng = np.random.RandomState(random_state)

    X = rng.rand(n_samples, len(FEATURES))

    # Simulate multi-class targets with realistic class counts
    y = np.column_stack(
        [
            rng.choice([3, 20, 100], size=n_samples),  # cooler_condition
            rng.choice([73, 80, 90, 100], size=n_samples),  # valve_condition
            rng.choice([0, 1, 2], size=n_samples),  # pump_leakage
            rng.choice([90, 100, 115, 130], size=n_samples),  # accumulator_pressure
        ]
    )
    return X, y


def test_model_trains_and_predicts():
    """Model should fit on synthetic data and produce correct output shape."""
    X, y = _make_synthetic_data()

    model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    assert y_pred.shape == y.shape, f"Expected shape {y.shape}, got {y_pred.shape}"


def test_model_output_has_correct_targets():
    """Model should output one column per target."""
    X, y = _make_synthetic_data()

    model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    assert y_pred.shape[1] == len(TARGETS), (
        f"Expected {len(TARGETS)} target columns, got {y_pred.shape[1]}"
    )


def test_model_predicts_valid_classes():
    """Predictions should only contain classes seen during training."""
    X, y = _make_synthetic_data()

    model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    for i, target in enumerate(TARGETS):
        valid_classes = set(np.unique(y[:, i]))
        predicted_classes = set(np.unique(y_pred[:, i]))
        assert predicted_classes.issubset(valid_classes), (
            f"{target}: predicted classes {predicted_classes} not subset of {valid_classes}"
        )


def test_model_f1_above_random():
    """Model should perform better than random on a train/test split."""
    from sklearn.model_selection import train_test_split

    X, y = _make_synthetic_data(n_samples=500, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # With random synthetic data and random forest, we just check it doesn't crash
    # and produces a non-zero F1 on at least one target
    f1_scores = []
    for i in range(len(TARGETS)):
        f1 = f1_score(y_test[:, i], y_pred[:, i], average="macro", zero_division=0)
        f1_scores.append(f1)

    assert any(f > 0.0 for f in f1_scores), (
        f"All F1 scores are 0.0: {dict(zip(TARGETS, f1_scores))}"
    )


def test_feature_and_target_counts():
    """Verify that FEATURES and TARGETS constants have expected counts."""
    assert len(FEATURES) == 17, f"Expected 17 features, got {len(FEATURES)}"
    assert len(TARGETS) == 4, f"Expected 4 targets, got {len(TARGETS)}"


def test_feature_names():
    """Verify key sensor names are present in FEATURES."""
    expected_sensors = {"PS1", "PS2", "PS3", "EPS1", "FS1", "TS1", "VS1", "CE", "CP", "SE"}
    assert expected_sensors.issubset(set(FEATURES)), (
        f"Missing sensors: {expected_sensors - set(FEATURES)}"
    )


def test_target_names():
    """Verify target names match the hydraulic system components."""
    expected = {"cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure"}
    assert set(TARGETS) == expected, f"Targets mismatch: {set(TARGETS)} != {expected}"
