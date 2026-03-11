"""Integration tests for the preprocessing and data-ingestion pipeline.

These tests use synthetic data in tmp_path to verify the logic in
src/preprocess.py and src/data_ingestion.py without downloading the
real UCI dataset.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_ingestion import PROFILE_COLS, SENSORS, load_profile, merge_sensors
from src.preprocess import TARGETS, preprocess

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CYCLE_COUNT = 50  # number of simulated cycles


def _create_sensor_files(extract_dir: Path, n_cycles: int = CYCLE_COUNT) -> None:
    """Write one .txt file per sensor (space-separated, 60 samples/cycle)."""
    rng = np.random.RandomState(0)
    extract_dir.mkdir(parents=True, exist_ok=True)
    for sensor in SENSORS:
        data = rng.rand(n_cycles, 60)
        np.savetxt(extract_dir / f"{sensor}.txt", data, delimiter="\t")


def _create_profile_file(
    extract_dir: Path, n_cycles: int = CYCLE_COUNT, unstable_ratio: float = 0.2
) -> pd.DataFrame:
    """Write a profile.txt matching the cycle count."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(1)
    n_unstable = int(n_cycles * unstable_ratio)
    stable_flags = np.array([0] * (n_cycles - n_unstable) + [1] * n_unstable)
    rng.shuffle(stable_flags)

    profile = pd.DataFrame(
        {
            "cooler_condition": rng.choice([3, 20, 100], n_cycles),
            "valve_condition": rng.choice([73, 80, 90, 100], n_cycles),
            "pump_leakage": rng.choice([0, 1, 2], n_cycles),
            "accumulator_pressure": rng.choice([90, 100, 115, 130], n_cycles),
            "stable_flag": stable_flags,
        }
    )
    np.savetxt(
        extract_dir / "profile.txt",
        profile.values,
        delimiter="\t",
        fmt="%d",
    )
    return profile


# ---------------------------------------------------------------------------
# Tests: data_ingestion helpers
# ---------------------------------------------------------------------------


class TestMergeSensors:
    """Tests for sensor merging logic in data_ingestion.py."""

    def test_merge_produces_csv(self, tmp_path: Path) -> None:
        """merge_sensors should produce a CSV with sensors + profile columns."""
        extract_dir = tmp_path / "hydraulic"
        output_csv = tmp_path / "hydraulic_data.csv"

        _create_sensor_files(extract_dir)
        _create_profile_file(extract_dir)

        with (
            patch("src.data_ingestion.EXTRACT_DIR", extract_dir),
            patch("src.data_ingestion.OUTPUT_CSV", output_csv),
        ):
            merge_sensors()

        assert output_csv.exists(), "merge_sensors did not create the output CSV"
        df = pd.read_csv(output_csv)
        assert len(df) == CYCLE_COUNT
        assert list(df.columns) == SENSORS + PROFILE_COLS

    def test_merge_sensor_values_are_means(self, tmp_path: Path) -> None:
        """Each sensor column should be the row-wise mean of the raw file."""
        extract_dir = tmp_path / "hydraulic"
        output_csv = tmp_path / "hydraulic_data.csv"

        _create_sensor_files(extract_dir)
        _create_profile_file(extract_dir)

        with (
            patch("src.data_ingestion.EXTRACT_DIR", extract_dir),
            patch("src.data_ingestion.OUTPUT_CSV", output_csv),
        ):
            merge_sensors()

        df = pd.read_csv(output_csv)
        # Verify first sensor (PS1) — mean of 60-column raw file
        raw = np.loadtxt(extract_dir / "PS1.txt")
        expected_means = raw.mean(axis=1)
        np.testing.assert_allclose(df["PS1"].values, expected_means, rtol=1e-5)

    def test_load_profile_shape(self, tmp_path: Path) -> None:
        """load_profile should return a DataFrame with PROFILE_COLS columns."""
        extract_dir = tmp_path / "hydraulic"
        _create_profile_file(extract_dir)

        with patch("src.data_ingestion.EXTRACT_DIR", extract_dir):
            profile = load_profile()

        assert list(profile.columns) == PROFILE_COLS
        assert len(profile) == CYCLE_COUNT


# ---------------------------------------------------------------------------
# Tests: preprocessing
# ---------------------------------------------------------------------------


class TestPreprocess:
    """Tests for the preprocessing pipeline in src/preprocess.py."""

    def _create_raw_csv(self, raw_dir: Path, n_stable: int = 40, n_unstable: int = 10) -> Path:
        """Create a synthetic hydraulic_data.csv in raw_dir."""
        raw_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.RandomState(2)
        n = n_stable + n_unstable

        data = {sensor: rng.rand(n) for sensor in SENSORS}
        data["cooler_condition"] = rng.choice([3, 20, 100], n)
        data["valve_condition"] = rng.choice([73, 80, 90, 100], n)
        data["pump_leakage"] = rng.choice([0, 1, 2], n)
        data["accumulator_pressure"] = rng.choice([90, 100, 115, 130], n)
        data["stable_flag"] = [0] * n_stable + [1] * n_unstable

        csv_path = raw_dir / "hydraulic_data.csv"
        pd.DataFrame(data).to_csv(csv_path, index=False)
        return csv_path

    def test_preprocess_filters_unstable(self, tmp_path: Path) -> None:
        """preprocess() should remove rows where stable_flag != 0."""
        n_stable, n_unstable = 40, 10
        input_csv = self._create_raw_csv(tmp_path / "raw", n_stable, n_unstable)
        output_csv = tmp_path / "processed" / "hydraulic_clean.csv"

        with (
            patch("src.preprocess.INPUT_PATH", str(input_csv)),
            patch("src.preprocess.OUTPUT_PATH", str(output_csv)),
        ):
            preprocess()

        df = pd.read_csv(output_csv)
        assert len(df) == n_stable, f"Expected {n_stable} rows, got {len(df)}"

    def test_preprocess_keeps_correct_columns(self, tmp_path: Path) -> None:
        """Output should have exactly SENSORS + TARGETS columns (no stable_flag)."""
        input_csv = self._create_raw_csv(tmp_path / "raw")
        output_csv = tmp_path / "processed" / "hydraulic_clean.csv"

        with (
            patch("src.preprocess.INPUT_PATH", str(input_csv)),
            patch("src.preprocess.OUTPUT_PATH", str(output_csv)),
        ):
            preprocess()

        df = pd.read_csv(output_csv)
        expected_cols = SENSORS + TARGETS
        assert list(df.columns) == expected_cols, (
            f"Columns mismatch: {list(df.columns)} != {expected_cols}"
        )

    def test_preprocess_drops_na(self, tmp_path: Path) -> None:
        """Rows with NaN in any sensor/target column should be dropped."""
        raw_dir = tmp_path / "raw"
        input_csv = self._create_raw_csv(raw_dir, n_stable=30, n_unstable=0)

        # Inject NaN into first 5 rows of PS1
        df = pd.read_csv(input_csv)
        df.loc[:4, "PS1"] = np.nan
        df.to_csv(input_csv, index=False)

        output_csv = tmp_path / "processed" / "hydraulic_clean.csv"
        with (
            patch("src.preprocess.INPUT_PATH", str(input_csv)),
            patch("src.preprocess.OUTPUT_PATH", str(output_csv)),
        ):
            preprocess()

        result = pd.read_csv(output_csv)
        assert len(result) == 25, f"Expected 25 rows after dropping NaN, got {len(result)}"
        assert not result.isnull().any().any(), "Output still contains NaN values"

    def test_preprocess_creates_output_dir(self, tmp_path: Path) -> None:
        """preprocess() should create the output directory if it doesn't exist."""
        input_csv = self._create_raw_csv(tmp_path / "raw")
        output_csv = tmp_path / "new_dir" / "deep" / "hydraulic_clean.csv"

        assert not output_csv.parent.exists()

        with (
            patch("src.preprocess.INPUT_PATH", str(input_csv)),
            patch("src.preprocess.OUTPUT_PATH", str(output_csv)),
        ):
            preprocess()

        assert output_csv.exists()


# ---------------------------------------------------------------------------
# Tests: end-to-end ingestion -> preprocess consistency
# ---------------------------------------------------------------------------


class TestPipelineConsistency:
    """Cross-module consistency checks."""

    def test_sensor_lists_match(self) -> None:
        """SENSORS in data_ingestion and preprocess should be identical."""
        from src.data_ingestion import SENSORS as INGEST_SENSORS
        from src.preprocess import SENSORS as PREPROCESS_SENSORS

        assert INGEST_SENSORS == PREPROCESS_SENSORS, (
            f"Sensor mismatch: ingestion={INGEST_SENSORS}, preprocess={PREPROCESS_SENSORS}"
        )

    def test_target_lists_match(self) -> None:
        """TARGETS in preprocess and train should be identical."""
        from src.preprocess import TARGETS as PREPROCESS_TARGETS
        from src.train import TARGETS as TRAIN_TARGETS

        assert PREPROCESS_TARGETS == TRAIN_TARGETS, (
            f"Target mismatch: preprocess={PREPROCESS_TARGETS}, train={TRAIN_TARGETS}"
        )

    def test_features_match_sensors(self) -> None:
        """FEATURES in train.py should match SENSORS in data_ingestion/preprocess."""
        from src.data_ingestion import SENSORS as INGEST_SENSORS
        from src.train import FEATURES

        assert FEATURES == INGEST_SENSORS, (
            f"Feature/sensor mismatch: "
            f"train.FEATURES={FEATURES}, ingestion.SENSORS={INGEST_SENSORS}"
        )
