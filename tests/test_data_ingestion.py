"""Tests for src/data_ingestion.py."""

import os
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_ingestion import SENSORS


def _create_sensor_files(extract_dir: Path, n_cycles: int = 50) -> None:
    rng = np.random.RandomState(0)
    extract_dir.mkdir(parents=True, exist_ok=True)
    for sensor in SENSORS:
        data = rng.rand(n_cycles, 60)
        np.savetxt(extract_dir / f"{sensor}.txt", data, delimiter="\t")


def test_sensors_count():
    assert len(SENSORS) == 10


def test_sensors_are_strings():
    assert all(isinstance(s, str) for s in SENSORS)


def test_download_skips_if_zip_exists(tmp_path):
    zip_path = tmp_path / "hydraulic.zip"
    zip_path.touch()
    with (
        patch("src.data_ingestion.RAW_DIR", tmp_path),
        patch("src.data_ingestion.ZIP_PATH", zip_path),
        patch("requests.get") as mock_get,
    ):
        from src.data_ingestion import download_dataset
        download_dataset()
    mock_get.assert_not_called()


def test_download_calls_requests(tmp_path):
    zip_path = tmp_path / "hydraulic.zip"
    mock_response = MagicMock()
    mock_response.content = b"fake zip content"
    with (
        patch("src.data_ingestion.RAW_DIR", tmp_path),
        patch("src.data_ingestion.ZIP_PATH", zip_path),
        patch("requests.get", return_value=mock_response) as mock_get,
    ):
        from src.data_ingestion import download_dataset
        download_dataset()
    mock_get.assert_called_once()


def test_unzip_skips_if_already_extracted(tmp_path):
    extract_dir = tmp_path / "hydraulic"
    extract_dir.mkdir()
    (extract_dir / "dummy.txt").touch()
    with (
        patch("src.data_ingestion.EXTRACT_DIR", extract_dir),
        patch("zipfile.ZipFile") as mock_zip,
    ):
        from src.data_ingestion import unzip_dataset
        unzip_dataset()
    mock_zip.assert_not_called()


def test_unzip_extracts_zip(tmp_path):
    extract_dir = tmp_path / "hydraulic"
    zip_path = tmp_path / "hydraulic.zip"

    # Créer un vrai zip
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dummy.txt", "data")

    with (
        patch("src.data_ingestion.EXTRACT_DIR", extract_dir),
        patch("src.data_ingestion.ZIP_PATH", zip_path),
    ):
        from src.data_ingestion import unzip_dataset
        unzip_dataset()

    assert extract_dir.exists()
    assert (extract_dir / "dummy.txt").exists()


def test_find_sensor_file_raises_if_missing(tmp_path):
    with patch("src.data_ingestion.EXTRACT_DIR", tmp_path):
        from src.data_ingestion import find_sensor_file
        with pytest.raises(FileNotFoundError):
            find_sensor_file("PS1")


def test_find_sensor_file_returns_path(tmp_path):
    (tmp_path / "PS1.txt").touch()
    with patch("src.data_ingestion.EXTRACT_DIR", tmp_path):
        from src.data_ingestion import find_sensor_file
        result = find_sensor_file("PS1")
    assert result == tmp_path / "PS1.txt"


def test_merge_sensors_creates_csv(tmp_path):
    extract_dir = tmp_path / "hydraulic"
    output_csv = tmp_path / "hydraulic_data.csv"
    _create_sensor_files(extract_dir)

    with (
        patch("src.data_ingestion.EXTRACT_DIR", extract_dir),
        patch("src.data_ingestion.OUTPUT_CSV", output_csv),
    ):
        from src.data_ingestion import merge_sensors
        merge_sensors()

    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    assert list(df.columns) == SENSORS
    assert len(df) == 50


def test_merge_sensors_values_are_means(tmp_path):
    extract_dir = tmp_path / "hydraulic"
    output_csv = tmp_path / "hydraulic_data.csv"
    _create_sensor_files(extract_dir)

    with (
        patch("src.data_ingestion.EXTRACT_DIR", extract_dir),
        patch("src.data_ingestion.OUTPUT_CSV", output_csv),
    ):
        from src.data_ingestion import merge_sensors
        merge_sensors()

    df = pd.read_csv(output_csv)
    raw = np.loadtxt(extract_dir / "PS1.txt")
    np.testing.assert_allclose(df["PS1"].values, raw.mean(axis=1), rtol=1e-5)
