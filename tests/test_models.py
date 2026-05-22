"""Tests for data loading and metrics."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.cleaned_loader import load_cleaned_dataset


def test_load_cleaned_dataset_splits():
    data_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "cleaned_dataset_sentiment.csv"
    if not data_path.exists():
        pytest.skip("Cleaned dataset not available")

    train, val, test = load_cleaned_dataset(data_path=str(data_path), sample_size=1000)
    assert len(train) == 600
    assert len(val) == 200
    assert len(test) == 200
    assert set(train.columns) >= {"cleaned_text", "target"}


def test_load_cleaned_dataset_stratified():
    data_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "cleaned_dataset_sentiment.csv"
    if not data_path.exists():
        pytest.skip("Cleaned dataset not available")

    train, val, test = load_cleaned_dataset(data_path=str(data_path), sample_size=1000)
    full = pd.concat([train, val, test])
    ratio = full["target"].mean()
    assert 0.3 < ratio < 0.7
