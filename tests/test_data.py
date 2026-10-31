"""Tests for ecgops.data: validation, splitting and the dataset."""

import numpy as np
import pandas as pd
import pytest

from ecgops.data import N_COLUMNS, SIGNAL_LENGTH, ECGDataset, split, validate


def make_frame(n_rows: int = 100, abnormal_fraction: float = 0.7, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signals = rng.random((n_rows, SIGNAL_LENGTH))
    labels = (np.arange(n_rows) < int(n_rows * abnormal_fraction)).astype(float)
    return pd.DataFrame(np.column_stack([signals, labels]))


def test_validate_accepts_clean_data():
    validate(make_frame())


def test_validate_rejects_wrong_column_count():
    with pytest.raises(ValueError, match="columns"):
        validate(make_frame().iloc[:, :-2])


def test_validate_rejects_missing_values():
    df = make_frame()
    df.iloc[3, 5] = np.nan
    with pytest.raises(ValueError, match="missing"):
        validate(df)


def test_validate_rejects_out_of_range_signal():
    df = make_frame()
    df.iloc[0, 0] = 1.5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate(df)


def test_validate_rejects_unknown_labels():
    df = make_frame()
    df.iloc[0, -1] = 2
    with pytest.raises(ValueError, match="Labels"):
        validate(df)


def test_split_sizes_and_stratification():
    df = make_frame(n_rows=1000, abnormal_fraction=0.72)
    parts = split(df, val_size=0.15, test_size=0.15, seed=42)

    assert sum(len(p) for p in parts.values()) == len(df)
    assert len(parts["test"]) == 150
    assert len(parts["val"]) == 150
    for part in parts.values():
        assert part.iloc[:, -1].mean() == pytest.approx(0.72, abs=0.01)
    # No row ends up in two splits.
    indices = [set(p.index) for p in parts.values()]
    assert not (indices[0] & indices[1] or indices[0] & indices[2] or indices[1] & indices[2])


def test_split_is_reproducible():
    df = make_frame(n_rows=200)
    first = split(df, 0.15, 0.15, seed=1)
    second = split(df, 0.15, 0.15, seed=1)
    assert first["test"].index.equals(second["test"].index)


def test_dataset_shapes_and_types():
    df = make_frame(n_rows=10)
    dataset = ECGDataset(df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy())
    x, y = dataset[0]
    assert len(dataset) == 10
    assert x.shape == (1, SIGNAL_LENGTH)
    assert y.ndim == 0 and y.dtype.is_floating_point
    assert N_COLUMNS == SIGNAL_LENGTH + 1
