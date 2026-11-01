"""Tests for ecgops.data: validation, splitting and the dataset."""

import numpy as np
import pytest

from ecgops.data import (
    N_COLUMNS,
    SIGNAL_LENGTH,
    ECGDataset,
    load_raw,
    load_split,
    prepare_data,
    processed_data_exists,
    split,
    validate,
)


def test_validate_accepts_clean_data(make_frame):
    validate(make_frame())


def test_validate_rejects_wrong_column_count(make_frame):
    with pytest.raises(ValueError, match="columns"):
        validate(make_frame().iloc[:, :-2])


def test_validate_rejects_missing_values(make_frame):
    df = make_frame()
    df.iloc[3, 5] = np.nan
    with pytest.raises(ValueError, match="missing"):
        validate(df)


def test_validate_rejects_out_of_range_signal(make_frame):
    df = make_frame()
    df.iloc[0, 0] = 1.5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate(df)


def test_validate_rejects_unknown_labels(make_frame):
    df = make_frame()
    df.iloc[0, -1] = 2
    with pytest.raises(ValueError, match="Labels"):
        validate(df)


def test_split_sizes_and_stratification(make_frame):
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


def test_split_is_reproducible(make_frame):
    df = make_frame(n_rows=200)
    first = split(df, 0.15, 0.15, seed=1)
    second = split(df, 0.15, 0.15, seed=1)
    assert first["test"].index.equals(second["test"].index)


def test_dataset_shapes_and_types(make_frame):
    df = make_frame(n_rows=10)
    dataset = ECGDataset(df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy())
    x, y = dataset[0]
    assert len(dataset) == 10
    assert x.shape == (1, SIGNAL_LENGTH)
    assert y.ndim == 0 and y.dtype.is_floating_point
    assert N_COLUMNS == SIGNAL_LENGTH + 1


def test_load_raw_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        load_raw(tmp_path)


def test_prepare_data_writes_three_splits(tmp_path, make_frame, compose_cfg):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    make_frame(n_rows=60, abnormal_fraction=0.0, seed=1).to_csv(
        raw_dir / "ptbdb_normal.csv", header=False, index=False
    )
    make_frame(n_rows=140, abnormal_fraction=1.0, seed=2).to_csv(
        raw_dir / "ptbdb_abnormal.csv", header=False, index=False
    )
    cfg = compose_cfg(
        [f"data.raw_dir={raw_dir.as_posix()}", f"data.processed_dir={tmp_path.as_posix()}/out"]
    )

    paths = prepare_data(cfg)

    assert processed_data_exists(cfg)
    sizes = {name: len(load_split(path)[1]) for name, path in paths.items()}
    assert sum(sizes.values()) == 200
    assert sizes["test"] == 30
    assert 29 <= sizes["val"] <= 31  # sklearn rounds the rescaled val fraction up
    features, labels = load_split(paths["train"])
    assert features.dtype == np.float32 and features.shape[1] == SIGNAL_LENGTH
    assert set(np.unique(labels)) == {0, 1}
    assert len(ECGDataset.from_npz(paths["test"])) == 30
