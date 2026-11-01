"""Data loading, validation, splitting and the PyTorch dataset for the PTBDB heartbeats.

Pipeline (``uv run prepare-data``)::

    data/raw/*.csv -> load_raw -> validate -> split -> data/processed/{train,val,test}.npz

``uv run make-samples`` then copies a few test-split beats to samples/ for ``uv run predict``.

Each raw row is one heartbeat: 187 signal samples scaled to [0, 1] followed by the label
(0 = normal, 1 = abnormal). The CSV files have no header.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from ecgops.utils import CONFIG_DIR, abs_path, setup_logging

SIGNAL_LENGTH = 187
N_COLUMNS = SIGNAL_LENGTH + 1  # signal samples + label
SAMPLING_RATE_HZ = 125  # PTBDB beats were resampled to 125 Hz: 187 samples = 1.5 s
CLASS_NAMES = ("normal", "abnormal")  # index = label
RAW_FILES = ("ptbdb_normal.csv", "ptbdb_abnormal.csv")
SPLITS = ("train", "val", "test")


def load_raw(raw_dir: str | Path) -> pd.DataFrame:
    """Read both PTBDB CSV files and stack them into one DataFrame (rows = heartbeats)."""
    raw_dir = Path(raw_dir)
    frames = []
    for name in RAW_FILES:
        path = raw_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `uv run dvc pull` first.")
        frame = pd.read_csv(path, header=None)
        logger.info("Loaded {}: {} rows x {} columns", name, *frame.shape)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validate_signals(signals: np.ndarray) -> None:
    """Check a (N, 187) array of beats: non-empty, no NaN, every value in [0, 1].

    Shared by ``validate`` (training data) and ``ecgops.inference`` (beats sent for
    prediction), so both apply exactly the same rules.
    """
    if signals.ndim != 2 or signals.shape[1] != SIGNAL_LENGTH or len(signals) == 0:
        raise ValueError(
            f"Expected beats of shape (N, {SIGNAL_LENGTH}) with N >= 1, got {signals.shape}."
        )
    n_missing = int(np.isnan(signals).sum())
    if n_missing:
        raise ValueError(f"Found {n_missing} missing values (NaN).")
    low, high = float(signals.min()), float(signals.max())
    if low < 0.0 or high > 1.0:
        raise ValueError(f"Signal values must lie in [0, 1]; found range [{low:.4f}, {high:.4f}].")


def validate_labels(labels: np.ndarray) -> None:
    """Check that every label is 0 (normal) or 1 (abnormal)."""
    if np.isnan(labels).any():
        raise ValueError("Found missing labels (NaN).")
    found = set(np.unique(labels).tolist())
    if not found <= {0, 1}:
        raise ValueError(f"Labels must be 0 or 1; found {sorted(found)}.")


def validate(df: pd.DataFrame) -> None:
    """Check the shape, missing values, signal range and labels; raise ValueError if broken."""
    if df.shape[1] != N_COLUMNS:
        raise ValueError(
            f"Expected {N_COLUMNS} columns ({SIGNAL_LENGTH} signal + 1 label), got {df.shape[1]}."
        )
    validate_signals(df.iloc[:, :-1].to_numpy(dtype=np.float64))
    validate_labels(df.iloc[:, -1].to_numpy(dtype=np.float64))

    counts = df.iloc[:, -1].value_counts().sort_index()
    n_normal, n_abnormal = int(counts.get(0, 0)), int(counts.get(1, 0))
    logger.info(
        "Validation passed: {} rows | normal={} ({:.1%}) | abnormal={} ({:.1%})",
        len(df),
        n_normal,
        n_normal / len(df),
        n_abnormal,
        n_abnormal / len(df),
    )


def split(
    df: pd.DataFrame, val_size: float, test_size: float, seed: int
) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test split; sizes are fractions of the whole dataset.

    Stratifying on the label keeps the ~28/72 normal/abnormal ratio identical in every split,
    so validation and test metrics are not skewed by an unlucky draw.
    """
    if not 0 < val_size + test_size < 1:
        raise ValueError("val_size + test_size must be between 0 and 1.")

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df.iloc[:, -1], random_state=seed
    )
    # val_size is relative to the full dataset, so rescale it to the remaining rows.
    val_fraction = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_fraction, stratify=train_val.iloc[:, -1], random_state=seed
    )
    return {"train": train, "val": val, "test": test}


def processed_paths(cfg: DictConfig) -> dict[str, Path]:
    """Location of each split's .npz file."""
    processed_dir = abs_path(cfg.data.processed_dir)
    return {name: processed_dir / f"{name}.npz" for name in SPLITS}


def processed_data_exists(cfg: DictConfig) -> bool:
    return all(path.exists() for path in processed_paths(cfg).values())


def prepare_data(cfg: DictConfig) -> dict[str, Path]:
    """Load -> validate -> split -> save each split as an .npz with X (N, 187) float32 and y."""
    df = load_raw(abs_path(cfg.data.raw_dir))
    validate(df)
    splits = split(df, cfg.data.val_size, cfg.data.test_size, cfg.seed)

    paths = processed_paths(cfg)
    for name, part in splits.items():
        features = part.iloc[:, :-1].to_numpy(dtype=np.float32)
        labels = part.iloc[:, -1].to_numpy(dtype=np.int64)
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(paths[name], X=features, y=labels)
        logger.info(
            "Saved {:<5} {:>6} rows ({:.1%} abnormal) -> {}",
            name,
            len(labels),
            labels.mean(),
            paths[name],
        )
    return paths


def load_split(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one processed split back as (X, y)."""
    with np.load(path) as data:
        return data["X"], data["y"]


class ECGDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Heartbeats as tensors: x has shape (1, 187) (one channel for Conv1d), y is a float."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if features.ndim != 2 or features.shape[1] != SIGNAL_LENGTH:
            raise ValueError(
                f"Expected features of shape (N, {SIGNAL_LENGTH}), got {features.shape}."
            )
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length.")
        self.features = torch.from_numpy(features.astype(np.float32)).unsqueeze(1)
        self.labels = torch.from_numpy(labels.astype(np.float32))

    @classmethod
    def from_npz(cls, path: str | Path) -> ECGDataset:
        return cls(*load_split(path))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def make_samples(cfg: DictConfig) -> dict[str, Path]:
    """Copy a few beats from the TEST split to samples/ as small example CSV files.

    ``cfg.samples.n_per_class`` normal and abnormal beats are drawn with ``cfg.samples.seed``
    (so the files are reproducible) and shuffled. Only the test split is used, never train
    or val, so the trained model has not seen these beats. Two files are written:
    example_beats.csv in the raw PTBDB format (187 values + label, no header) and
    example_beats_unlabeled.csv with the same beats but without the label column.
    """
    test_path = processed_paths(cfg)["test"]
    if not test_path.exists():
        logger.info("{} missing - running prepare_data first.", test_path)
        prepare_data(cfg)
    features, labels = load_split(test_path)

    n_per_class = int(cfg.samples.n_per_class)
    rng = np.random.default_rng(cfg.samples.seed)
    picked = []
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < n_per_class:
            raise ValueError(f"Only {len(candidates)} beats with label {label} in {test_path}.")
        picked.append(rng.choice(candidates, size=n_per_class, replace=False))
    order = rng.permutation(np.concatenate(picked))
    beats, beat_labels = features[order], labels[order]

    out_dir = abs_path(cfg.samples.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "labeled": out_dir / "example_beats.csv",
        "unlabeled": out_dir / "example_beats_unlabeled.csv",
    }
    # newline="\n" so the files are byte-identical on Windows and Linux.
    with paths["labeled"].open("w", newline="\n") as f:
        for beat, label in zip(beats, beat_labels, strict=True):
            f.write(",".join([*(f"{v:.7g}" for v in beat), str(int(label))]) + "\n")
    with paths["unlabeled"].open("w", newline="\n") as f:
        for beat in beats:
            f.write(",".join(f"{v:.7g}" for v in beat) + "\n")
    logger.info(
        "Wrote {} test beats ({} normal, {} abnormal; test rows {}) to {} and {}",
        len(order),
        int((beat_labels == 0).sum()),
        int((beat_labels == 1).sum()),
        order.tolist(),
        *paths.values(),
    )
    return paths


@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Entry point for the ``prepare-data`` console script."""
    setup_logging()
    prepare_data(cfg)


@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base="1.3")
def samples_main(cfg: DictConfig) -> None:
    """Entry point for the ``make-samples`` console script."""
    setup_logging()
    make_samples(cfg)


if __name__ == "__main__":
    main()
