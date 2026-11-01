"""Shared fixtures. Everything is synthetic or lives in tmp_path: no data/raw, DVC or MLflow."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from ecgops.data import SIGNAL_LENGTH
from ecgops.model import build_model
from ecgops.utils import CONFIG_DIR


def _make_frame(n_rows: int = 100, abnormal_fraction: float = 0.7, seed: int = 0) -> pd.DataFrame:
    """A raw-style DataFrame: 187 signal columns in [0, 1) plus a 0/1 label column."""
    rng = np.random.default_rng(seed)
    signals = rng.random((n_rows, SIGNAL_LENGTH))
    labels = (np.arange(n_rows) < int(n_rows * abnormal_fraction)).astype(float)
    return pd.DataFrame(np.column_stack([signals, labels]))


@pytest.fixture
def make_frame() -> Callable[..., pd.DataFrame]:
    """Factory for synthetic heartbeat DataFrames (call it with n_rows=..., seed=...)."""
    return _make_frame


@pytest.fixture
def compose_cfg() -> Callable[..., DictConfig]:
    """Compose the real Hydra config from configs/ with optional CLI-style overrides."""

    def _compose(overrides: list[str] | None = None) -> DictConfig:
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
            return compose(config_name="config", overrides=overrides or [])

    return _compose


def write_synthetic_splits(processed_dir: Path, n_rows: int = 64, seed: int = 0) -> None:
    """Write tiny train/val/test .npz files in the format prepare_data produces."""
    rng = np.random.default_rng(seed)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        features = rng.random((n_rows, SIGNAL_LENGTH), dtype=np.float32)
        labels = (np.arange(n_rows) % 3 != 0).astype(np.int64)  # ~2/3 abnormal, like PTBDB
        np.savez_compressed(processed_dir / f"{name}.npz", X=features, y=labels)


@pytest.fixture
def train_cfg(tmp_path: Path, compose_cfg: Callable[..., DictConfig]) -> DictConfig:
    """A config for a tiny 1-epoch CPU run on synthetic data with all tracking switched off."""
    processed_dir = tmp_path / "processed"
    write_synthetic_splits(processed_dir)
    return compose_cfg(
        [
            f"data.processed_dir={processed_dir.as_posix()}",
            f"data.raw_dir={(tmp_path / 'raw').as_posix()}",  # must never be read
            f"model_out={(tmp_path / 'models' / 'model.pt').as_posix()}",
            "model.channels=[8,16]",
            "model.hidden_dim=16",
            "training.epochs=1",
            "training.batch_size=16",
            "tracking.mlflow.enabled=false",
            "tracking.wandb.enabled=false",
        ]
    )


SMALL_MODEL_CFG = {
    "in_channels": 1,
    "channels": [8, 16],
    "convs_per_block": 1,
    "kernel_size": 3,
    "hidden_dim": 16,
    "dropout": 0.0,
}


@pytest.fixture
def model_checkpoint(tmp_path: Path) -> Path:
    """A randomly initialised small VGG1D saved in the same format as `uv run train`."""
    torch.manual_seed(0)
    model = build_model(SMALL_MODEL_CFG)
    path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "model_cfg": SMALL_MODEL_CFG}, path)
    return path


@pytest.fixture
def beats_csv(tmp_path: Path) -> Callable[..., tuple[Path, np.ndarray, np.ndarray]]:
    """Factory writing a tiny beats CSV (187 or 188 columns); returns (path, signals, labels)."""

    def _write(labeled: bool = True, n_rows: int = 6) -> tuple[Path, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(3)
        signals = rng.random((n_rows, SIGNAL_LENGTH))
        labels = np.arange(n_rows) % 2
        rows = np.column_stack([signals, labels]) if labeled else signals
        path = tmp_path / ("labeled.csv" if labeled else "unlabeled.csv")
        np.savetxt(path, rows, delimiter=",", fmt="%.8g")
        return path, signals, labels

    return _write
