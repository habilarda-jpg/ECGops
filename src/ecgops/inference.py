"""Reusable inference: load a trained model, read heartbeats, predict.

This module is the single place that turns beats into predictions. The ``predict`` CLI uses
it, and the FastAPI service and web page will call the same functions. It deliberately has
no Hydra, no printing and no file output: callers decide how to show the results.

Typical use::

    from ecgops.inference import load_model, predict, read_beats

    model = load_model("models/ecg_vgg1d.pt")
    beats = read_beats("samples/example_beats.csv")
    result = predict(model, beats.signals, threshold=0.5)
    result.probabilities, result.labels

The model expects pre-segmented heartbeats: 187 samples at 125 Hz, scaled to [0, 1], as in
the PTBDB dataset. It does not accept raw ECG recordings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd
import torch
from torch import nn

from ecgops.data import N_COLUMNS, SIGNAL_LENGTH, validate_labels, validate_signals
from ecgops.model import build_model

MLFLOW_HINT = (
    "Start Docker Desktop and run `docker compose up -d mlflow`, "
    "or use the local checkpoint (model_source=local)."
)


class ModelUnavailableError(RuntimeError):
    """The requested model could not be loaded (missing checkpoint, MLflow down, ...)."""


@dataclass(frozen=True)
class Beats:
    """Heartbeats read from a CSV: signals (N, 187) float32, labels (N,) int or None."""

    signals: np.ndarray
    labels: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.signals)


@dataclass(frozen=True)
class Predictions:
    """Model output for N beats: abnormal probabilities (N,) and 0/1 labels (N,)."""

    probabilities: np.ndarray
    labels: np.ndarray
    threshold: float


def load_model(path: str | Path, device: str | torch.device = "cpu") -> nn.Module:
    """Load a checkpoint written by ``uv run train`` and return the model in eval mode.

    The checkpoint holds the weights and the architecture config, so no Hydra config is
    needed. ``weights_only=True`` means loading never runs code from the file.
    """
    path = Path(path)
    if not path.exists():
        raise ModelUnavailableError(
            f"Checkpoint {path} not found. Train a model first (`uv run train`) or load "
            "version 8 from the MLflow registry (model_source=mlflow)."
        )
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not {"state_dict", "model_cfg"} <= checkpoint.keys():
        raise ModelUnavailableError(
            f"{path} is not an ecgops checkpoint (expected 'state_dict' and 'model_cfg')."
        )
    model = build_model(checkpoint["model_cfg"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def load_registry_model(
    name: str = "ecg-vgg1d",
    version: str | int = "latest",
    tracking_uri: str = "http://localhost:5000",
    timeout: float = 10.0,
) -> nn.Module:
    """Load a registered model from the MLflow model registry (e.g. ``ecg-vgg1d`` v8).

    MLflow is imported only here, so local inference never needs it. Raises
    ``ModelUnavailableError`` with a hint when the tracking server is not running.
    """
    from ecgops.tracking import mlflow_server_reachable  # imports mlflow

    if not mlflow_server_reachable(tracking_uri, timeout, log_failure=False):
        raise ModelUnavailableError(f"MLflow server {tracking_uri} is not reachable. {MLFLOW_HINT}")

    import mlflow
    import mlflow.pytorch

    # Download artifacts through the MLflow server instead of presigned MinIO URLs, which
    # point at the compose-internal hostname minio:9000 (see README).
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    mlflow.set_tracking_uri(tracking_uri)
    uri = f"models:/{name}/{version}"
    try:
        model = mlflow.pytorch.load_model(uri)
    except Exception as exc:  # MlflowException, network errors, ...
        raise ModelUnavailableError(f"Could not load {uri} from {tracking_uri}: {exc}") from exc
    return model  # type: ignore[no-any-return]


def read_beats(source: str | Path | IO[str] | IO[bytes]) -> Beats:
    """Read heartbeats from a CSV path or an open file (e.g. an uploaded file).

    Each row is one beat without a header: 187 values (unlabeled) or 187 values followed by
    the label 0/1 (the raw PTBDB format). Values must be numbers in [0, 1], the same rules
    ``ecgops.data.validate`` applies to the training data. Raises ``ValueError`` otherwise.
    """
    try:
        frame = pd.read_csv(source, header=None)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse the CSV file: {exc}") from exc

    n_columns = frame.shape[1]
    if n_columns not in (SIGNAL_LENGTH, N_COLUMNS):
        raise ValueError(
            f"Expected {SIGNAL_LENGTH} columns (one beat of {SIGNAL_LENGTH} samples per row) "
            f"or {N_COLUMNS} (beat + label), got {n_columns}. The model needs pre-segmented "
            f"beats of {SIGNAL_LENGTH} samples, not raw ECG recordings."
        )
    try:
        values = frame.to_numpy(dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"The CSV file contains non-numeric values ({exc}).") from exc

    signals = values[:, :SIGNAL_LENGTH]
    validate_signals(signals)
    labels = None
    if n_columns == N_COLUMNS:
        validate_labels(values[:, -1])
        labels = values[:, -1].astype(np.int64)
    return Beats(signals=signals.astype(np.float32), labels=labels)


@torch.no_grad()
def predict(
    model: nn.Module,
    beats: np.ndarray | torch.Tensor,
    threshold: float = 0.5,
    batch_size: int = 512,
) -> Predictions:
    """Predict a batch of beats of shape (N, 187) or (N, 1, 187).

    Returns the probability that each beat is abnormal (sigmoid of the logit) and the label
    ``probability >= threshold`` (1 = abnormal). Beats are checked with the same rules as
    ``read_beats``, so arrays coming straight from an API request are validated too.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}.")
    array = beats.detach().cpu().numpy() if isinstance(beats, torch.Tensor) else np.asarray(beats)
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    validate_signals(array.astype(np.float64))

    try:
        model.eval()
    except NotImplementedError:  # modules from torch.export do not support eval()
        pass
    param = next(model.parameters(), None)
    device = param.device if param is not None else torch.device("cpu")

    x = torch.from_numpy(array.astype(np.float32)).unsqueeze(1)  # (N, 1, 187)
    chunks = [
        torch.sigmoid(model(x[i : i + batch_size].to(device))).reshape(-1).cpu()
        for i in range(0, len(x), batch_size)
    ]
    probabilities = torch.cat(chunks).numpy().astype(np.float64)
    labels = (probabilities >= threshold).astype(np.int64)
    return Predictions(probabilities=probabilities, labels=labels, threshold=threshold)
