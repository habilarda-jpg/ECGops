"""Classify heartbeats from a CSV file with the trained model.

Examples::

    uv run predict input=samples/example_beats.csv plot=true
    uv run predict input=samples/example_beats_unlabeled.csv
    uv run predict model_source=mlflow model_version=8     # needs the MLflow server

Settings live in the ``predict`` section of configs/config.yaml. The short keys ``input``,
``model_source``, ``model_version``, ``output``, ``plot`` and ``figure`` are accepted on the
command line as shortcuts for ``predict.<key>``. The threshold comes from
``evaluation.threshold``. The actual inference happens in ``ecgops.inference``, which the
API and web page reuse.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.figure import Figure
from omegaconf import DictConfig
from torch import nn

from ecgops.data import CLASS_NAMES, SAMPLING_RATE_HZ, SIGNAL_LENGTH
from ecgops.inference import (
    Beats,
    ModelUnavailableError,
    Predictions,
    load_model,
    load_registry_model,
    predict,
    read_beats,
)
from ecgops.utils import CONFIG_DIR, abs_path, setup_logging

SHORTCUTS = ("input", "model_source", "model_version", "output", "plot", "figure")
MAX_PLOTTED_BEATS = 20


def expand_shortcuts(args: list[str]) -> list[str]:
    """Turn ``input=x`` into ``predict.input=x`` for the keys in SHORTCUTS."""
    expanded = []
    for arg in args:
        key = arg.split("=", 1)[0]
        expanded.append(f"predict.{arg}" if "=" in arg and key in SHORTCUTS else arg)
    return expanded


def load_selected_model(cfg: DictConfig) -> tuple[nn.Module, str]:
    """Load the model chosen by ``predict.model_source``; return it with a description."""
    source = cfg.predict.model_source
    if source == "local":
        path = abs_path(cfg.model_out)
        return load_model(path), str(path)
    if source == "mlflow":
        mlflow_cfg = cfg.tracking.mlflow
        name, version = mlflow_cfg.registered_model_name, cfg.predict.model_version
        model = load_registry_model(
            name, version, mlflow_cfg.tracking_uri, mlflow_cfg.connect_timeout
        )
        return model, f"models:/{name}/{version} ({mlflow_cfg.tracking_uri})"
    raise ValueError(f"predict.model_source must be 'local' or 'mlflow', got '{source}'.")


def results_table(beats: Beats, result: Predictions) -> pd.DataFrame:
    """One row per beat: beat number, true label (if known), probability, prediction."""
    table = pd.DataFrame({"beat": np.arange(1, len(beats) + 1)})
    if beats.labels is not None:
        table["true_label"] = [CLASS_NAMES[label] for label in beats.labels]
    table["abnormal_probability"] = result.probabilities.round(4)
    table["prediction"] = [CLASS_NAMES[label] for label in result.labels]
    if beats.labels is not None:
        table["correct"] = beats.labels == result.labels
    return table


def plot_beats(beats: Beats, result: Predictions, path: Path) -> None:
    """Save a grid of the waveforms with the predicted label and probability in each title.

    Red lines are beats predicted abnormal, blue ones normal. When true labels are known,
    the title also shows them and marks wrong predictions.
    """
    n = min(len(beats), MAX_PLOTTED_BEATS)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)
    seconds = np.arange(SIGNAL_LENGTH) / SAMPLING_RATE_HZ
    fig = Figure(figsize=(3.2 * ncols, 2.5 * nrows), layout="constrained")
    for i in range(n):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        predicted = int(result.labels[i])
        ax.plot(seconds, beats.signals[i], color="tab:red" if predicted else "tab:blue", lw=1)
        title = f"#{i + 1} {CLASS_NAMES[predicted]} (p={result.probabilities[i]:.2f})"
        if beats.labels is not None:
            true = int(beats.labels[i])
            title += f"\ntrue: {CLASS_NAMES[true]}" + ("" if true == predicted else "  WRONG")
        ax.set_title(
            title,
            fontsize=9,
            color="black" if beats.labels is None or beats.labels[i] == predicted else "tab:red",
        )
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)
        if i % ncols == 0:
            ax.set_ylabel("amplitude", fontsize=8)
        if i >= n - ncols:
            ax.set_xlabel("time (s)", fontsize=8)
    shown = f"first {n} of {len(beats)}" if n < len(beats) else f"{n}"
    fig.suptitle(
        f"ECG beats: {shown} beats, p = P(abnormal), threshold {result.threshold}", fontsize=11
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)


def run(cfg: DictConfig) -> pd.DataFrame:
    """Read the CSV, predict, print the table, save predictions (and the figure)."""
    input_path = abs_path(cfg.predict.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} not found.")
    beats = read_beats(input_path)
    model, model_name = load_selected_model(cfg)
    result = predict(model, beats.signals, threshold=cfg.evaluation.threshold)
    logger.info("Predicted {} beats from {} with {}", len(beats), input_path, model_name)

    table = results_table(beats, result)
    print(table.to_string(index=False))
    n_abnormal = int(result.labels.sum())
    print(
        f"\n{n_abnormal} of {len(beats)} beats predicted abnormal (threshold {result.threshold})."
    )
    if beats.labels is not None:
        accuracy = float((beats.labels == result.labels).mean())
        print(f"Accuracy: {accuracy:.1%} ({int(table['correct'].sum())}/{len(beats)} correct)")

    output = abs_path(cfg.predict.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    saved = table.drop(columns=["correct"], errors="ignore")
    saved.insert(saved.columns.get_loc("prediction"), "predicted_label", result.labels)
    saved.to_csv(output, index=False)
    print(f"Saved predictions to {output}")

    if cfg.predict.plot:
        figure = abs_path(cfg.predict.figure)
        plot_beats(beats, result, figure)
        print(f"Saved figure to {figure}")
    return table


@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> pd.DataFrame:
    """Hydra entry point; expected problems end with a one-line error instead of a traceback."""
    setup_logging()
    try:
        return run(cfg)
    except (ValueError, FileNotFoundError, ModelUnavailableError) as exc:
        logger.error("{}", exc)
        raise SystemExit(1) from exc


def cli() -> None:
    """Entry point for the ``predict`` console script (expands the short keys first)."""
    sys.argv[1:] = expand_shortcuts(sys.argv[1:])
    main()


if __name__ == "__main__":
    cli()
