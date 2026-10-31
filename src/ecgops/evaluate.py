"""Evaluate the trained heartbeat classifier on the held-out test split.

``uv run evaluate`` loads the checkpoint from ``model_out``, scores the test set, writes
reports/metrics.json and reports/figures/confusion_matrix.png and, if MLflow is reachable,
logs both to a separate run tagged ``run_type=evaluation``.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from loguru import logger
from matplotlib.figure import Figure
from omegaconf import DictConfig
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from ecgops.data import ECGDataset, prepare_data, processed_data_exists, processed_paths
from ecgops.model import build_model
from ecgops.tracking import ExperimentTracker
from ecgops.utils import CONFIG_DIR, abs_path, get_device, setup_logging

CLASS_NAMES = ["normal", "abnormal"]


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[float | None, np.ndarray, np.ndarray]:
    """Run the model over a loader; return (mean loss or None, probabilities, true labels)."""
    model.eval()
    total_loss, probs, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * len(y)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.cpu().numpy())
    mean_loss = total_loss / len(loader.dataset) if criterion is not None else None  # type: ignore[arg-type]
    return mean_loss, np.concatenate(probs), np.concatenate(labels).astype(int)


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    """Accuracy, precision, recall and F1 at ``threshold``, plus threshold-free ROC-AUC.

    "Positive" is the abnormal class (label 1).
    """
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
    }


def save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    """Plot the confusion matrix with counts and save it as a PNG."""
    fig = Figure(figsize=(5, 4.5), layout="constrained")
    ax = fig.add_subplot()
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False
    )
    ax.set_title("ECG test set - confusion matrix")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> dict[str, float]:
    """Entry point for the ``evaluate`` console script."""
    setup_logging()
    device = get_device()

    checkpoint_path = abs_path(cfg.model_out)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path} not found - run `uv run train` first.")
    if not processed_data_exists(cfg):
        prepare_data(cfg)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint["model_cfg"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    logger.info("Loaded {} (best epoch {})", checkpoint_path, checkpoint.get("best_epoch"))

    test_set = ECGDataset.from_npz(processed_paths(cfg)["test"])
    loader = DataLoader(test_set, batch_size=cfg.training.batch_size, shuffle=False)
    _, probs, labels = predict(model, loader, device)

    threshold = cfg.evaluation.threshold
    metrics = binary_metrics(labels, probs, threshold)
    matrix = confusion_matrix(labels, (probs >= threshold).astype(int), labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())

    logger.info("Test set: {} beats, threshold {}", len(labels), threshold)
    for name, value in metrics.items():
        logger.info("  test_{:<10} {:.4f}", name, value)
    logger.info("Confusion matrix [[TN FP] [FN TP]]: [[{} {}] [{} {}]]", tn, fp, fn, tp)

    reports_dir = abs_path(cfg.evaluation.reports_dir)
    metrics_path = reports_dir / "metrics.json"
    figure_path = reports_dir / "figures" / "confusion_matrix.png"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(checkpoint_path),
        "training_run_id": checkpoint.get("mlflow_run_id"),
        "n_test": int(len(labels)),
        "threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": {"labels": CLASS_NAMES, "matrix": matrix.tolist()},
    }
    metrics_path.write_text(json.dumps(report, indent=2))
    save_confusion_matrix(matrix, figure_path)
    logger.info("Wrote {} and {}", metrics_path, figure_path)

    with ExperimentTracker(cfg, run_name="evaluate", run_type="evaluation") as tracker:
        if checkpoint.get("mlflow_run_id"):
            tracker.set_tags({"training_run_id": checkpoint["mlflow_run_id"]})
        tracker.log_metrics(
            {f"test_{k}": v for k, v in metrics.items()}
            | {"test_tn": tn, "test_fp": fp, "test_fn": fn, "test_tp": tp}
        )
        tracker.log_artifact(metrics_path)
        tracker.log_artifact(figure_path)
    return metrics


if __name__ == "__main__":
    main()
