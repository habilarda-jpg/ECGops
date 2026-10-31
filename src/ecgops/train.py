"""Train the VGG1D heartbeat classifier and track the run in MLflow (and optionally W&B).

Examples::

    uv run train                                   # defaults from configs/
    uv run train training.epochs=3 training.lr=5e-4
    uv run train tracking.mlflow.enabled=false     # no tracking server needed
"""

from __future__ import annotations

import copy
import math
import time

import hydra
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from ecgops.data import (
    SIGNAL_LENGTH,
    ECGDataset,
    prepare_data,
    processed_data_exists,
    processed_paths,
)
from ecgops.evaluate import binary_metrics, predict
from ecgops.model import build_model, count_parameters
from ecgops.tracking import ExperimentTracker
from ecgops.utils import (
    CONFIG_DIR,
    abs_path,
    get_device,
    git_info,
    seed_everything,
    setup_logging,
)


def make_loaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    """Build shuffled training and ordered validation DataLoaders from the processed splits."""
    paths = processed_paths(cfg)
    train_set = ECGDataset.from_npz(paths["train"])
    val_set = ECGDataset.from_npz(paths["val"])
    generator = torch.Generator().manual_seed(cfg.seed)  # reproducible shuffling
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    logger.info("Train: {} beats | Val: {} beats", len(train_set), len(val_set))
    return train_loader, val_loader


def positive_class_weight(labels: torch.Tensor) -> torch.Tensor:
    """pos_weight = n_negative / n_positive for BCEWithLogitsLoss.

    Abnormal beats (1) outnumber normal ones (0) about 72/28, so this is < 1: it scales down
    the loss on the majority class so both classes contribute equally on average.
    """
    n_positive = labels.sum()
    n_negative = len(labels) - n_positive
    return (n_negative / n_positive).reshape(1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One pass over the training data; return the mean training loss."""
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)  # type: ignore[arg-type]


@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    """Entry point for the ``train`` console script; returns the best validation loss."""
    setup_logging()
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    device = get_device()

    if not processed_data_exists(cfg):
        logger.info("Processed data missing - running prepare_data first.")
        prepare_data(cfg)
    train_loader, val_loader = make_loaders(cfg)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = build_model(model_cfg).to(device)  # type: ignore[arg-type]
    logger.info("VGG1D with {:,} trainable parameters", count_parameters(model))

    pos_weight = positive_class_weight(train_loader.dataset.labels).to(device)  # type: ignore[attr-defined]
    logger.info("pos_weight = n_negative / n_positive = {:.4f}", pos_weight.item())
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    patience = cfg.training.early_stopping_patience
    threshold = cfg.evaluation.threshold
    best_val_loss, best_epoch, epochs_without_improvement = math.inf, 0, 0
    best_state = copy.deepcopy(model.state_dict())
    epoch = 0

    with ExperimentTracker(cfg, run_name="train", run_type="training") as tracker:
        start = time.perf_counter()
        for epoch in range(1, cfg.training.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_probs, val_labels = predict(model, val_loader, device, criterion)
            assert val_loss is not None
            scores = binary_metrics(val_labels, val_probs, threshold)
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": scores["accuracy"],
                "val_f1": scores["f1"],
                "val_auc": scores["roc_auc"],
            }
            tracker.log_metrics(metrics, step=epoch)
            logger.info(
                "Epoch {:>2}/{} | train_loss {:.4f} | val_loss {:.4f} | val_acc {:.4f} "
                "| val_f1 {:.4f} | val_auc {:.4f}",
                epoch,
                cfg.training.epochs,
                *metrics.values(),
            )

            if val_loss < best_val_loss:
                best_val_loss, best_epoch, epochs_without_improvement = val_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info("Early stopping: no val_loss improvement for {} epochs.", patience)
                    break

        train_seconds = time.perf_counter() - start
        logger.info(
            "Trained {} epochs in {:.1f}s; best val_loss {:.4f} at epoch {}",
            epoch,
            train_seconds,
            best_val_loss,
            best_epoch,
        )
        tracker.log_metrics(
            {
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "epochs_trained": epoch,
                "train_time_seconds": train_seconds,
            }
        )

        # Save the best weights (not the last epoch's) together with the architecture config.
        model.load_state_dict(best_state)
        model_out = abs_path(cfg.model_out)
        model_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "model_cfg": model_cfg,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "mlflow_run_id": tracker.run_id,
                "git_commit": git_info()["commit"],
            },
            model_out,
        )
        logger.info("Saved best checkpoint to {}", model_out)

        if tracker.use_mlflow:
            model.cpu().eval()
            example = np.zeros((1, 1, SIGNAL_LENGTH), dtype=np.float32)  # documents the input shape
            model_info = mlflow.pytorch.log_model(
                model,
                name="model",
                input_example=example,
                registered_model_name=cfg.tracking.mlflow.registered_model_name,
            )
            logger.info(
                "Logged model to MLflow and registered '{}' version {}",
                cfg.tracking.mlflow.registered_model_name,
                model_info.registered_model_version,
            )

    return best_val_loss


if __name__ == "__main__":
    main()
