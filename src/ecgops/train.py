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
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from importlib.metadata import version
from unittest.mock import patch

import hydra
import mlflow
import mlflow.pytorch
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from ecgops.data import (
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

# Packages the logged MLflow model needs at load time (mlflow itself is added automatically).
MODEL_REQUIREMENTS = ("torch", "numpy", "cloudpickle")


def model_pip_requirements() -> list[str]:
    """Pin the model's runtime packages to the versions installed right now.

    Passing these to ``log_model`` stops MLflow from inferring requirements itself, which
    spawns ``uv export`` / pip subprocesses and took minutes per run (see
    reports/profiling.md). Local version labels such as ``+cpu`` are dropped because they only
    exist on the PyTorch index, not on PyPI.
    """
    return [f"{name}=={version(name).split('+')[0]}" for name in MODEL_REQUIREMENTS]


@contextmanager
def skip_mlflow_requirement_check() -> Iterator[None]:
    """Skip MLflow's "do the model requirements match this environment?" warning.

    MLflow builds that warning by scanning the metadata of every installed package
    (``importlib_metadata.packages_distributions``), which took ~80 s per run on this Windows
    machine and was the largest cost in the profile. It can never fire for us, because
    ``model_pip_requirements`` pins exactly the versions that are installed. The patch is
    limited to the ``log_model`` call and is skipped if a future MLflow renames the function.
    """
    import mlflow.pyfunc
    import mlflow.utils.environment

    name = "warn_dependency_requirement_mismatches"
    with ExitStack() as stack:
        for module in (mlflow.utils.environment, mlflow.pyfunc):
            if hasattr(module, name):
                stack.enter_context(patch.object(module, name, lambda *_args, **_kwargs: None))
            else:
                logger.debug("{}.{} not found; check not skipped", module.__name__, name)
        yield


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
            # Two real validation beats, shape (2, 1, 187). MLflow exports the model with
            # torch.export, which freezes any dimension of size 1: a 1-beat example would give
            # a registered model that only accepts batches of exactly one beat.
            example = val_loader.dataset.features[:2].numpy()  # type: ignore[attr-defined]
            log_start = time.perf_counter()
            with skip_mlflow_requirement_check():
                model_info = tracker.mlflow_call(
                    "log_model",
                    mlflow.pytorch.log_model,
                    model,
                    name="model",
                    input_example=example,
                    pip_requirements=model_pip_requirements(),
                    registered_model_name=cfg.tracking.mlflow.registered_model_name,
                )
            if model_info is not None:
                log_seconds = time.perf_counter() - log_start
                tracker.log_metrics({"log_model_seconds": log_seconds})
                logger.info(
                    "Logged model to MLflow in {:.1f}s and registered '{}' version {}",
                    log_seconds,
                    cfg.tracking.mlflow.registered_model_name,
                    model_info.registered_model_version,
                )

    return best_val_loss


if __name__ == "__main__":
    main()
