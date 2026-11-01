"""Tests for ecgops.train: a tiny end-to-end run on synthetic data with tracking disabled."""

import math

import pytest
import torch

from ecgops import evaluate, train
from ecgops.utils import abs_path


def test_positive_class_weight_is_negatives_over_positives():
    weight = train.positive_class_weight(torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert weight.shape == (1,)
    assert weight.item() == pytest.approx(1 / 3)


def test_model_pip_requirements_are_pinned_without_local_labels():
    requirements = train.model_pip_requirements()
    assert [r.split("==")[0] for r in requirements] == list(train.MODEL_REQUIREMENTS)
    assert all("==" in r and "+" not in r for r in requirements)


def test_one_epoch_run_writes_checkpoint_with_finite_loss(train_cfg):
    best_val_loss = train.main(train_cfg)

    assert math.isfinite(best_val_loss)
    checkpoint_path = abs_path(train_cfg.model_out)
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["best_epoch"] == 1
    assert checkpoint["mlflow_run_id"] is None  # MLflow was disabled
    assert math.isfinite(checkpoint["best_val_loss"])
    assert checkpoint["model_cfg"]["channels"] == [8, 16]


def test_checkpoint_can_be_evaluated(train_cfg, tmp_path):
    train.main(train_cfg)
    train_cfg.evaluation.reports_dir = (tmp_path / "reports").as_posix()

    metrics = evaluate.main(train_cfg)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    assert (tmp_path / "reports" / "metrics.json").exists()
    assert (tmp_path / "reports" / "figures" / "confusion_matrix.png").exists()
