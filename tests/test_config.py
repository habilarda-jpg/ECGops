"""Tests for the Hydra configs in configs/: composition, defaults and CLI-style overrides."""

import pytest
from hydra.errors import ConfigCompositionException

from ecgops.model import build_model


def test_config_composes_with_overrides(compose_cfg):
    cfg = compose_cfg(["training.epochs=3", "training.lr=5e-4"])
    assert cfg.training.epochs == 3
    assert cfg.training.lr == 5e-4
    assert cfg.model.channels == [32, 64, 128, 256]
    assert build_model(cfg.model)


def test_default_config_has_every_section(compose_cfg):
    cfg = compose_cfg()
    assert {"seed", "data", "model", "training", "evaluation", "tracking", "model_out"} <= set(cfg)
    assert cfg.data.val_size + cfg.data.test_size < 1
    assert cfg.model.kernel_size % 2 == 1
    assert cfg.tracking.wandb.enabled is False


def test_nested_list_and_bool_overrides(compose_cfg):
    cfg = compose_cfg(["model.channels=[8,16]", "tracking.mlflow.enabled=false", "+extra=1"])
    assert cfg.model.channels == [8, 16]
    assert cfg.tracking.mlflow.enabled is False
    assert cfg.extra == 1


def test_unknown_key_override_is_rejected(compose_cfg):
    # Without the "+" prefix Hydra refuses keys that are not in the config (catches typos).
    with pytest.raises(ConfigCompositionException):
        compose_cfg(["training.epocs=3"])
