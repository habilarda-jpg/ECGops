"""Smoke tests for the package layout and the Hydra configs."""

import importlib
import sys

import pytest
from hydra import compose, initialize_config_dir

from ecgops.model import build_model
from ecgops.utils import CONFIG_DIR, flatten_dict

MODULES = ["data", "model", "train", "evaluate", "tracking", "utils", "api", "monitoring"]


def test_package_imports():
    import ecgops

    assert ecgops.__version__


@pytest.mark.parametrize("name", MODULES)
def test_modules_import(name):
    importlib.import_module(f"ecgops.{name}")


def test_wandb_not_imported_by_default():
    importlib.import_module("ecgops.train")
    assert "wandb" not in sys.modules


def test_config_composes_with_overrides():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["training.epochs=3", "training.lr=5e-4"])
    assert cfg.training.epochs == 3
    assert cfg.training.lr == 5e-4
    assert cfg.model.channels == [32, 64, 128, 256]
    assert build_model(cfg.model)


def test_flatten_dict():
    assert flatten_dict({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}
