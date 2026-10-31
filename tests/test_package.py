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


def test_abs_path_resolves_against_project_root(tmp_path):
    from ecgops.utils import PROJECT_ROOT, abs_path

    assert abs_path("data/raw") == PROJECT_ROOT / "data" / "raw"
    assert abs_path(tmp_path) == tmp_path


def test_paths_and_git_info_follow_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("ECGOPS_ROOT", str(tmp_path))
    monkeypatch.setenv("ECGOPS_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("ECGOPS_GIT_COMMIT", "abc123")
    import ecgops.utils

    reloaded = importlib.reload(ecgops.utils)
    try:
        assert reloaded.PROJECT_ROOT == tmp_path
        assert reloaded.CONFIG_DIR == tmp_path / "cfg"
        assert reloaded.git_info()["commit"] == "abc123"
    finally:
        monkeypatch.undo()
        importlib.reload(ecgops.utils)
