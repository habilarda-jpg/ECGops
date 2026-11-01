"""Smoke tests for the package layout: every module imports and optional deps stay optional."""

import importlib
import sys

import pytest

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
