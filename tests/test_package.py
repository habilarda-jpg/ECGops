"""Smoke tests for the ecgops package layout and console-script entry points."""

import importlib

import pytest

MODULES = ["data", "model", "train", "evaluate", "api", "monitoring"]


def test_package_imports():
    import ecgops

    assert ecgops.__version__


@pytest.mark.parametrize("name", MODULES)
def test_modules_import(name):
    importlib.import_module(f"ecgops.{name}")


@pytest.mark.parametrize("name", ["train", "evaluate"])
def test_entrypoints_run(name):
    module = importlib.import_module(f"ecgops.{name}")
    module.main()
