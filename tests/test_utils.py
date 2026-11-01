"""Tests for ecgops.utils: paths, env-var overrides, seeding and helpers."""

import importlib

import torch

from ecgops.utils import PROJECT_ROOT, abs_path, flatten_dict, seed_everything


def test_flatten_dict():
    assert flatten_dict({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


def test_abs_path_resolves_against_project_root(tmp_path):
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


def test_seed_everything_makes_torch_repeatable():
    seed_everything(7)
    first = torch.rand(3)
    seed_everything(7)
    assert torch.equal(first, torch.rand(3))
