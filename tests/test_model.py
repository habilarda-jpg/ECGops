"""Tests for ecgops.model.VGG1D."""

import pytest
import torch

from ecgops.model import VGG1D, build_model, count_parameters


def test_forward_returns_one_logit_per_beat():
    model = VGG1D().eval()
    logits = model(torch.rand(8, 1, 187))
    assert logits.shape == (8,)


def test_per_block_conv_counts():
    model = VGG1D(channels=[16, 32, 64], convs_per_block=[1, 2, 3])
    n_convs = sum(isinstance(m, torch.nn.Conv1d) for m in model.modules())
    assert n_convs == 6
    assert model(torch.rand(2, 1, 187)).shape == (2,)


def test_wider_model_has_more_parameters():
    assert count_parameters(VGG1D(channels=[64, 128])) > count_parameters(VGG1D(channels=[8, 16]))


@pytest.mark.parametrize(
    "kwargs", [{"kernel_size": 4}, {"channels": [8, 16], "convs_per_block": [1, 1, 1]}]
)
def test_invalid_configs_raise(kwargs):
    with pytest.raises(ValueError):
        VGG1D(**kwargs)


def test_build_model_from_config_mapping():
    model_cfg = {
        "in_channels": 1,
        "channels": [8, 16],
        "convs_per_block": 1,
        "kernel_size": 3,
        "hidden_dim": 16,
        "dropout": 0.0,
    }
    model = build_model(model_cfg)
    assert isinstance(model, VGG1D)
    assert model.eval()(torch.rand(3, 1, 187)).shape == (3,)
