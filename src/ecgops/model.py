"""VGG-style 1D convolutional network for heartbeat classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


class VGG1D(nn.Module):
    """A VGG16-inspired CNN for single-lead ECG beats of shape (batch, 1, 187).

    Like VGG16 it stacks blocks of small (kernel 3) convolutions with "same" padding, each
    block followed by max-pooling that halves the length while the channel count grows. The
    differences from the original 2D VGG16 (ImageNet, 224x224x3 images):

    * 1D instead of 2D: ``Conv1d``/``MaxPool1d`` slide along time, not across an image.
    * Smaller: 4 configurable blocks with 32-256 channels (VGG16: 5 blocks, 64-512 channels,
      ~138M parameters) because a beat has only 187 samples and the dataset ~15k rows.
    * ``BatchNorm1d`` after every convolution (VGG16 predates batch norm); it stabilises and
      speeds up training. Conv biases are dropped because batch norm adds its own shift.
    * ``AdaptiveAvgPool1d(1)`` (global average pooling) replaces flattening the feature map,
      so the classifier does not depend on the input length and has far fewer weights.
    * One small hidden layer instead of two 4096-unit layers, and a single output logit for
      binary normal/abnormal classification instead of 1000 class scores. Apply a sigmoid
      (or use ``BCEWithLogitsLoss``) to turn the logit into a probability.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256),
        convs_per_block: int | Sequence[int] = 2,
        kernel_size: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so padding keeps the signal length.")
        if isinstance(convs_per_block, int):
            convs_per_block = [convs_per_block] * len(channels)
        if len(convs_per_block) != len(channels):
            raise ValueError("convs_per_block needs one entry per block in channels.")

        layers: list[nn.Module] = []
        current = in_channels
        for out_channels, n_convs in zip(channels, convs_per_block, strict=True):
            for _ in range(n_convs):
                layers += [
                    nn.Conv1d(
                        current, out_channels, kernel_size, padding=kernel_size // 2, bias=False
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                ]
                current = out_channels
            layers.append(nn.MaxPool1d(kernel_size=2))  # halves the length: 187->93->46->23->11

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # (batch, C, L) -> (batch, C, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(current, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch (batch, 1, 187) to one logit per beat, shape (batch,)."""
        return self.classifier(self.pool(self.features(x))).squeeze(-1)


def build_model(model_cfg: Mapping[str, Any]) -> VGG1D:
    """Create a VGG1D from the ``model`` section of the Hydra config (or a checkpoint)."""
    convs = model_cfg["convs_per_block"]
    return VGG1D(
        in_channels=model_cfg["in_channels"],
        channels=list(model_cfg["channels"]),
        convs_per_block=convs if isinstance(convs, int) else list(convs),
        kernel_size=model_cfg["kernel_size"],
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
