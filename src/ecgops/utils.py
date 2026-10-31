"""Small helpers shared by the data, training and evaluation scripts."""

from __future__ import annotations

import io
import random
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def abs_path(path: str | Path) -> Path:
    """Resolve a config path against the directory the command was started from."""
    return Path(to_absolute_path(str(path)))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Use the GPU when one is available, otherwise the CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}", device)
    return device


def git_info() -> dict[str, str]:
    """Return the current git commit and branch ("unknown" outside a git checkout)."""
    info = {}
    for key, args in {
        "commit": ["rev-parse", "HEAD"],
        "branch": ["branch", "--show-current"],
    }.items():
        try:
            result = subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
            )
            info[key] = result.stdout.strip() or "unknown"
        except (OSError, subprocess.CalledProcessError):
            info[key] = "unknown"
    return info


def flatten_dict(nested: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten {"training": {"lr": 0.001}} into {"training.lr": 0.001}."""
    flat: dict[str, Any] = {}
    for key, value in nested.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_dict(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


def setup_logging() -> None:
    """Prepare console and file logging at the start of each script.

    * Switch stdout/stderr to UTF-8: on Windows a redirected stream defaults to cp1252, and
      MLflow prints emoji ("View run ... at ...") that would otherwise crash the run.
    * Also write loguru output into the Hydra run directory (outputs/<date>/<time>/run.log).
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if HydraConfig.initialized():
        log_file = Path(HydraConfig.get().runtime.output_dir) / "run.log"
        logger.add(log_file, level="DEBUG")
        logger.info("Logging to {}", log_file)
