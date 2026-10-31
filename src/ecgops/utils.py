"""Small helpers shared by the data, training and evaluation scripts."""

from __future__ import annotations

import io
import os
import random
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from loguru import logger

# Where the project lives. With the editable install (uv sync) this file sits in
# <repo>/src/ecgops/, so the repo root is two levels up. Inside Docker the package is
# installed into site-packages instead, so the images set ECGOPS_ROOT=/app and
# ECGOPS_CONFIG_DIR=/app/configs explicitly.
PROJECT_ROOT = Path(os.environ.get("ECGOPS_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = Path(os.environ.get("ECGOPS_CONFIG_DIR", PROJECT_ROOT / "configs"))


def abs_path(path: str | Path) -> Path:
    """Resolve a (relative) config path such as data/raw against the project root.

    Absolute paths are returned unchanged, so you can still point anywhere from the CLI.
    """
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


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
    """Return the current git commit and branch.

    ECGOPS_GIT_COMMIT / ECGOPS_GIT_BRANCH win when set (Docker images have no .git folder and
    get them as build args); otherwise ask git, falling back to "unknown".
    """
    info = {}
    for key, args in {
        "commit": ["rev-parse", "HEAD"],
        "branch": ["branch", "--show-current"],
    }.items():
        if value := os.environ.get(f"ECGOPS_GIT_{key.upper()}"):
            info[key] = value
            continue
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
