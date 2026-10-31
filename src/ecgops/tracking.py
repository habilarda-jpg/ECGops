"""Experiment tracking: one small wrapper that forwards to MLflow and, optionally, W&B.

Both backends are optional. MLflow is used when ``tracking.mlflow.enabled`` is true *and* the
server answers; W&B only when ``tracking.wandb.enabled`` is true (``wandb`` is imported only
then, so it need not be installed otherwise: ``uv sync --group wandb``).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import mlflow
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from ecgops.utils import flatten_dict, git_info


def mlflow_server_reachable(tracking_uri: str, timeout: float) -> bool:
    """Ping the MLflow server's /health endpoint; local file/sqlite URIs need no server."""
    if not tracking_uri.startswith(("http://", "https://")):
        return True
    try:
        with urllib.request.urlopen(f"{tracking_uri.rstrip('/')}/health", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError) as exc:
        logger.warning(
            "MLflow server {} unreachable ({}); continuing without MLflow.", tracking_uri, exc
        )
        return False


class ExperimentTracker:
    """Start a run, log params/metrics/artifacts to every enabled backend, then close it.

    Use as a context manager::

        with ExperimentTracker(cfg, run_name="train", run_type="training") as tracker:
            tracker.log_metrics({"val_loss": 0.3}, step=1)
    """

    def __init__(self, cfg: DictConfig, run_name: str, run_type: str) -> None:
        self.cfg = cfg
        self.run_name = run_name
        self.run_type = run_type
        mlflow_cfg = cfg.tracking.mlflow
        self.use_mlflow = bool(mlflow_cfg.enabled) and mlflow_server_reachable(
            mlflow_cfg.tracking_uri, mlflow_cfg.connect_timeout
        )
        self.use_wandb = bool(cfg.tracking.wandb.enabled)
        self.run_id: str | None = None
        self._wandb_run: Any = None

    def __enter__(self) -> ExperimentTracker:
        git = git_info()
        params = flatten_dict(OmegaConf.to_container(self.cfg, resolve=True))  # type: ignore[arg-type]
        tags = {
            "run_type": self.run_type,
            "git_commit": git["commit"],
            "git_branch": git["branch"],
            "mlflow.source.git.commit": git["commit"],
        }

        if self.use_mlflow:
            mlflow_cfg = self.cfg.tracking.mlflow
            mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
            mlflow.set_experiment(mlflow_cfg.experiment_name)
            run = mlflow.start_run(run_name=self.run_name, tags=tags)
            self.run_id = run.info.run_id
            mlflow.log_params(params)
            mlflow.log_text(OmegaConf.to_yaml(self.cfg, resolve=True), "config.yaml")
            logger.info(
                "MLflow run {} in experiment '{}' ({})",
                self.run_id,
                mlflow_cfg.experiment_name,
                mlflow_cfg.tracking_uri,
            )

        if self.use_wandb:
            import wandb  # lazy: only needed (and installed) when W&B is enabled

            self._wandb_run = wandb.init(
                project=self.cfg.tracking.wandb.project,
                name=self.run_name,
                job_type=self.run_type,
                config={**params, "git_commit": git["commit"], "git_branch": git["branch"]},
            )
            logger.info("W&B run {}", self._wandb_run.url)
        return self

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def set_tags(self, tags: dict[str, str]) -> None:
        if self.use_mlflow:
            mlflow.set_tags(tags)
        if self._wandb_run is not None:
            self._wandb_run.config.update(tags)

    def log_artifact(self, path: str | Path) -> None:
        if self.use_mlflow:
            mlflow.log_artifact(str(path))
        if self._wandb_run is not None:
            self._wandb_run.save(str(path), policy="now")

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        failed = exc_type is not None
        if self.use_mlflow:
            mlflow.end_run(status="FAILED" if failed else "FINISHED")
        if self._wandb_run is not None:
            self._wandb_run.finish(exit_code=1 if failed else 0)
