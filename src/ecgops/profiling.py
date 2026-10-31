"""Profile one training epoch with torch.profiler (M4).

    uv run python -m ecgops.profiling                 # same Hydra overrides as train work

Each batch is split into labelled phases (dataloader / forward / backward / optimizer) with
``record_function``. The per-phase totals and the top PyTorch operators are logged and
written to reports/profiling/torch_epoch.txt. Nothing is logged to MLflow.
"""

from __future__ import annotations

import time

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

from ecgops.data import prepare_data, processed_data_exists
from ecgops.model import build_model
from ecgops.train import make_loaders, positive_class_weight
from ecgops.utils import CONFIG_DIR, abs_path, get_device, seed_everything, setup_logging

PHASES = ("dataloader", "forward", "backward", "optimizer")


@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    seed_everything(cfg.seed)
    device = get_device()
    if not processed_data_exists(cfg):
        prepare_data(cfg)

    train_loader, _ = make_loaders(cfg)
    model = build_model(OmegaConf.to_container(cfg.model, resolve=True)).to(device)  # type: ignore[arg-type]
    labels = train_loader.dataset.labels  # type: ignore[attr-defined]
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_class_weight(labels).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    model.train()
    start = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        batches = iter(train_loader)
        while True:
            with record_function("dataloader"):
                batch = next(batches, None)
            if batch is None:
                break
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            with record_function("forward"):
                loss = criterion(model(x), y)
            with record_function("backward"):
                loss.backward()
            with record_function("optimizer"):
                optimizer.step()
    elapsed = time.perf_counter() - start

    events = prof.key_averages()
    phase_ms = {e.key: e.cpu_time_total / 1000 for e in events if e.key in PHASES}
    total_ms = sum(phase_ms.values())
    lines = [f"One epoch ({len(train_loader)} batches) in {elapsed:.1f}s wall (profiler on)", ""]
    for phase in PHASES:
        ms = phase_ms.get(phase, 0.0)
        lines.append(f"{phase:<11} {ms / 1000:7.2f}s  {ms / total_ms:6.1%}")
    lines += ["", events.table(sort_by="self_cpu_time_total", row_limit=15)]
    report = "\n".join(lines)
    logger.info("torch.profiler summary:\n{}", report)

    out = abs_path(cfg.evaluation.reports_dir) / "profiling" / "torch_epoch.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    logger.info("Wrote {}", out)


if __name__ == "__main__":
    main()
