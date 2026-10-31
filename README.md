# ECGops

MLOps course project: detecting abnormal heartbeats from ECG signals.

> Status: scaffold only — no ML code yet.

## Layout

```
├── .github/workflows/ci.yml   # CI: ruff + pytest
├── configs/                   # Hydra configs
├── data/
│   ├── raw/                   # PTBDB CSVs (tracked with DVC, not git)
│   └── processed/
├── dockerfiles/
│   ├── api.dockerfile         # FastAPI inference image
│   └── mlflow.dockerfile      # MLflow tracking server image
├── infra/
│   ├── grafana/               # Grafana provisioning (Prometheus datasource)
│   └── prometheus/            # Prometheus scrape config
├── models/                    # trained weights (git-ignored)
├── notebooks/                 # exploration
├── reports/figures/           # generated plots
├── src/ecgops/
│   ├── data.py                # loading, preprocessing, validation
│   ├── model.py               # PyTorch model
│   ├── train.py               # training entrypoint (`uv run train`)
│   ├── evaluate.py            # evaluation entrypoint (`uv run evaluate`)
│   ├── api.py                 # FastAPI service (M7)
│   └── monitoring.py          # drift detection + Prometheus metrics (M8)
├── tests/                     # pytest
├── docker-compose.yml         # local MLflow (PostgreSQL + MinIO), Prometheus, Grafana
├── pyproject.toml
└── uv.lock
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`
(the package supports 3.12 and 3.13).

```bash
uv sync                      # create .venv, install ecgops (editable) + dev tools
uv run pre-commit install    # enable ruff hooks
uv run pytest                # run tests
uv run ruff check .          # lint
uv run train                 # training entrypoint
```

Optional dependency groups are not installed by default:

```bash
uv sync --group serving      # fastapi, uvicorn
uv sync --group tuning       # optuna
uv sync --group monitoring   # evidently, great-expectations, prometheus-client
uv sync --all-groups         # everything
```

On Linux (CI, Docker) torch comes from the CPU-only PyTorch index.

## CI (`.github/workflows/ci.yml`)

**test**: ruff (lint + format check) and pytest with coverage on Python 3.12, for every
push to `main` and every pull request. Image build and deploy jobs return with the API in M7.
