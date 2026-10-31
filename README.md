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
│   ├── evaluate.dockerfile    # slim image that runs `evaluate`
│   ├── mlflow.dockerfile      # MLflow tracking server image
│   └── train.dockerfile       # image that runs `train`
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
│   ├── monitoring.py          # drift detection + Prometheus metrics (M8)
│   ├── profiling.py           # torch.profiler snapshot of one epoch
│   ├── tracking.py            # MLflow / W&B experiment tracker
│   └── utils.py               # paths, seeding, git info, logging
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

## Training and evaluation

```bash
uv run prepare-data                              # raw CSVs -> data/processed/*.npz
uv run train                                     # Hydra config from configs/
uv run train training.epochs=3 training.lr=5e-4  # override any setting
uv run evaluate                                  # reports/metrics.json + confusion matrix
```

Re-run a past run with its exact config (every run saves it under `outputs/<date>/<time>/.hydra/`):

```bash
uv run train --config-path "$PWD/outputs/2026-10-31/18-12-34/.hydra" --config-name config
```

## Docker (PowerShell, from the repo root)

Data, models and reports are mounted, not baked into the images. The MLflow stack from
`docker-compose.yml` must be running for tracking; without it training logs a warning and
continues.

```powershell
docker build -f dockerfiles/train.dockerfile -t ecgops-train:latest `
  --build-arg GIT_COMMIT=$(git rev-parse HEAD) --build-arg GIT_BRANCH=$(git branch --show-current) .
docker run --rm --network ecgops_default -v "${PWD}\data:/app/data" -v "${PWD}\models:/app/models" `
  ecgops-train:latest training.epochs=2 tracking.mlflow.tracking_uri=http://mlflow:5000

docker build -f dockerfiles/evaluate.dockerfile -t ecgops-evaluate:latest .
docker run --rm -v "${PWD}\data:/app/data" -v "${PWD}\models:/app/models" `
  -v "${PWD}\reports:/app/reports" ecgops-evaluate:latest tracking.mlflow.enabled=false
```

## Loading a registered model

```bash
# PowerShell: $env:MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD = "false"
export MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false
uv run python -c "import mlflow, mlflow.pytorch, torch; mlflow.set_tracking_uri('http://localhost:5000'); m = mlflow.pytorch.load_model('models:/ecg-vgg1d/8'); print(torch.sigmoid(m(torch.rand(5, 1, 187))))"
```

`MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` is needed on the host. Otherwise MLflow downloads
artifacts through presigned MinIO URLs that use the compose-internal hostname `minio:9000`,
which only resolves inside the Docker network. Version 8 is the current model; versions 1–7
only accept a batch of one beat and are tagged `superseded`.

## Profiling

See [reports/profiling.md](reports/profiling.md).

```bash
uv run python -m cProfile -o reports/profiling/train.prof -m ecgops.train training.epochs=2
uv run snakeviz reports/profiling/train.prof      # interactive view in the browser
uv run python -m ecgops.profiling                 # torch.profiler: one epoch by phase
```

## CI (`.github/workflows/ci.yml`)

**test**: ruff (lint + format check) and pytest with coverage on Python 3.12, for every
push to `main` and every pull request. Image build and deploy jobs return with the API in M7.
