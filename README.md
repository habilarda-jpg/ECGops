# ECGops

MLOps course project: detecting abnormal heartbeats from ECG signals.

> Status: scaffold only — no ML code yet.

## Layout

```
data/raw, data/processed   # datasets (tracked with DVC, not git)
src/data                   # data validation (Great Expectations)
src/models                 # training (PyTorch, Optuna, MLflow)
src/api                    # FastAPI inference service
src/monitoring             # drift detection (Evidently) + Prometheus metrics
tests                      # pytest
docker                     # Dockerfile
.github/workflows          # CI/CD
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```bash
uv sync                      # create .venv and install dependencies
uv run pre-commit install    # enable ruff hooks
uv run pytest                # run tests
uv run ruff check .          # lint
```

## CI/CD (`.github/workflows/ci.yml`)

1. **test** — ruff + pytest on Python 3.12 (every push / PR)
2. **build-push** — build `docker/Dockerfile`, push `ghcr.io/<owner>/ecgops:<git-sha>` (main only)
3. **deploy** — SSH to server, pull image, `docker compose up -d`, check `/health` (main only)

The deploy job is skipped unless these repository secrets are set:
`DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`. Optional repository variable `DEPLOY_PATH`
(directory containing `docker-compose.yml` on the server, default `~/ecgops`).
