# ECGops

[![tests](https://github.com/habilarda-jpg/ECGops/actions/workflows/tests.yml/badge.svg?branch=feature/sprint-a)](https://github.com/habilarda-jpg/ECGops/actions/workflows/tests.yml?query=branch%3Afeature%2Fsprint-a)
[![lint](https://github.com/habilarda-jpg/ECGops/actions/workflows/lint.yml/badge.svg?branch=feature/sprint-a)](https://github.com/habilarda-jpg/ECGops/actions/workflows/lint.yml?query=branch%3Afeature%2Fsprint-a)
[![docker](https://github.com/habilarda-jpg/ECGops/actions/workflows/docker.yml/badge.svg?branch=feature/sprint-a)](https://github.com/habilarda-jpg/ECGops/actions/workflows/docker.yml?query=branch%3Afeature%2Fsprint-a)

MLOps course project: detecting abnormal heartbeats from ECG signals (PTBDB) with a
VGG-style 1D CNN, tracked in MLflow, configured with Hydra and packaged in Docker.

> The badges show `feature/sprint-a` for now. After merging to `main`, replace
> `branch=feature/sprint-a` with `branch=main` and `branch%3Afeature%2Fsprint-a` with
> `branch%3Amain` in the three badge lines.

## Quick start / Try it

```bash
git clone https://github.com/habilarda-jpg/ECGops.git
cd ECGops
uv sync
uv run predict input=samples/example_beats.csv plot=true
```

This classifies the 10 example heartbeats in `samples/` and prints a table like:

```
 beat true_label  abnormal_probability prediction  correct
    1   abnormal                0.9995   abnormal     True
    2     normal                0.0049     normal     True
  ...
Accuracy: 100.0% (10/10 correct)
```

It also writes `reports/predictions.csv` and, with `plot=true`, a grid of the waveforms in
`reports/figures/example_beats.png`. Other variants:

```bash
uv run predict input=samples/example_beats_unlabeled.csv   # no labels: no accuracy
uv run predict input=my_beats.csv evaluation.threshold=0.3 # your own CSV, other threshold
uv run predict model_source=mlflow model_version=8         # registry model (MLflow running)
uv run make-samples                                        # regenerate samples/ (needs data)
```

**You need a trained checkpoint first.** The weights are not in git. Either run
`uv run train` (needs the data: `uv run dvc pull`, then it writes `models/ecg_vgg1d.pt`),
or use version 8 from the MLflow registry with `model_source=mlflow` (start Docker and run
`docker compose up -d mlflow` first; see [Loading a registered model](#loading-a-registered-model)).

**Input format.** The model expects *pre-segmented heartbeats*: one beat per CSV row,
187 samples at 125 Hz (1.5 s, zero-padded at the end), scaled to [0, 1], optionally followed
by the label (0 = normal, 1 = abnormal), with no header. This is the PTBDB format. It does
not accept raw ECG recordings; anything else is rejected with a clear error. The sample
beats come from the test split only, so the model never saw them during training.

The prediction code lives in `src/ecgops/inference.py` (`load_model`, `read_beats`,
`predict`); the CLI, the API and the web page below all use those same functions, so they
give identical answers.

## Run the local website

```bash
uv sync --group serving      # FastAPI, uvicorn, python-multipart
uv run serve                 # http://127.0.0.1:8000 (uv run serve --port 8080 to change)
```

Open <http://127.0.0.1:8000>:

- **Try a sample**: pick one of the 10 example beats; the page draws it and shows the
  probability, the predicted label and the true label.
- **Upload a CSV**: drag `samples/demo_beats_500.csv` (500 labeled test beats) onto the
  drop zone. The bar shows the upload (`Uploading… 45%`), then the analysis
  (`Analysing… 120/300 beats (40%)`). Then you get a summary, accuracy, a table (click a row
  to draw that beat) and a **Download results CSV** button.
- **Threshold slider**: re-labels everything in the browser and shows how many beats change.

The API docs (Swagger) are at <http://127.0.0.1:8000/docs>. Endpoints: `GET /health`,
`GET /samples`, `POST /predict` (JSON beats), `POST /jobs` (CSV upload, max 5 MB and
10,000 beats), `GET /jobs/{id}` (progress and results), `GET /jobs/{id}/beats/{n}`,
`GET /jobs/{id}/results.csv`.

The model is loaded once at startup from `models/ecg_vgg1d.pt`. Set
`ECGOPS_MODEL_SOURCE=mlflow` (and optionally `ECGOPS_MODEL_VERSION=8`) to use the MLflow
registry instead; that needs `docker compose up -d mlflow`. Without a model the site
still starts: `/health` says why, and predictions return 503.

This runs **locally only** and is not deployed anywhere. Deployment comes in M7. It is
an educational project, not a medical diagnostic tool.

## Layout

```
├── .github/workflows/         # CI: tests.yml, lint.yml, docker.yml
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
├── samples/                   # example test-split beats (10 + a 500-beat demo file)
├── src/ecgops/
│   ├── data.py                # loading, preprocessing, validation
│   ├── model.py               # PyTorch model
│   ├── train.py               # training entrypoint (`uv run train`)
│   ├── evaluate.py            # evaluation entrypoint (`uv run evaluate`)
│   ├── inference.py           # reusable load_model / read_beats / predict
│   ├── predict.py             # `uv run predict`: CSV of beats -> table, CSV, figure
│   ├── api.py                 # FastAPI service + local web page (`uv run serve`)
│   ├── web/                   # index.html, app.js, style.css (no build step)
│   ├── monitoring.py          # drift detection + Prometheus metrics (M8)
│   ├── profiling.py           # torch.profiler snapshot of one epoch
│   ├── tracking.py            # MLflow / W&B experiment tracker
│   └── utils.py               # paths, seeding, git info, logging
├── tests/                     # pytest, one file per area (+ conftest.py fixtures)
├── docker-compose.yml         # local MLflow (PostgreSQL + MinIO), Prometheus, Grafana
├── pyproject.toml
└── uv.lock
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`
(the package supports 3.12 and 3.13).

```bash
uv sync                      # create .venv, install ecgops (editable) + dev tools
uv run pre-commit install    # run the pre-commit hooks on every commit
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
uv run make-samples                              # 10 test beats -> samples/*.csv
uv run predict input=samples/example_beats.csv   # try the model on a CSV of beats
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

## Tests and coverage

```bash
uv run pytest --cov --cov-report=term-missing   # coverage of the ecgops package only
```

Tests use synthetic data and `tmp_path` only: no `data/raw`, DVC or MLflow server is needed.
`tests/test_training.py` trains a tiny model for one epoch with MLflow disabled, then
evaluates the checkpoint.

## CI (GitHub Actions)

All three workflows run on pushes to `main`, `development`, `feature/**`, `bugfix/**`,
`release/**` and on pull requests to `main` and `development`.

| Workflow | Job name(s) | What it does |
|---|---|---|
| [`tests.yml`](.github/workflows/tests.yml) | `tests (3.12)`, `tests (3.13)` | `uv sync --frozen`, pytest with coverage (matrix over Python 3.12/3.13); coverage report uploaded as an artifact |
| [`lint.yml`](.github/workflows/lint.yml) | `lint` | `uv lock --check`, `ruff check .`, `ruff format --check .`, `mypy src` |
| [`docker.yml`](.github/workflows/docker.yml) | `docker build` | builds `train.dockerfile` and `evaluate.dockerfile` with the GitHub Actions layer cache (not pushed) and runs `--help` + an import smoke test, only when `src/`, `configs/`, `dockerfiles/`, `pyproject.toml` or `uv.lock` changed (otherwise the job passes without building) |

Pre-commit (`.pre-commit-config.yaml`) runs the same ruff checks locally plus
trailing-whitespace, end-of-file-fixer, check-yaml, check-toml, check-added-large-files
(1000 kB) and check-merge-conflict. mypy only runs in CI.

## Branching strategy

| Branch | Purpose |
|---|---|
| `main` | Always deployable. Protected: changes only arrive by pull request with green CI. |
| `development` | Integration branch; features are merged here first. |
| `feature/<name>` | New work, branched from `development` (e.g. `feature/sprint-a`). |
| `bugfix/<name>` | Fixes, branched from `development`. |
| `release/<version>` | Stabilising a release before it goes to `main`. |

Work flows `feature/*` → PR → `development` → PR → `main`. Branch protection on `main`
requires the checks `tests (3.12)`, `tests (3.13)`, `lint` and `docker build`.

## Sprint A checklist

| # | Item (module) | Status | Where |
|---|---|---|---|
| 1 | Project scaffolded from a cookiecutter template (M2) | Done, with a caveat | The layout follows the course's M2 package layout (`src/ecgops/` with `data.py`, `model.py`, `train.py`, plus `configs/`, `dockerfiles/`, `tests/`, `reports/`, `notebooks/`). It was restructured to that layout by hand (commit `b9ff30c`), not generated by running `cookiecutter`. |
| 2 | Git repository created, team has write access (M2) | Done | [github.com/habilarda-jpg/ECGops](https://github.com/habilarda-jpg/ECGops). Solo project: the owner has full write access. |
| 3 | Standard branching strategy (dev/feature/bugfix/release) (M2) | Done | `main`, `development`, `feature/sprint-a` on GitHub; workflows trigger on `feature/**`, `bugfix/**`, `release/**`; see [Branching strategy](#branching-strategy) |
| 4 | `pyproject.toml` kept in sync with actual imports (M2) | Done | `pyproject.toml` + `uv.lock`; CI runs `uv lock --check` and `uv sync --frozen`, and `tests/test_package.py` imports every module |
| 5 | Application containerized with a working Dockerfile (M3) | Done | `dockerfiles/train.dockerfile`, `dockerfiles/evaluate.dockerfile`; built and smoke-tested by `docker.yml` |
| 6 | Hardcoded hyperparameters moved into a Hydra config (M3) | Done | `configs/config.yaml`, `configs/model/vgg1d.yaml`, `configs/training/default.yaml`; `uv run train training.lr=5e-4`; `tests/test_config.py` |
| 7 | CI pipeline running on push (lint + at least one test) (M5) | Done | `.github/workflows/tests.yml` and `lint.yml` (plus `docker.yml`) |
| 8 | Pre-commit hooks installed (M5) | Done | `.pre-commit-config.yaml`; `uv run pre-commit install`, `uv run pre-commit run --all-files` |

## Known limitations

- **Patient leakage between splits.** The PTBDB CSVs contain beats only, with no patient
  IDs, so the split is per beat. Beats from the same patient can land in both train and
  test, which makes the test scores optimistic for patients the model has never seen.
- **Pre-segmented input only.** The model expects single heartbeats already cut to 187
  samples and scaled to [0, 1], as in the dataset. It does not take a raw ECG recording;
  beat detection and segmentation would have to happen before it.
