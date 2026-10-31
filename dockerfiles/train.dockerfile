# syntax=docker/dockerfile:1
# Training image (M3). ENTRYPOINT is the `train` script, so any arguments after the image
# name are Hydra overrides. Data and models are mounted at run time, never baked in.
# Build and run from the repo root (PowerShell):
#   docker build -f dockerfiles/train.dockerfile -t ecgops-train:latest `
#     --build-arg GIT_COMMIT=$(git rev-parse HEAD) --build-arg GIT_BRANCH=$(git branch --show-current) .
#   docker run --rm --network ecgops_default `
#     -v "${PWD}\data:/app/data" -v "${PWD}\models:/app/models" `
#     ecgops-train:latest training.epochs=2 tracking.mlflow.tracking_uri=http://mlflow:5000

FROM python:3.12-slim

# uv pinned to the same version as api.dockerfile (and the local uv).
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# 1) Third-party dependencies only. These files rarely change, so Docker reuses this
#    (large, slow) layer on every rebuild after a code change.
COPY uv.lock pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2) Our code and Hydra configs, then install the ecgops package itself (non-editable).
#    UV_COMPILE_BYTECODE=0 here: the dependencies were already compiled in step 1, and
#    recompiling them would copy ~13k .pyc files into this layer a second time.
COPY src/ src/
COPY configs/ configs/
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_COMPILE_BYTECODE=0 uv sync --frozen --no-dev --no-editable

# 3) Run as an unprivileged user; it owns only the folders the run writes to.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p data models outputs \
    && chown appuser:appuser data models outputs

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    ECGOPS_ROOT=/app \
    ECGOPS_CONFIG_DIR=/app/configs \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet

# Git metadata for the MLflow tags (the image has no .git folder). Declared last so a new
# commit only rebuilds this tiny layer.
ARG GIT_COMMIT=unknown
ARG GIT_BRANCH=unknown
ENV ECGOPS_GIT_COMMIT=$GIT_COMMIT \
    ECGOPS_GIT_BRANCH=$GIT_BRANCH

USER appuser
ENTRYPOINT ["train"]
