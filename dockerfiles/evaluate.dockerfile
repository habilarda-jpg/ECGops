# syntax=docker/dockerfile:1
# Evaluation image (M3): a slimmer sibling of train.dockerfile that only runs `evaluate`.
# The trained checkpoint, the test split and the reports folder are mounted at run time.
# Build and run from the repo root (PowerShell):
#   docker build -f dockerfiles/evaluate.dockerfile -t ecgops-evaluate:latest .
#   docker run --rm -v "${PWD}\data:/app/data" -v "${PWD}\models:/app/models" `
#     -v "${PWD}\reports:/app/reports" ecgops-evaluate:latest tracking.mlflow.enabled=false

# ---- builder: install dependencies with uv, then strip what evaluation never needs ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
# No bytecode compilation: the .pyc caches add ~300 MB; Python compiles in memory instead.
ENV UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Packages that are installed as dependencies of torch/mlflow but are never imported when
# evaluating: torch.compile's symbolic maths (sympy, mpmath, networkx), mlflow's Arrow and
# Databricks integrations (pyarrow, databricks-sdk). Leaving them out saves ~270 MB.
ARG SKIP_PACKAGES="--no-install-package sympy --no-install-package mpmath \
    --no-install-package networkx --no-install-package pyarrow --no-install-package databricks-sdk"

# Dependency layer first (cached until uv.lock / pyproject.toml change), then our package.
COPY uv.lock pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project $SKIP_PACKAGES
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable $SKIP_PACKAGES

# Delete files that are only needed for development: test suites, C++ headers and CMake
# files for building torch extensions, and any stray bytecode caches.
RUN cd .venv/lib/python3.12/site-packages \
    && rm -rf torch/include torch/share/cmake \
    && find . -depth -type d \( -name tests -o -name __pycache__ \) -exec rm -rf {} +

# ---- runtime: a fresh slim image with only the trimmed venv and the configs ----
# The builder stage (uv binary, uv cache, deleted files) is left behind entirely.
FROM python:3.12-slim
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY configs/ configs/
RUN mkdir -p data models reports outputs && chown appuser:appuser data models reports outputs

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ECGOPS_ROOT=/app \
    ECGOPS_CONFIG_DIR=/app/configs \
    MPLCONFIGDIR=/tmp/matplotlib \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet

ARG GIT_COMMIT=unknown
ARG GIT_BRANCH=unknown
ENV ECGOPS_GIT_COMMIT=$GIT_COMMIT \
    ECGOPS_GIT_BRANCH=$GIT_BRANCH

USER appuser
ENTRYPOINT ["evaluate"]
