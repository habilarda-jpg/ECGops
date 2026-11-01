# syntax=docker/dockerfile:1
# Multi-stage build for the ECGops FastAPI service (src/ecgops/api.py, implemented in M7).
# Build from the repo root: docker build -f dockerfiles/api.dockerfile .

# ---- builder: resolve and install dependencies into a venv ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
# Dependencies first (cached layer), then the package itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --group serving --no-install-project
COPY src ./src
RUN uv sync --frozen --no-default-groups --group serving --no-editable

# ---- runtime: slim image with only the venv ----
FROM python:3.12-slim AS runtime
RUN useradd --create-home appuser
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER appuser
EXPOSE 8000
CMD ["uvicorn", "ecgops.api:app", "--host", "0.0.0.0", "--port", "8000"]
