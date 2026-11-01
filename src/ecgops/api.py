"""FastAPI service and local web page for heartbeat classification.

Run it with ``uv sync`` (serving is a default group) and ``uv run serve``, then open http://127.0.0.1:8000
(the web page) or http://127.0.0.1:8000/docs (Swagger). It runs locally only; deployment is
M7.

All predictions go through ``ecgops.inference`` (the same functions as ``uv run predict``),
so the CLI, the API and the page give identical answers.

Endpoints
---------
GET  /                       the web page (src/ecgops/web/)
GET  /health                 model status, source/version and default threshold
GET  /samples                the beats in samples/example_beats.csv (values + true label)
POST /predict                JSON {"beats": [[187 floats], ...], "threshold": optional}
POST /jobs                   upload a CSV (multipart) -> {job_id, total}; runs in background
GET  /jobs/{id}              status, processed/total/percent; results + summary when done
GET  /jobs/{id}/beats/{n}    the n-th uploaded beat (1-based), for drawing its waveform
GET  /jobs/{id}/results.csv  results as CSV (optionally re-labelled with ?threshold=)
DELETE /jobs/{id}            forget a job (the page's "Clear results"); frees its memory

Configuration (environment variables)
-------------------------------------
ECGOPS_MODEL_SOURCE   local (default) | mlflow
ECGOPS_MODEL_VERSION  registry version for mlflow (default latest)
ECGOPS_MODEL_PATH     local checkpoint (default model_out from configs/config.yaml)
ECGOPS_THRESHOLD      default threshold (default evaluation.threshold)
ECGOPS_SAMPLES_PATH   sample beats for /samples (default samples/example_beats.csv)
MLFLOW_TRACKING_URI   MLflow server (default tracking.mlflow.tracking_uri)
"""

from __future__ import annotations

import argparse
import io
import os
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from torch import nn

from ecgops.data import CLASS_NAMES, SAMPLING_RATE_HZ, SIGNAL_LENGTH
from ecgops.inference import (
    Predictions,
    load_model,
    load_registry_model,
    predict,
    predictions_table,
    read_beats,
)
from ecgops.utils import CONFIG_DIR, abs_path

WEB_DIR = Path(__file__).parent / "web"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_BEATS = 10_000
JOB_BATCH_SIZE = 32  # beats per batch; progress is updated after every batch
MAX_JOBS = 10  # jobs kept in memory at once
JOB_TTL_SECONDS = 60 * 60  # finished jobs are forgotten after an hour


# ----------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class Settings:
    model_source: str
    model_version: str
    model_path: Path
    threshold: float
    tracking_uri: str
    registered_model_name: str
    connect_timeout: float
    samples_path: Path

    @classmethod
    def from_env(cls) -> Settings:
        """Defaults from configs/config.yaml (when present), overridden by env variables."""
        config_file = CONFIG_DIR / "config.yaml"
        cfg = OmegaConf.load(config_file) if config_file.exists() else OmegaConf.create()
        env = os.environ.get
        return cls(
            model_source=env("ECGOPS_MODEL_SOURCE", "local"),
            model_version=env("ECGOPS_MODEL_VERSION", "latest"),
            model_path=abs_path(
                env("ECGOPS_MODEL_PATH", OmegaConf.select(cfg, "model_out", default=None))
                or "models/ecg_vgg1d.pt"
            ),
            threshold=float(
                env("ECGOPS_THRESHOLD", OmegaConf.select(cfg, "evaluation.threshold", default=0.5))
            ),
            tracking_uri=env(
                "MLFLOW_TRACKING_URI",
                OmegaConf.select(cfg, "tracking.mlflow.tracking_uri", default=None)
                or "http://localhost:5000",
            ),
            registered_model_name=OmegaConf.select(
                cfg, "tracking.mlflow.registered_model_name", default="ecg-vgg1d"
            ),
            connect_timeout=float(
                OmegaConf.select(cfg, "tracking.mlflow.connect_timeout", default=10)
            ),
            samples_path=abs_path(env("ECGOPS_SAMPLES_PATH", "samples/example_beats.csv")),
        )

    @property
    def model_description(self) -> str:
        if self.model_source == "mlflow":
            return f"models:/{self.registered_model_name}/{self.model_version}"
        return str(self.model_path)


def load_configured_model(settings: Settings) -> nn.Module:
    """Load the model selected by the settings; raises if that is impossible."""
    if settings.model_source == "local":
        return load_model(settings.model_path)
    if settings.model_source == "mlflow":
        # Download artifacts through the MLflow server, not via presigned MinIO URLs that
        # use the compose-internal hostname minio:9000 (see README).
        os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] = "false"
        return load_registry_model(
            settings.registered_model_name,
            settings.model_version,
            settings.tracking_uri,
            settings.connect_timeout,
        )
    raise ValueError(
        f"ECGOPS_MODEL_SOURCE must be 'local' or 'mlflow', got '{settings.model_source}'."
    )


# ----------------------------------------------------------------------------- jobs

JobStatus = Literal["queued", "running", "done", "error"]


@dataclass
class Job:
    """One uploaded CSV being classified in the background, batch by batch."""

    id: str
    filename: str
    signals: np.ndarray
    labels: np.ndarray | None
    threshold: float
    status: JobStatus = "queued"
    processed: int = 0
    error: str | None = None
    probabilities: np.ndarray | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def total(self) -> int:
        return len(self.signals)

    @property
    def percent(self) -> int:
        return int(100 * self.processed / self.total) if self.total else 100

    def predictions(self, threshold: float | None = None) -> Predictions:
        assert self.probabilities is not None, "job has no results yet"
        result = Predictions(self.probabilities, np.zeros(0, dtype=np.int64), self.threshold)
        return result.at_threshold(self.threshold if threshold is None else threshold)


class JobStore:
    """In-memory jobs with a size cap and expiry (enough for one local user)."""

    def __init__(self, max_jobs: int = MAX_JOBS, ttl_seconds: float = JOB_TTL_SECONDS) -> None:
        self.max_jobs = max_jobs
        self.ttl_seconds = ttl_seconds
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._evict()
            if len(self._jobs) >= self.max_jobs:
                raise HTTPException(
                    429, f"Too many jobs in progress (max {self.max_jobs}); try again shortly."
                )
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"Job '{job_id}' not found (unknown or expired).")
        return job

    def remove(self, job_id: str) -> None:
        """Forget a job. A running job finishes in its thread, but nobody can read it."""
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            raise HTTPException(404, f"Job '{job_id}' not found (unknown or expired).")

    def _evict(self) -> None:
        """Drop expired jobs, then the oldest finished ones while the store is full."""
        now = time.time()
        for job_id, job in list(self._jobs.items()):
            if job.finished_at is not None and now - job.finished_at > self.ttl_seconds:
                del self._jobs[job_id]
        for job_id, job in list(self._jobs.items()):
            if len(self._jobs) < self.max_jobs:
                break
            if job.finished_at is not None:
                del self._jobs[job_id]


def run_job(job: Job, model: nn.Module, batch_size: int = JOB_BATCH_SIZE) -> None:
    """Predict the job's beats batch by batch, updating ``processed`` after each batch.

    Runs in a worker thread (FastAPI runs sync background tasks in its thread pool), so
    the server keeps answering GET /jobs/{id} polls while it works.
    """
    job.status = "running"
    probabilities = np.empty(job.total, dtype=np.float64)
    try:
        for start in range(0, job.total, batch_size):
            chunk = job.signals[start : start + batch_size]
            probabilities[start : start + len(chunk)] = predict(model, chunk).probabilities
            job.processed = start + len(chunk)
        job.probabilities = probabilities
        job.status = "done"
    except Exception as exc:  # report any failure to the polling client
        logger.exception("Job {} failed", job.id)
        job.error = str(exc)
        job.status = "error"
    finally:
        job.finished_at = time.time()


# ----------------------------------------------------------------------------- app


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load settings and the model once at startup; keep running even if loading fails."""
    settings = Settings.from_env()
    app.state.settings = settings
    app.state.jobs = JobStore()
    app.state.model = None
    app.state.model_error = None
    try:
        app.state.model = load_configured_model(settings)
        logger.info("Loaded model {}", settings.model_description)
    except Exception as exc:  # the app still starts; /health and 503s explain the problem
        app.state.model_error = str(exc)
        logger.error("Model not loaded: {}", exc)
    yield


app = FastAPI(
    title="ECGops",
    description=(
        "Classify pre-segmented ECG heartbeats (187 samples at 125 Hz, scaled to [0, 1]) as "
        "normal or abnormal. Educational project, not a medical diagnostic tool."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def get_model(request: Request) -> nn.Module:
    model: nn.Module | None = request.app.state.model
    if model is None:
        raise HTTPException(
            503, f"Model not loaded: {request.app.state.model_error} See GET /health."
        )
    return model


def label_name(label: int) -> str:
    return CLASS_NAMES[int(label)]


class PredictRequest(BaseModel):
    beats: list[list[float]] = Field(
        ..., description=f"One or more beats of {SIGNAL_LENGTH} values in [0, 1]."
    )
    threshold: float | None = Field(None, ge=0.0, le=1.0, description="Default: /health.")


class BeatPrediction(BaseModel):
    beat: int
    probability: float
    label: int
    label_name: str


class PredictResponse(BaseModel):
    model: str
    threshold: float
    n_abnormal: int
    predictions: list[BeatPrediction]


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Liveness and model status. Always 200; ``model_loaded`` says whether predictions work."""
    settings: Settings = request.app.state.settings
    loaded = request.app.state.model is not None
    return {
        "status": "ok" if loaded else "model_unavailable",
        "model_loaded": loaded,
        "model_source": settings.model_source,
        "model_version": settings.model_version if settings.model_source == "mlflow" else None,
        "model": settings.model_description,
        "threshold": settings.threshold,
        "error": request.app.state.model_error,
        "signal_length": SIGNAL_LENGTH,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "limits": {"max_upload_bytes": MAX_UPLOAD_BYTES, "max_beats": MAX_BEATS},
    }


@app.get("/samples")
def samples(request: Request) -> dict[str, Any]:
    """The example beats from samples/example_beats.csv, for the page's "Try a sample"."""
    path: Path = request.app.state.settings.samples_path
    if not path.exists():
        raise HTTPException(404, f"{path} not found - run `uv run make-samples`.")
    beats = read_beats(path)
    labels = beats.labels if beats.labels is not None else [None] * len(beats)
    return {
        "source": path.name,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "beats": [
            {
                "id": i + 1,
                "values": [round(float(v), 6) for v in signal],
                "label": None if label is None else int(label),
                "label_name": None if label is None else label_name(label),
            }
            for i, (signal, label) in enumerate(zip(beats.signals, labels, strict=True))
        ],
    }


@app.post("/predict", response_model=PredictResponse)
def predict_beats(body: PredictRequest, request: Request) -> PredictResponse:
    """Classify beats sent as JSON. Invalid beats -> 422 with the same message as the CLI."""
    model = get_model(request)
    threshold = request.app.state.settings.threshold if body.threshold is None else body.threshold
    if len(body.beats) > MAX_BEATS:
        raise HTTPException(422, f"Too many beats: {len(body.beats)} (limit {MAX_BEATS}).")
    lengths = {len(beat) for beat in body.beats}
    if lengths - {SIGNAL_LENGTH} or not body.beats:
        raise HTTPException(
            422,
            f"Every beat must have exactly {SIGNAL_LENGTH} values; got lengths "
            f"{sorted(lengths) or '[] (no beats)'}. The model needs pre-segmented beats of "
            f"{SIGNAL_LENGTH} samples, not raw ECG recordings.",
        )
    try:
        result = predict(model, np.asarray(body.beats, dtype=np.float64), threshold)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return PredictResponse(
        model=request.app.state.settings.model_description,
        threshold=threshold,
        n_abnormal=int(result.labels.sum()),
        predictions=[
            BeatPrediction(beat=i + 1, probability=float(p), label=int(y), label_name=label_name(y))
            for i, (p, y) in enumerate(zip(result.probabilities, result.labels, strict=True))
        ],
    )


@app.post("/jobs", status_code=202)
async def create_job(
    request: Request,
    file: Annotated[UploadFile, File(description="CSV: 187 columns, or 188 with the label.")],
    threshold: Annotated[float | None, Form(ge=0.0, le=1.0)] = None,
) -> JSONResponse:
    """Upload a CSV of beats; it is validated now and classified in the background."""
    model = get_model(request)
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"File too large: the limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )
    try:
        beats = read_beats(io.BytesIO(data))
    except ValueError as exc:  # also covers UnicodeDecodeError for binary files
        raise HTTPException(422, f"{file.filename}: {exc}") from exc
    if len(beats) > MAX_BEATS:
        raise HTTPException(413, f"Too many beats: {len(beats)} (limit {MAX_BEATS:,} per upload).")

    settings: Settings = request.app.state.settings
    job = Job(
        id=uuid.uuid4().hex,
        filename=file.filename or "upload.csv",
        signals=beats.signals,
        labels=beats.labels,
        threshold=settings.threshold if threshold is None else threshold,
    )
    request.app.state.jobs.add(job)
    # The job starts after the response is sent. run_job is a plain function, so Starlette
    # runs it in its thread pool and GET /jobs/{id} polls are answered while it works.
    return JSONResponse(
        {"job_id": job.id, "total": job.total, "status": job.status},
        status_code=202,
        background=BackgroundTask(run_job, job, model),
    )


def job_results(job: Job) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = job.predictions()
    rows = []
    for i, (p, y) in enumerate(zip(result.probabilities, result.labels, strict=True)):
        row: dict[str, Any] = {
            "beat": i + 1,
            "probability": float(p),
            "label": int(y),
            "label_name": label_name(y),
        }
        if job.labels is not None:
            row["true_label"] = int(job.labels[i])
            row["true_label_name"] = label_name(job.labels[i])
            row["correct"] = bool(job.labels[i] == y)
        rows.append(row)
    summary: dict[str, Any] = {
        "n_total": job.total,
        "n_abnormal": int(result.labels.sum()),
        "threshold": result.threshold,
        "accuracy": None if job.labels is None else float((job.labels == result.labels).mean()),
    }
    return rows, summary


@app.get("/jobs/{job_id}")
def job_status(job_id: str, request: Request) -> dict[str, Any]:
    """Progress of an upload job; includes results and a summary once it is done."""
    job: Job = request.app.state.jobs.get(job_id)
    payload: dict[str, Any] = {
        "job_id": job.id,
        "filename": job.filename,
        "status": job.status,
        "processed": job.processed,
        "total": job.total,
        "percent": job.percent,
        "has_labels": job.labels is not None,
        "error": job.error,
    }
    if job.status == "done":
        payload["results"], payload["summary"] = job_results(job)
    return payload


@app.get("/jobs/{job_id}/beats/{beat}")
def job_beat(job_id: str, beat: int, request: Request) -> dict[str, Any]:
    """The values of one uploaded beat (1-based), so the page can draw its waveform."""
    job: Job = request.app.state.jobs.get(job_id)
    if not 1 <= beat <= job.total:
        raise HTTPException(404, f"Beat {beat} out of range (1-{job.total}).")
    return {"beat": beat, "values": [round(float(v), 6) for v in job.signals[beat - 1]]}


@app.get("/jobs/{job_id}/results.csv")
def job_results_csv(job_id: str, request: Request, threshold: float | None = None) -> Response:
    """Download the results (same columns as `uv run predict`'s reports/predictions.csv)."""
    job: Job = request.app.state.jobs.get(job_id)
    if job.status != "done":
        raise HTTPException(409, f"Job is {job.status}; results are not ready yet.")
    try:
        result = job.predictions(threshold)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    table = predictions_table(result, job.labels).drop(columns=["correct"], errors="ignore")
    name = Path(job.filename).stem or "beats"
    return Response(
        table.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}_predictions.csv"'},
    )


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str, request: Request) -> Response:
    """Forget a job and its beats (used by the page's "Clear results")."""
    request.app.state.jobs.remove(job_id)
    return Response(status_code=204)


def serve() -> None:
    """Entry point for ``uv run serve``: run the site with uvicorn on 127.0.0.1:8000."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the ECGops API and web page locally.")
    parser.add_argument("--host", default=os.environ.get("ECGOPS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ECGOPS_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="restart on code changes")
    args = parser.parse_args()
    print(f"ECGops: open http://{args.host}:{args.port} (API docs at /docs). Ctrl+C to stop.")
    uvicorn.run("ecgops.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    serve()
