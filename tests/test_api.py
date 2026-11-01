"""Tests for the FastAPI service (ecgops.api) with a tiny random model; no MLflow server.

Needs the serving group (a default group, so `uv sync` installs it). Without it the module
is skipped locally; CI sets ECGOPS_REQUIRE_SERVING=1 so a missing install fails instead of skipping.
"""

import io
import os

import numpy as np
import pytest
import torch

if not os.environ.get("ECGOPS_REQUIRE_SERVING"):
    pytest.importorskip("fastapi", reason="serving group not installed")

from fastapi.testclient import TestClient  # noqa: E402

from ecgops import api  # noqa: E402
from ecgops.data import SIGNAL_LENGTH  # noqa: E402
from ecgops.inference import load_model, predict  # noqa: E402


@pytest.fixture
def client(monkeypatch, model_checkpoint, beats_csv):
    samples_path, _, _ = beats_csv(labeled=True)
    monkeypatch.setenv("ECGOPS_MODEL_SOURCE", "local")
    monkeypatch.setenv("ECGOPS_MODEL_PATH", str(model_checkpoint))
    monkeypatch.setenv("ECGOPS_SAMPLES_PATH", str(samples_path))
    monkeypatch.setenv("ECGOPS_THRESHOLD", "0.5")
    with TestClient(api.app) as test_client:  # runs the lifespan (loads the model)
        yield test_client


@pytest.fixture
def client_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("ECGOPS_MODEL_SOURCE", "local")
    monkeypatch.setenv("ECGOPS_MODEL_PATH", str(tmp_path / "missing.pt"))
    with TestClient(api.app) as test_client:
        yield test_client


def csv_bytes(rows: np.ndarray) -> bytes:
    buffer = io.StringIO()
    np.savetxt(buffer, rows, delimiter=",", fmt="%.6g")
    return buffer.getvalue().encode()


def upload(client, content: bytes, name: str = "beats.csv"):
    return client.post("/jobs", files={"file": (name, content, "text/csv")})


# ---------------------------------------------------------------- health, page, samples


def test_health_with_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_source"] == "local"
    assert body["threshold"] == 0.5
    assert body["error"] is None


def test_health_without_model_and_503_for_predictions(client_without_model):
    body = client_without_model.get("/health").json()
    assert body["status"] == "model_unavailable" and body["model_loaded"] is False
    assert "uv run train" in body["error"]

    beat = [[0.5] * SIGNAL_LENGTH]
    response = client_without_model.post("/predict", json={"beats": beat})
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]
    assert upload(client_without_model, csv_bytes(np.full((2, 187), 0.5))).status_code == 503


def test_page_static_files_and_docs_are_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "not a medical diagnostic tool" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/docs").status_code == 200


def test_samples_returns_values_and_true_labels(client):
    body = client.get("/samples").json()
    assert len(body["beats"]) == 6
    first = body["beats"][0]
    assert len(first["values"]) == SIGNAL_LENGTH
    assert first["label"] in (0, 1) and first["label_name"] in ("normal", "abnormal")


# ---------------------------------------------------------------- /predict


def test_predict_matches_the_inference_module(client, model_checkpoint):
    beats = np.random.default_rng(0).random((4, SIGNAL_LENGTH))
    body = client.post("/predict", json={"beats": beats.tolist(), "threshold": 0.3}).json()

    expected = predict(load_model(model_checkpoint), beats, threshold=0.3)
    probabilities = [p["probability"] for p in body["predictions"]]
    np.testing.assert_allclose(probabilities, expected.probabilities, atol=1e-6)
    assert [p["label"] for p in body["predictions"]] == expected.labels.tolist()
    assert body["threshold"] == 0.3
    assert all(0 <= p <= 1 for p in probabilities)


@pytest.mark.parametrize(
    ("beats", "message"),
    [
        ([[0.5] * 100], "exactly 187 values"),
        ([], "exactly 187 values"),
        ([[0.5] * 187, [0.5] * 186], "exactly 187 values"),
        ([[1.5] + [0.5] * 186], r"must lie in [0, 1]"),
    ],
)
def test_predict_rejects_bad_beats_with_a_readable_422(client, beats, message):
    response = client.post("/predict", json={"beats": beats})
    assert response.status_code == 422
    assert message in response.json()["detail"]


def test_predict_rejects_a_bad_threshold(client):
    response = client.post("/predict", json={"beats": [[0.5] * 187], "threshold": 2})
    assert response.status_code == 422


# ---------------------------------------------------------------- upload jobs


def test_upload_job_flow(client):
    rng = np.random.default_rng(1)
    rows = np.column_stack([rng.random((40, SIGNAL_LENGTH)), np.arange(40) % 2])
    created = upload(client, csv_bytes(rows), "demo.csv")
    assert created.status_code == 202
    job_id, total = created.json()["job_id"], created.json()["total"]
    assert total == 40

    for _ in range(100):  # poll until done
        status = client.get(f"/jobs/{job_id}").json()
        assert status["status"] in ("queued", "running", "done")
        if status["status"] == "done":
            break
    assert status["percent"] == 100
    assert status["processed"] == status["total"] == 40
    assert len(status["results"]) == 40
    assert status["summary"]["n_total"] == 40
    assert 0 <= status["summary"]["accuracy"] <= 1
    assert {"beat", "probability", "label", "true_label", "correct"} <= status["results"][0].keys()

    beat = client.get(f"/jobs/{job_id}/beats/3").json()
    np.testing.assert_allclose(beat["values"], rows[2, :SIGNAL_LENGTH], atol=1e-5)
    assert client.get(f"/jobs/{job_id}/beats/41").status_code == 404

    download = client.get(f"/jobs/{job_id}/results.csv")
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("text/csv")
    assert "demo_predictions.csv" in download.headers["content-disposition"]
    lines = download.text.strip().splitlines()
    assert lines[0] == "beat,true_label,abnormal_probability,predicted_label,prediction"
    assert len(lines) == 41

    everything = client.get(f"/jobs/{job_id}/results.csv?threshold=0").text.splitlines()[1:]
    assert all(line.endswith(",1,abnormal") for line in everything)


def test_unlabeled_upload_has_no_accuracy(client):
    created = upload(client, csv_bytes(np.full((5, SIGNAL_LENGTH), 0.25)))
    status = client.get(f"/jobs/{created.json()['job_id']}").json()
    assert status["status"] == "done"
    assert status["summary"]["accuracy"] is None
    assert "true_label" not in status["results"][0]


def test_invalid_upload_is_rejected_with_the_cli_message(client):
    response = upload(client, csv_bytes(np.full((3, 100), 0.5)), "short.csv")
    assert response.status_code == 422
    assert "short.csv: Expected 187 columns" in response.json()["detail"]

    response = upload(client, b"\x89PNG\r\n\x1a\n\x00\x00binary", "image.png")
    assert response.status_code == 422


def test_oversized_uploads_are_rejected(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 1000)
    response = upload(client, csv_bytes(np.full((5, SIGNAL_LENGTH), 0.5)))
    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]

    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 5 * 1024 * 1024)
    monkeypatch.setattr(api, "MAX_BEATS", 3)
    response = upload(client, csv_bytes(np.full((5, SIGNAL_LENGTH), 0.5)))
    assert response.status_code == 413
    assert "Too many beats" in response.json()["detail"]


def test_unknown_job_is_404(client):
    for path in ("/jobs/nope", "/jobs/nope/results.csv", "/jobs/nope/beats/1"):
        response = client.get(path)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


def test_delete_job_forgets_it(client):
    created = upload(client, csv_bytes(np.full((3, SIGNAL_LENGTH), 0.5)))
    job_id = created.json()["job_id"]
    assert client.get(f"/jobs/{job_id}").status_code == 200

    response = client.delete(f"/jobs/{job_id}")
    assert response.status_code == 204
    assert client.get(f"/jobs/{job_id}").status_code == 404
    assert client.get(f"/jobs/{job_id}/results.csv").status_code == 404
    assert client.delete(f"/jobs/{job_id}").status_code == 404


# ---------------------------------------------------------------- job internals


def test_run_job_updates_progress_after_every_batch(model_checkpoint):
    model = load_model(model_checkpoint)
    job = api.Job(
        id="j",
        filename="x.csv",
        signals=np.full((7, SIGNAL_LENGTH), 0.5),
        labels=None,
        threshold=0.5,
    )
    seen = []
    original_forward = model.forward

    def recording_forward(x: torch.Tensor) -> torch.Tensor:
        seen.append((job.processed, len(x)))
        return original_forward(x)

    model.forward = recording_forward  # type: ignore[method-assign]
    assert job.status == "queued" and job.percent == 0

    api.run_job(job, model, batch_size=3)

    assert seen == [(0, 3), (3, 3), (6, 1)]  # progress advanced between batches
    assert job.status == "done" and job.processed == 7 and job.percent == 100
    assert job.probabilities is not None and job.probabilities.shape == (7,)


def test_job_store_caps_and_expires(monkeypatch):
    store = api.JobStore(max_jobs=2, ttl_seconds=60)

    def job(job_id, finished_at=None):
        j = api.Job(job_id, "f.csv", np.zeros((1, SIGNAL_LENGTH)), None, 0.5)
        j.finished_at = finished_at
        return j

    store.add(job("a", finished_at=0.0))  # finished long ago: expired
    store.add(job("b"))
    store.add(job("c"))  # "a" is evicted to make room
    with pytest.raises(api.HTTPException) as excinfo:
        store.add(job("d"))  # b and c still running
    assert excinfo.value.status_code == 429
    with pytest.raises(api.HTTPException):
        store.get("a")
    assert store.get("b").id == "b"
