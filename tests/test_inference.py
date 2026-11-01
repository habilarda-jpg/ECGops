"""Tests for ecgops.inference with a randomly initialised model: no real data, no MLflow."""

import io
import subprocess
import sys

import numpy as np
import pytest
import torch

from ecgops.data import SIGNAL_LENGTH
from ecgops.inference import (
    ModelUnavailableError,
    load_model,
    load_registry_model,
    predict,
    read_beats,
)


def test_read_labeled_csv(beats_csv):
    path, signals, labels = beats_csv(labeled=True)
    beats = read_beats(path)
    assert beats.signals.shape == (len(signals), SIGNAL_LENGTH)
    assert beats.signals.dtype == np.float32
    np.testing.assert_allclose(beats.signals, signals, atol=1e-6)
    np.testing.assert_array_equal(beats.labels, labels)


def test_read_unlabeled_csv(beats_csv):
    path, signals, _ = beats_csv(labeled=False)
    beats = read_beats(path)
    assert beats.signals.shape == (len(signals), SIGNAL_LENGTH)
    assert beats.labels is None


def test_read_beats_from_an_open_file(beats_csv):
    path, signals, _ = beats_csv(labeled=False)
    beats = read_beats(io.StringIO(path.read_text()))  # what an upload handler would pass
    assert len(beats) == len(signals)


@pytest.mark.parametrize("n_columns", [100, 186, 189])
def test_wrong_column_count_is_rejected(tmp_path, n_columns):
    path = tmp_path / "bad.csv"
    np.savetxt(path, np.full((3, n_columns), 0.5), delimiter=",")
    with pytest.raises(ValueError, match=rf"Expected 187 columns .* got {n_columns}"):
        read_beats(path)


@pytest.mark.parametrize("bad_value", [1.5, -0.1])
def test_out_of_range_values_are_rejected(beats_csv, bad_value):
    path, signals, _ = beats_csv(labeled=False)
    signals[0, 10] = bad_value
    np.savetxt(path, signals, delimiter=",")
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        read_beats(path)


def test_bad_labels_and_text_are_rejected(tmp_path):
    rows = np.full((2, SIGNAL_LENGTH + 1), 0.5)
    rows[:, -1] = [0, 2]
    np.savetxt(tmp_path / "labels.csv", rows, delimiter=",")
    with pytest.raises(ValueError, match="Labels must be 0 or 1"):
        read_beats(tmp_path / "labels.csv")

    (tmp_path / "text.csv").write_text(",".join(["0.5"] * (SIGNAL_LENGTH - 1) + ["abc"]))
    with pytest.raises(ValueError, match="non-numeric"):
        read_beats(tmp_path / "text.csv")

    (tmp_path / "empty.csv").write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_beats(tmp_path / "empty.csv")


def test_predict_returns_one_probability_per_beat(model_checkpoint, beats_csv):
    path, signals, _ = beats_csv(labeled=True)
    model = load_model(model_checkpoint)
    assert not model.training  # loaded in eval mode

    result = predict(model, read_beats(path).signals, threshold=0.5)

    assert result.probabilities.shape == (len(signals),)
    assert ((result.probabilities >= 0) & (result.probabilities <= 1)).all()
    np.testing.assert_array_equal(result.labels, (result.probabilities >= 0.5).astype(int))
    assert set(np.unique(result.labels)) <= {0, 1}


def test_predict_accepts_tensors_and_batches_consistently(model_checkpoint):
    model = load_model(model_checkpoint)
    beats = np.random.default_rng(0).random((7, SIGNAL_LENGTH))
    whole = predict(model, beats)
    chunked = predict(model, torch.from_numpy(beats).unsqueeze(1), batch_size=2)
    np.testing.assert_allclose(whole.probabilities, chunked.probabilities, atol=1e-5)


def test_threshold_controls_the_labels(model_checkpoint):
    model = load_model(model_checkpoint)
    beats = np.random.default_rng(1).random((5, SIGNAL_LENGTH))
    assert predict(model, beats, threshold=0.0).labels.all()
    strict = predict(model, beats, threshold=1.0)
    np.testing.assert_array_equal(strict.labels, (strict.probabilities >= 1.0).astype(int))
    with pytest.raises(ValueError, match="threshold"):
        predict(model, beats, threshold=1.5)


def test_predict_validates_its_input(model_checkpoint):
    model = load_model(model_checkpoint)
    with pytest.raises(ValueError, match="shape"):
        predict(model, np.zeros((3, 100)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        predict(model, np.full((2, SIGNAL_LENGTH), 2.0))


def test_missing_or_foreign_checkpoint_is_reported(tmp_path):
    with pytest.raises(ModelUnavailableError, match="uv run train"):
        load_model(tmp_path / "missing.pt")
    torch.save({"weights": 1}, tmp_path / "other.pt")
    with pytest.raises(ModelUnavailableError, match="not an ecgops checkpoint"):
        load_model(tmp_path / "other.pt")


def test_unreachable_registry_tells_you_to_start_mlflow():
    with pytest.raises(ModelUnavailableError, match="docker compose up -d mlflow"):
        load_registry_model("ecg-vgg1d", 8, tracking_uri="http://127.0.0.1:9", timeout=1)


def test_inference_does_not_import_mlflow():
    # Fresh interpreter: other tests in this session have already imported mlflow.
    code = "import sys, ecgops.inference; assert 'mlflow' not in sys.modules, 'mlflow imported'"
    subprocess.run([sys.executable, "-c", code], check=True)
