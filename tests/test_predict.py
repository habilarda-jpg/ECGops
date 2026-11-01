"""Tests for the `predict` CLI (ecgops.predict) on synthetic beats with a random model."""

import pandas as pd
import pytest

from ecgops import predict


def predict_cfg(compose_cfg, tmp_path, checkpoint, csv_path, plot=False):
    return compose_cfg(
        [
            f"model_out={checkpoint.as_posix()}",
            f"predict.input={csv_path.as_posix()}",
            f"predict.output={(tmp_path / 'out' / 'predictions.csv').as_posix()}",
            f"predict.figure={(tmp_path / 'out' / 'beats.png').as_posix()}",
            f"predict.plot={str(plot).lower()}",
        ]
    )


def test_expand_shortcuts():
    args = ["input=a.csv", "plot=true", "evaluation.threshold=0.3", "--help", "predict.output=x"]
    assert predict.expand_shortcuts(args) == [
        "predict.input=a.csv",
        "predict.plot=true",
        "evaluation.threshold=0.3",
        "--help",
        "predict.output=x",
    ]


def test_labeled_run_writes_predictions_and_figure(
    compose_cfg, tmp_path, model_checkpoint, beats_csv, capsys
):
    csv_path, signals, _ = beats_csv(labeled=True)
    cfg = predict_cfg(compose_cfg, tmp_path, model_checkpoint, csv_path, plot=True)

    table = predict.main(cfg)

    assert list(table.columns) == [
        "beat",
        "true_label",
        "abnormal_probability",
        "prediction",
        "correct",
    ]
    assert len(table) == len(signals)
    assert "Accuracy:" in capsys.readouterr().out
    saved = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert set(saved["predicted_label"]) <= {0, 1}
    assert saved["abnormal_probability"].between(0, 1).all()
    assert (tmp_path / "out" / "beats.png").stat().st_size > 0


def test_unlabeled_run_has_no_accuracy(compose_cfg, tmp_path, model_checkpoint, beats_csv, capsys):
    csv_path, _, _ = beats_csv(labeled=False)
    table = predict.main(predict_cfg(compose_cfg, tmp_path, model_checkpoint, csv_path))
    assert "true_label" not in table.columns
    assert "Accuracy" not in capsys.readouterr().out
    assert not (tmp_path / "out" / "beats.png").exists()  # plot=false


def test_invalid_input_exits_with_a_message(compose_cfg, tmp_path, model_checkpoint):
    bad = tmp_path / "bad.csv"
    bad.write_text("0.1,0.2,0.3\n")
    with pytest.raises(SystemExit) as excinfo:
        predict.main(predict_cfg(compose_cfg, tmp_path, model_checkpoint, bad))
    assert excinfo.value.code == 1


def test_unknown_model_source_exits(compose_cfg, tmp_path, model_checkpoint, beats_csv):
    csv_path, _, _ = beats_csv()
    cfg = predict_cfg(compose_cfg, tmp_path, model_checkpoint, csv_path)
    cfg.predict.model_source = "s3"
    with pytest.raises(SystemExit):
        predict.main(cfg)
