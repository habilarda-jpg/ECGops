"""Tests for ecgops.tracking. No MLflow server is needed: every test runs offline."""

from omegaconf import OmegaConf

from ecgops.tracking import ExperimentTracker, mlflow_server_reachable


def offline_cfg():
    return OmegaConf.create(
        {"tracking": {"mlflow": {"enabled": False}, "wandb": {"enabled": False}}}
    )


def test_unreachable_mlflow_server_is_reported_not_raised():
    assert mlflow_server_reachable("http://127.0.0.1:9", timeout=1) is False
    assert mlflow_server_reachable("file:./mlruns", timeout=1) is True


def test_failing_mlflow_call_disables_mlflow_instead_of_raising():
    tracker = ExperimentTracker(offline_cfg(), run_name="test", run_type="test")
    tracker.use_mlflow = True  # pretend the server was reachable at start

    def broken(*_args, **_kwargs):
        raise ConnectionResetError("connection reset by peer")

    assert tracker.mlflow_call("log_metrics", broken, {"loss": 1.0}) is None
    assert tracker.use_mlflow is False
    tracker.log_metrics({"loss": 1.0}, step=1)  # now a no-op, must not raise


def test_disabled_tracker_is_a_silent_no_op(tmp_path):
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}")
    with ExperimentTracker(offline_cfg(), run_name="test", run_type="test") as tracker:
        assert tracker.use_mlflow is False and tracker.use_wandb is False
        tracker.log_metrics({"loss": 0.5}, step=1)
        tracker.set_tags({"k": "v"})
        tracker.log_artifact(artifact)
    assert tracker.run_id is None
