"""Tests for AMLGuard MLflow experiment tracking."""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import SimpleNamespace

import pytest

from src.models import tracking
from src.models.train import train_model


def _metadata() -> dict:
    return {
        "model_name": "XGBoost - Scale Pos Weight",
        "model_version": "0.1.0",
        "git_commit": "abcdef123456",
        "random_state": 42,
        "test_size": 0.30,
        "n_train": 700,
        "n_test": 300,
        "n_train_positive": 7,
        "n_train_negative": 693,
        "scale_pos_weight": 99.0,
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.08,
        },
        "features": [
            "Payment Format",
            "Amount Paid",
            "sender_previous_tx_count",
            "is_business_hours",
            "same_account",
        ],
        "target": "Is Laundering",
        "metrics": {
            "average_precision": 0.036833,
            "max_achievable_precision": 0.08,
            "operating_point": {
                "precision_target": 0.02,
                "precision_target_reached": True,
                "threshold": 0.892163,
                "accuracy": 0.966,
                "precision": 0.02,
                "recall": 0.6703,
                "f1_score": 0.0388,
                "alerts": 52_044,
                "alert_rate_pct": 3.42,
                "true_positives": 1_041,
                "false_positives": 51_003,
                "false_negatives": 512,
                "true_negatives": 1_471_947,
            },
        },
        "baseline_gate": {
            "min_average_precision": 0.035,
            "min_recall_at_2pct_precision": 0.65,
            "status": "PASS",
        },
    }


class _FakeRun(AbstractContextManager):
    def __init__(self, run_id: str) -> None:
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uri = None
        self.experiment_name = None
        self.run_name = None
        self.params = {}
        self.metrics = {}
        self.tags = {}
        self.dicts = []
        self.artifacts = []

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.experiment_name = name

    def start_run(self, run_name=None):
        self.run_name = run_name
        return _FakeRun("run-123")

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def set_tags(self, tags):
        self.tags.update(tags)

    def log_dict(self, dictionary, artifact_file):
        self.dicts.append((dictionary, artifact_file))

    def log_artifact(self, local_path, artifact_path=None):
        self.artifacts.append((local_path, artifact_path))


def test_tracking_payload_contains_model_and_operating_metrics():
    metadata = _metadata()

    params = tracking._build_params(metadata)
    metrics = tracking._build_metrics(metadata)

    assert params["xgb_n_estimators"] == 200
    assert params["feature_count"] == 5
    assert metrics["average_precision"] == 0.036833
    assert metrics["recall"] == 0.6703
    assert metrics["false_positives"] == 51_003


def test_resolve_tracking_uri_prefers_explicit_then_environment(
    monkeypatch,
):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///environment")

    assert tracking.resolve_tracking_uri("file:///explicit") == (
        "file:///explicit"
    )
    assert tracking.resolve_tracking_uri() == "file:///environment"


def test_log_training_run_records_contract(monkeypatch, tmp_path):
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr(tracking, "_import_mlflow", lambda: fake_mlflow)

    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model_metadata.json"
    baseline_path = tmp_path / "baseline_metrics.json"
    metrics_path = tmp_path / "metrics.json"
    config_path = tmp_path / "src" / "config.py"
    config_path.parent.mkdir(parents=True)

    for path in (
        model_path,
        metadata_path,
        baseline_path,
        metrics_path,
        config_path,
    ):
        path.write_text("test", encoding="utf-8")

    monkeypatch.setattr(tracking, "MODEL_PATH", model_path)
    monkeypatch.setattr(
        tracking,
        "MODEL_METADATA_PATH",
        metadata_path,
    )
    monkeypatch.setattr(
        tracking,
        "BASELINE_METRICS_PATH",
        baseline_path,
    )
    monkeypatch.setattr(tracking, "METRICS_JSON_PATH", metrics_path)
    monkeypatch.setattr(tracking, "PROJECT_ROOT", tmp_path)

    run_id = tracking.log_training_run(
        _metadata(),
        tracking_uri="file:///tracking",
        experiment_name="AMLGuard-Test",
    )

    assert run_id == "run-123"
    assert fake_mlflow.tracking_uri == "file:///tracking"
    assert fake_mlflow.experiment_name == "AMLGuard-Test"
    assert fake_mlflow.params["random_state"] == 42
    assert fake_mlflow.metrics["threshold"] == 0.892163
    assert fake_mlflow.tags["baseline_gate"] == "PASS"
    assert fake_mlflow.dicts[0][0]["mlflow_run_id"] == "run-123"
    assert {artifact for _, artifact in fake_mlflow.artifacts} == {
        "model",
        "metadata",
        "baseline",
        "evaluation",
        "configuration",
    }


def test_tracking_requires_persisted_artifacts():
    with pytest.raises(ValueError, match="save_artifacts=True"):
        train_model(
            None,
            save_artifacts=False,
            track_experiment=True,
        )


# === AMLGUARD DAY 13 SQLITE TEST START ===
def test_default_tracking_uri_uses_sqlite(monkeypatch, tmp_path):
    """The default backend is SQLite, not the legacy filesystem store."""

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    database_path = tmp_path / "mlflow.db"
    monkeypatch.setattr(tracking, "MLFLOW_DB_PATH", database_path)

    expected = f"sqlite:///{database_path.resolve().as_posix()}"
    assert tracking.resolve_tracking_uri() == expected
# === AMLGUARD DAY 13 SQLITE TEST END ===


# === AMLGUARD DAY 13 LEGACY METADATA TEST START ===
def test_build_metrics_supports_legacy_metadata_without_accuracy():
    """Older model metadata can be logged without an explicit accuracy."""

    metadata = _metadata()
    operating_point = metadata["metrics"]["operating_point"]
    expected_accuracy = (
        operating_point["true_positives"]
        + operating_point["true_negatives"]
    ) / sum(
        operating_point[name]
        for name in (
            "true_positives",
            "false_positives",
            "false_negatives",
            "true_negatives",
        )
    )
    operating_point.pop("accuracy")

    metrics = tracking._build_metrics(metadata)

    assert metrics["accuracy"] == expected_accuracy
    assert metrics["average_precision"] == 0.036833
    assert metrics["recall"] == 0.6703
# === AMLGUARD DAY 13 LEGACY METADATA TEST END ===
