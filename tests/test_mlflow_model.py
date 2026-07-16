"""Tests for the versioned AMLGuard MLflow model contract."""

from __future__ import annotations

from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import FINAL_THRESHOLD, MODEL_FEATURES, MODEL_VERSION
from src.models.mlflow_model import (
    AMLGuardPythonModel,
    _assert_prediction_parity,
    _find_registered_version,
    _predict_with_pipeline,
    build_input_example,
)


class _FakePipeline:
    def predict_proba(self, frame):
        scores = np.where(
            frame["Amount Paid"].to_numpy() >= 50_000.0,
            0.95,
            0.10,
        )
        return np.column_stack([1.0 - scores, scores])


def test_input_example_matches_frozen_feature_contract():
    example = build_input_example()

    assert list(example.columns) == MODEL_FEATURES
    assert len(example) == 2
    assert example["Amount Paid"].dtype == "float64"
    assert example["sender_previous_tx_count"].dtype == "int64"


def test_prediction_output_preserves_threshold_and_version():
    output = _predict_with_pipeline(
        _FakePipeline(),
        build_input_example(),
    )

    assert list(output.columns) == [
        "risk_score",
        "is_alert",
        "threshold",
        "model_version",
    ]
    assert output["risk_score"].tolist() == [0.95, 0.10]
    assert output["is_alert"].tolist() == [True, False]
    assert output["threshold"].tolist() == [
        FINAL_THRESHOLD,
        FINAL_THRESHOLD,
    ]
    assert output["model_version"].tolist() == [
        MODEL_VERSION,
        MODEL_VERSION,
    ]


def test_mlflow_wrapper_loads_pipeline_artifact(tmp_path):
    pipeline_path = tmp_path / "pipeline.joblib"
    joblib.dump(_FakePipeline(), pipeline_path)

    model = AMLGuardPythonModel(
        threshold=0.90,
        model_version="test-version",
        required_features=MODEL_FEATURES,
    )
    context = SimpleNamespace(
        artifacts={"pipeline": str(pipeline_path)}
    )
    model.load_context(context)
    output = model.predict(context, build_input_example())

    assert output["risk_score"].tolist() == [0.95, 0.10]
    assert output["is_alert"].tolist() == [True, False]
    assert output["threshold"].tolist() == [0.90, 0.90]
    assert output["model_version"].tolist() == [
        "test-version",
        "test-version",
    ]


def test_mlflow_wrapper_rejects_missing_feature():
    example = build_input_example().drop(columns=["same_account"])
    model = AMLGuardPythonModel(
        threshold=FINAL_THRESHOLD,
        model_version=MODEL_VERSION,
        required_features=MODEL_FEATURES,
    )
    model._pipeline = _FakePipeline()

    with pytest.raises(ValueError, match="same_account"):
        model.predict(None, example)


def test_registered_version_selection_prefers_current_run():
    client = SimpleNamespace(
        search_model_versions=lambda _filter: [
            SimpleNamespace(version="1", run_id="older"),
            SimpleNamespace(version="2", run_id="current"),
            SimpleNamespace(version="3", run_id="other"),
        ]
    )

    selected = _find_registered_version(
        client,
        "AMLGuard",
        "current",
    )

    assert selected == "2"


def test_prediction_parity_detects_score_drift():
    expected = pd.DataFrame(
        {
            "risk_score": [0.90],
            "is_alert": [True],
            "threshold": [0.80],
            "model_version": ["0.1.0"],
        }
    )
    observed = expected.copy()
    observed.loc[0, "risk_score"] = 0.70

    with pytest.raises(RuntimeError, match="scores differ"):
        _assert_prediction_parity(expected, observed)
