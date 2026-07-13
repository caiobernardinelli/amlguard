"""Formal tests for src.models.predict (Day 6 acceptance criterion)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FINAL_THRESHOLD, MODEL_FEATURES, MODEL_VERSION
from src.features import build_model_features
from src.models.predict import (
    PredictionInputError,
    predict_batch,
    predict_transaction,
)


@pytest.fixture
def sample_features(synthetic_signal_df, cached_model):
    """20 feature dicts drawn from the synthetic frame."""
    features_df = build_model_features(synthetic_signal_df.head(1000)).head(20)
    return features_df.to_dict(orient="records")


def test_transaction_returns_documented_keys(sample_features, cached_model):
    """The response has exactly {risk_score, is_alert, threshold, model_version}."""
    result = predict_transaction(sample_features[0])
    assert set(result.keys()) == {"risk_score", "is_alert", "threshold", "model_version"}
    assert isinstance(result["risk_score"], float)
    assert isinstance(result["is_alert"], bool)
    assert isinstance(result["threshold"], float)
    assert result["model_version"] == MODEL_VERSION


def test_transaction_deterministic(sample_features, cached_model):
    """Repeated calls with identical input return identical dicts."""
    assert predict_transaction(sample_features[0]) == predict_transaction(sample_features[0])


def test_batch_equals_individual(sample_features, cached_model):
    """predict_batch(N inputs) equals [predict_transaction(x) for x in inputs]."""
    batch = predict_batch(sample_features)
    individual = [predict_transaction(f) for f in sample_features]
    assert batch == individual


def test_score_matches_predict_proba(sample_features, cached_model):
    """risk_score is byte-identical to pipeline.predict_proba (no drift)."""
    batch = predict_batch(sample_features)
    df = pd.DataFrame(sample_features)[MODEL_FEATURES]
    direct = cached_model.predict_proba(df)[:, 1]
    via_batch = np.array([r["risk_score"] for r in batch])
    assert float(np.max(np.abs(direct - via_batch))) == 0.0


def test_alert_flag_flips_around_threshold(sample_features, cached_model):
    """is_alert = True when score >= threshold, False when score < threshold."""
    batch = predict_batch(sample_features)
    scores = np.array([r["risk_score"] for r in batch])
    mid_idx = int(np.argsort(scores)[len(scores) // 2])
    mid_score = float(scores[mid_idx])
    eps = 1e-6

    below = predict_transaction(sample_features[mid_idx], threshold=mid_score - eps)
    above = predict_transaction(sample_features[mid_idx], threshold=mid_score + eps)
    assert below["is_alert"] is True
    assert above["is_alert"] is False


def test_missing_features_raise_input_error(cached_model):
    """Malformed input raises PredictionInputError, not a raw KeyError."""
    with pytest.raises(PredictionInputError, match="missing required feature"):
        predict_transaction({"Amount Paid": 100.0})


def test_default_threshold_matches_config(sample_features, cached_model):
    """When threshold is omitted, the response echoes FINAL_THRESHOLD."""
    result = predict_transaction(sample_features[0])
    assert result["threshold"] == FINAL_THRESHOLD
