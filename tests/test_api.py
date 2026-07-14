"""Formal tests for src.api.main (Day 8 acceptance criterion).

Uses FastAPI's TestClient so we exercise the real HTTP layer -- routing,
serialization, status codes -- without spinning up uvicorn.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import FINAL_MODEL_NAME, FINAL_THRESHOLD, MODEL_FEATURES, MODEL_VERSION


@pytest.fixture
def client_with_model(cached_model):
    """TestClient with a pipeline pre-injected into the predict cache.

    The `cached_model` fixture from conftest.py hands us a real fitted
    pipeline via monkeypatch of predict._MODEL_CACHE, so /health and
    /model-info see a "loaded" model without touching disk.
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_without_model(monkeypatch):
    """TestClient with the model cache explicitly cleared and ``get_model``
    stubbed to raise, so the lifespan hook cannot populate the cache from disk.

    Note: we must patch the reference **inside src.api.main** (not just in
    src.models.predict), because ``main.py`` did ``from ... import get_model``
    and now holds its own binding to the original function. Patching only the
    predict module leaves the lifespan's copy untouched.
    """
    from src.models import predict

    def _raise():
        raise FileNotFoundError("test: simulated missing model artifact")

    predict.reset_model_cache()
    monkeypatch.setattr(predict, "get_model", _raise)
    monkeypatch.setattr("src.api.main.get_model", _raise)

    with TestClient(app) as client:
        yield client
    predict.reset_model_cache()


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
class TestHealth:
    def test_returns_200_and_documented_shape(self, client_with_model):
        r = client_with_model.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) == {
            "status", "service", "service_version",
            "model_loaded", "timestamp_utc",
        }
        assert body["status"] == "ok"
        assert body["service"] == "amlguard-api"
        assert isinstance(body["service_version"], str)
        assert isinstance(body["timestamp_utc"], str)

    def test_reports_model_loaded_when_cache_populated(self, client_with_model):
        assert client_with_model.get("/health").json()["model_loaded"] is True

    def test_reports_model_not_loaded_when_cache_empty(self, client_without_model):
        r = client_without_model.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is False

    def test_health_makes_no_side_effects(self, client_with_model):
        """Two consecutive calls return equivalent shapes (except timestamp)."""
        a = client_with_model.get("/health").json()
        b = client_with_model.get("/health").json()
        assert a.keys() == b.keys()
        assert a["model_loaded"] == b["model_loaded"]


# --------------------------------------------------------------------------- #
# /model-info
# --------------------------------------------------------------------------- #
class TestModelInfo:
    def test_returns_200_with_full_contract(self, client_with_model):
        r = client_with_model.get("/model-info")
        assert r.status_code == 200
        body = r.json()
        expected = {
            "model_name", "model_version", "threshold", "precision_target",
            "feature_names", "hyperparameters", "model_artifact_path",
            "trained_at_utc", "trained_from_git_commit",
        }
        assert set(body.keys()) == expected

    def test_values_come_from_config(self, client_with_model):
        body = client_with_model.get("/model-info").json()
        assert body["model_name"] == FINAL_MODEL_NAME
        assert body["model_version"] == MODEL_VERSION
        assert body["threshold"] == FINAL_THRESHOLD
        assert body["feature_names"] == MODEL_FEATURES

    def test_returns_503_when_model_absent(self, client_without_model):
        r = client_without_model.get("/model-info")
        assert r.status_code == 503
        assert "not loaded" in r.json()["detail"].lower()


# --------------------------------------------------------------------------- #
# OpenAPI schema is generated (side effect: /docs also works)
# --------------------------------------------------------------------------- #
def test_openapi_schema_available(client_with_model):
    r = client_with_model.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    paths = schema["paths"]
    assert "/health" in paths
    assert "/model-info" in paths

# === AMLGUARD DAY 9 TESTS START ===
DAY9_VALID_TRANSACTION = {
    "payment_format": "ACH",
    "amount_paid": 13701.30,
    "sender_previous_tx_count": 238,
    "is_business_hours": 0,
    "same_account": 0,
}

DAY9_ALIAS_TRANSACTION = {
    "Payment Format": "ACH",
    "Amount Paid": 13701.30,
    "sender_previous_tx_count": 238,
    "is_business_hours": 0,
    "same_account": 0,
}

DAY9_RESPONSE_FIELDS = {
    "risk_score",
    "is_alert",
    "threshold",
    "model_version",
}


def _assert_day9_prediction_shape(payload: dict[str, object]) -> None:
    assert set(payload) == DAY9_RESPONSE_FIELDS
    assert 0.0 <= payload["risk_score"] <= 1.0
    assert isinstance(payload["is_alert"], bool)
    assert 0.0 <= payload["threshold"] <= 1.0
    assert isinstance(payload["model_version"], str)
    assert payload["model_version"]
    assert payload["is_alert"] is (
        payload["risk_score"] >= payload["threshold"]
    )


def test_day9_predict_returns_valid_response(client_with_model):
    response = client_with_model.post(
        "/predict",
        json=DAY9_VALID_TRANSACTION,
    )

    assert response.status_code == 200
    _assert_day9_prediction_shape(response.json())


def test_day9_predict_accepts_public_and_model_aliases(client_with_model):
    snake_case_response = client_with_model.post(
        "/predict",
        json=DAY9_VALID_TRANSACTION,
    )
    alias_response = client_with_model.post(
        "/predict",
        json=DAY9_ALIAS_TRANSACTION,
    )

    assert snake_case_response.status_code == 200
    assert alias_response.status_code == 200
    assert snake_case_response.json() == alias_response.json()


def test_day9_predict_rejects_missing_field(client_with_model):
    payload = DAY9_VALID_TRANSACTION.copy()
    payload.pop("amount_paid")

    response = client_with_model.post("/predict", json=payload)

    assert response.status_code == 422


def test_day9_predict_rejects_wrong_type(client_with_model):
    payload = {
        **DAY9_VALID_TRANSACTION,
        "amount_paid": {"not": "a number"},
    }

    response = client_with_model.post("/predict", json=payload)

    assert response.status_code == 422


def test_day9_predict_rejects_invalid_business_hours(client_with_model):
    payload = {
        **DAY9_VALID_TRANSACTION,
        "is_business_hours": 2,
    }

    response = client_with_model.post("/predict", json=payload)

    assert response.status_code == 422


def test_day9_predict_rejects_negative_amount(client_with_model):
    payload = {
        **DAY9_VALID_TRANSACTION,
        "amount_paid": -1,
    }

    response = client_with_model.post("/predict", json=payload)

    assert response.status_code == 422


def test_day9_predict_returns_503_without_model(client_without_model):
    response = client_without_model.post(
        "/predict",
        json=DAY9_VALID_TRANSACTION,
    )

    assert response.status_code == 503


def test_day9_predict_batch_preserves_order(client_with_model):
    transactions = [
        DAY9_VALID_TRANSACTION,
        {
            **DAY9_VALID_TRANSACTION,
            "payment_format": "Wire",
            "amount_paid": 250.0,
            "sender_previous_tx_count": 2,
        },
        {
            **DAY9_VALID_TRANSACTION,
            "payment_format": "Credit Card",
            "amount_paid": 50.0,
            "sender_previous_tx_count": 0,
            "same_account": 1,
        },
    ]

    expected = [
        client_with_model.post("/predict", json=item).json()
        for item in transactions
    ]
    response = client_with_model.post(
        "/predict-batch",
        json=transactions,
    )

    assert response.status_code == 200
    assert response.json() == expected
    assert len(response.json()) == len(transactions)


def test_day9_predict_batch_accepts_empty_list(client_with_model):
    response = client_with_model.post("/predict-batch", json=[])

    assert response.status_code == 200
    assert response.json() == []


def test_day9_predict_batch_rejects_more_than_1000_items(
    client_with_model,
):
    response = client_with_model.post(
        "/predict-batch",
        json=[DAY9_VALID_TRANSACTION] * 1001,
    )

    assert response.status_code == 422


def test_day9_predict_batch_rejects_one_invalid_item(client_with_model):
    invalid_transaction = {
        **DAY9_VALID_TRANSACTION,
        "same_account": 2,
    }

    response = client_with_model.post(
        "/predict-batch",
        json=[DAY9_VALID_TRANSACTION, invalid_transaction],
    )

    assert response.status_code == 422


def test_day9_predict_batch_returns_503_without_model(
    client_without_model,
):
    response = client_without_model.post(
        "/predict-batch",
        json=[DAY9_VALID_TRANSACTION],
    )

    assert response.status_code == 503
# === AMLGUARD DAY 9 TESTS END ===
