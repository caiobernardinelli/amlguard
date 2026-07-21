"""FastAPI scoring service — administrative endpoints (Day 8).

Exposes the AMLGuard model as an HTTP service. This first day of the API
adds only the two administrative endpoints; ``/predict`` is scheduled for
Day 9 once the request/response contract has its own file and tests.

Endpoints
---------
* ``GET /health``       — liveness probe used by orchestrators (Docker,
                          Kubernetes, cloud load balancers) to decide
                          whether the instance is up and can receive
                          traffic. Zero side effects, no model access.
* ``GET /model-info``   — introspection of the currently-loaded pipeline:
                          model name, version, threshold, training git
                          commit, key hyperparameters. Used for auditing
                          and by client applications to display which
                          model produced a given score.

Design
------
The FastAPI app loads the model **once** at process start (via the
lifespan hook) and reuses the cache built by :mod:`src.models.predict`.
Every request after the first pays no model-loading cost.

Run locally::

    uvicorn src.api.main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive schema.
"""

from __future__ import annotations

import json
import logging
import typing as _day9_typing
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import fastapi as _day9_fastapi
import pydantic as _day9_pydantic
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.config import (
    FINAL_MODEL_NAME,
    FINAL_PRECISION_TARGET,
    FINAL_THRESHOLD,
    MODEL_FEATURES,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    XGBOOST_PARAMS,
)
from src.models import predict as _day9_predict
from src.models.predict import get_model

logger = logging.getLogger(__name__)

SERVICE_NAME = "amlguard-api"
SERVICE_VERSION = "1.0.1"


# --------------------------------------------------------------------------- #
# Response models — Pydantic types double as validation and OpenAPI schema.
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    """Liveness signal for orchestrators."""

    status: str = Field(examples=["ok"])
    service: str = Field(examples=[SERVICE_NAME])
    service_version: str = Field(examples=[SERVICE_VERSION])
    model_loaded: bool = Field(
        description="True when the pipeline artifact is loaded in memory."
    )
    timestamp_utc: str = Field(
        description="ISO-8601 UTC timestamp of this response."
    )


class ModelInfoResponse(BaseModel):
    """Introspection of the loaded pipeline."""

    model_name: str
    model_version: str
    threshold: float = Field(
        description="Decision threshold used by /predict for is_alert."
    )
    precision_target: float = Field(
        description="Business operating-point target the threshold was chosen for."
    )
    feature_names: list[str]
    hyperparameters: dict
    model_artifact_path: str
    trained_at_utc: str | None = Field(
        default=None,
        description="From model_metadata.json when available; otherwise null.",
    )
    trained_from_git_commit: str | None = Field(
        default=None,
        description="From model_metadata.json when available; otherwise null.",
    )


# --------------------------------------------------------------------------- #
# Lifespan — load the model on startup so requests never pay the load cost.
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model cache before the server starts accepting requests."""
    try:
        get_model()
        logger.info("Model loaded from %s at startup", MODEL_PATH)
    except FileNotFoundError:
        # We let the app start even without a model so /health can report
        # model_loaded=False. Orchestrators can then decide to hold traffic
        # until the artifact is provisioned.
        logger.warning(
            "Model artifact not found at %s. /health will report "
            "model_loaded=false until an artifact is available.",
            MODEL_PATH,
        )
    yield
    # Nothing to clean up on shutdown yet.


app = FastAPI(
    title="AMLGuard API",
    description=(
        "Anti-money-laundering scoring service. Wraps the XGBoost pipeline "
        "trained on the IBM AML HI-Small dataset."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


def _model_is_loaded() -> bool:
    """Non-throwing probe: True iff the pipeline is currently in cache."""
    from src.models import predict as _predict
    return _predict._MODEL_CACHE is not None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["administrative"],
    summary="Liveness probe",
)
def health() -> HealthResponse:
    """Return a lightweight liveness signal.

    This endpoint intentionally does no work beyond checking whether the
    model cache is populated. Orchestrators can call it hundreds of times
    per minute without cost.
    """
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        service_version=SERVICE_VERSION,
        model_loaded=_model_is_loaded(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["administrative"],
    summary="Introspect the loaded pipeline",
)
def model_info() -> ModelInfoResponse:
    """Describe the currently-loaded pipeline for auditing and clients."""
    if not _model_is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model artifact not loaded (expected at {MODEL_PATH}). "
                "Run `python -m src.models.train` and restart the service."
            ),
        )

    trained_at: str | None = None
    trained_commit: str | None = None
    if MODEL_METADATA_PATH.exists():
        try:
            meta = json.loads(MODEL_METADATA_PATH.read_text())
            trained_at = meta.get("trained_at_utc")
            trained_commit = meta.get("git_commit")
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read model metadata: %s", exc)

    return ModelInfoResponse(
        model_name=FINAL_MODEL_NAME,
        model_version=MODEL_VERSION,
        threshold=FINAL_THRESHOLD,
        precision_target=FINAL_PRECISION_TARGET,
        feature_names=MODEL_FEATURES,
        hyperparameters=XGBOOST_PARAMS,
        model_artifact_path=str(MODEL_PATH),
        trained_at_utc=trained_at,
        trained_from_git_commit=trained_commit,
    )

# === AMLGUARD DAY 9 START ===
class TransactionRequest(_day9_pydantic.BaseModel):
    """Validated transaction features accepted by the scoring API."""

    model_config = _day9_pydantic.ConfigDict(
        populate_by_name=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "payment_format": "ACH",
                "amount_paid": 13701.30,
                "sender_previous_tx_count": 238,
                "is_business_hours": 0,
                "same_account": 0,
            }
        },
    )

    payment_format: str = _day9_pydantic.Field(
        alias="Payment Format",
        min_length=1,
        description="Payment channel used by the transaction.",
    )
    amount_paid: float = _day9_pydantic.Field(
        alias="Amount Paid",
        ge=0,
        description="Transaction amount paid in the source currency.",
    )
    sender_previous_tx_count: int = _day9_pydantic.Field(
        ge=0,
        description="Number of earlier transactions observed for the sender.",
    )
    is_business_hours: int = _day9_pydantic.Field(
        ge=0,
        le=1,
        description="1 when the transaction occurred from 08:00 through 18:59.",
    )
    same_account: int = _day9_pydantic.Field(
        ge=0,
        le=1,
        description="1 when source and destination account identifiers match.",
    )

    @_day9_pydantic.field_validator("payment_format")
    @classmethod
    def validate_payment_format(cls, value: str) -> str:
        """Reject empty or whitespace-only payment-format values."""

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("payment_format must not be empty")
        return stripped_value

    def to_model_features(self) -> dict[str, object]:
        """Translate the public API schema to the model's feature names."""

        return self.model_dump(by_alias=True)


class PredictionResponse(_day9_pydantic.BaseModel):
    """Stable public response returned by AMLGuard prediction endpoints."""

    risk_score: float = _day9_pydantic.Field(ge=0, le=1)
    is_alert: bool
    threshold: float = _day9_pydantic.Field(ge=0, le=1)
    model_version: str


TransactionBatch = _day9_typing.Annotated[
    list[TransactionRequest],
    _day9_fastapi.Body(
        max_length=1000,
        description="Ordered list of up to 1,000 transactions.",
    ),
]


_MODEL_UNAVAILABLE_EXCEPTIONS = (
    FileNotFoundError,
    OSError,
    RuntimeError,
)


def _raise_model_unavailable(exc: Exception | None = None) -> None:
    """Translate a missing model artifact into a stable HTTP 503 response."""

    error = _day9_fastapi.HTTPException(
        status_code=_day9_fastapi.status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Model artifact is unavailable.",
    )
    if exc is None:
        raise error
    raise error from exc


def _require_model_available() -> None:
    """Fail fast before inference when the cached model cannot be loaded."""

    try:
        model = get_model()
    except _MODEL_UNAVAILABLE_EXCEPTIONS as exc:
        _raise_model_unavailable(exc)

    if model is None:
        _raise_model_unavailable()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=_day9_fastapi.status.HTTP_200_OK,
    summary="Score one transaction",
)
def predict_endpoint(transaction: TransactionRequest) -> dict[str, object]:
    """Validate and score one transaction with the cached AMLGuard model."""

    _require_model_available()

    try:
        return _day9_predict.predict_transaction(
            transaction.to_model_features()
        )
    except _MODEL_UNAVAILABLE_EXCEPTIONS as exc:
        _raise_model_unavailable(exc)


@app.post(
    "/predict-batch",
    response_model=list[PredictionResponse],
    status_code=_day9_fastapi.status.HTTP_200_OK,
    summary="Score an ordered transaction batch",
)
def predict_batch_endpoint(
    transactions: TransactionBatch,
) -> list[dict[str, object]]:
    """Validate and score up to 1,000 transactions in the original order."""

    _require_model_available()

    model_features = [
        transaction.to_model_features() for transaction in transactions
    ]

    try:
        return _day9_predict.predict_batch(model_features)
    except _MODEL_UNAVAILABLE_EXCEPTIONS as exc:
        _raise_model_unavailable(exc)
# === AMLGUARD DAY 9 END ===
