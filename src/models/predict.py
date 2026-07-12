"""Standardised prediction contract for AMLGuard.

Two entry points:

    predict_transaction(features)              single-transaction convenience
    predict_batch(features_list)               batch, one round-trip

Both load the persisted pipeline once (cached in-process) from
``artifacts/model.joblib`` and apply the config-defined threshold. Both are
stateless and deterministic: identical inputs produce identical outputs.

Feature engineering is **not** performed here. Callers pass in feature dicts
whose keys match :data:`src.config.MODEL_FEATURES`. This keeps the serving
layer free of any dependency on transaction history: a production system
would compute the leakage-safe features upstream (feature store, streaming
job) and hand this API a vector. See :func:`src.features.build_model_features`
for the CSV -> features helper.

Return shape (per transaction)::

    {
        "risk_score": float,        # P(laundering) in [0, 1]
        "is_alert":   bool,         # risk_score >= threshold
        "threshold":  float,        # threshold used for this decision
        "model_version": str,       # from src.config.MODEL_VERSION
    }
"""

from __future__ import annotations

import logging
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    FINAL_THRESHOLD,
    MODEL_FEATURES,
    MODEL_PATH,
    MODEL_VERSION,
)

__all__ = [
    "PredictionInputError",
    "get_model",
    "predict_batch",
    "predict_transaction",
    "reset_model_cache",
]

logger = logging.getLogger(__name__)


class PredictionInputError(ValueError):
    """Raised when an input feature dict is missing required keys."""


# Module-level cache: the model is loaded exactly once per process.
_MODEL_CACHE: Pipeline | None = None


def get_model() -> Pipeline:
    """Return the cached pipeline, loading it from disk on first call.

    Raises
    ------
    FileNotFoundError
        If ``artifacts/model.joblib`` does not exist. Fix by running
        ``python -m src.models.train`` first.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_PATH}. "
                "Run `python -m src.models.train` to produce it."
            )
        logger.info("Loading model from %s", MODEL_PATH)
        _MODEL_CACHE = joblib.load(MODEL_PATH)
    return _MODEL_CACHE


def reset_model_cache() -> None:
    """Clear the cached model so the next :func:`get_model` reloads.

    Intended for tests and hot-reload workflows. Never call from production
    code; the cache exists precisely to avoid the ~100 ms reload cost per
    request.
    """
    global _MODEL_CACHE
    _MODEL_CACHE = None


def _validate_features(features_list: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of feature dicts into a DataFrame with the expected columns.

    Extra keys are silently ignored (useful when a feature store hands us
    dicts that include ``transaction_id``, ``timestamp`` etc). Missing keys
    raise :class:`PredictionInputError` naming exactly what is absent.
    """
    if not features_list:
        return pd.DataFrame(columns=MODEL_FEATURES)

    df = pd.DataFrame(features_list)
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise PredictionInputError(
            f"Input is missing required feature(s): {missing}. "
            f"Expected features: {MODEL_FEATURES}."
        )
    return df[MODEL_FEATURES]


def predict_batch(
    features_list: list[dict[str, Any]],
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Score a batch of transactions.

    Parameters
    ----------
    features_list
        List of dicts. Each dict must include every key in
        :data:`src.config.MODEL_FEATURES`. Extra keys are ignored.
    threshold
        Decision threshold. If ``None``, uses :data:`src.config.FINAL_THRESHOLD`
        (the operating point frozen alongside the baseline).

    Returns
    -------
    list[dict]
        One dict per input transaction, in the same order.
    """
    if not features_list:
        return []

    used_threshold = float(FINAL_THRESHOLD if threshold is None else threshold)
    pipeline = get_model()
    df = _validate_features(features_list)

    scores = pipeline.predict_proba(df)[:, 1]

    return [
        {
            "risk_score": float(score),
            "is_alert": bool(score >= used_threshold),
            "threshold": used_threshold,
            "model_version": MODEL_VERSION,
        }
        for score in scores
    ]


def predict_transaction(
    features: dict[str, Any],
    threshold: float | None = None,
) -> dict[str, Any]:
    """Score a single transaction (convenience wrapper around :func:`predict_batch`)."""
    return predict_batch([features], threshold=threshold)[0]
