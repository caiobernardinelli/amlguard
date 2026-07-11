"""Feature engineering for AMLGuard.

Each function mirrors a feature validated in ``notebooks/01_aml_pipeline.ipynb``
(Section 5), so the training scripts and the FastAPI service reuse exactly the
same logic. The set of model inputs is defined once in :mod:`src.config`.

Design rules
------------
* No function reads the target column (``Is Laundering``) — features are
  leakage-safe by construction.
* ``sender_previous_tx_count`` counts only transactions that happened
  *earlier in time* for the same sending account, so every row uses only
  information available at prediction time.
"""

from __future__ import annotations

import pandas as pd

from src.config import MODEL_FEATURES

__all__ = [
    "MODEL_FEATURES",
    "add_temporal_features",
    "add_same_account_flag",
    "add_sender_previous_tx_count",
    "build_model_features",
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse ``Timestamp`` and derive ``Hour`` and ``is_business_hours`` (08:00–18:59)."""
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Hour"] = df["Timestamp"].dt.hour
    df["is_business_hours"] = df["Hour"].between(8, 18).astype("int8")
    return df


def add_same_account_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag transactions where origin and destination account are identical."""
    df = df.copy()
    df["same_account"] = (df["Account"] == df["Account.1"]).astype("int8")
    return df


def add_sender_previous_tx_count(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe sender velocity: number of *earlier* transactions by the sender.

    Rows are chronologically sorted first (stable mergesort), then
    ``groupby().cumcount()`` counts prior activity only. The first observed
    transaction of an account gets 0. No target information is used.
    """
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    df["sender_previous_tx_count"] = (
        df.groupby("Account", sort=False).cumcount().astype("int32")
    )
    return df


def build_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature chain and return exactly ``MODEL_FEATURES``.

    Expects the raw IBM AML transaction schema (HI-Small). Timestamp parsing
    happens inside :func:`add_temporal_features`, so raw CSV input is fine.
    """
    df = add_temporal_features(df)
    df = add_same_account_flag(df)
    df = add_sender_previous_tx_count(df)

    out = df[MODEL_FEATURES].copy()
    out["Payment Format"] = out["Payment Format"].astype("category")
    out["Amount Paid"] = out["Amount Paid"].astype("float32")
    return out
