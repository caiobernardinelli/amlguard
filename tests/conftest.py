"""Shared fixtures for the AMLGuard test suite.

Fixtures deliberately avoid the raw CSV so the suite runs in CI without
network access or the 476 MB dataset. Signal is manufactured explicitly
(ACH transactions have a 10x higher positive rate) so trained models
exceed the baseline quality gate without the full data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET


@pytest.fixture
def valid_transactions_df() -> pd.DataFrame:
    """Minimal valid AML frame, 2 rows, for schema tests."""
    return pd.DataFrame({
        "Timestamp": ["2022/09/01 10:00", "2022/09/01 11:00"],
        "From Bank": [1, 2],
        "Account": ["A1", "B1"],
        "To Bank": [2, 3],
        "Account.1": ["A2", "B2"],
        "Amount Received": [100.0, 250.5],
        "Receiving Currency": ["USD", "EUR"],
        "Amount Paid": [100.0, 250.5],
        "Payment Currency": ["USD", "EUR"],
        "Payment Format": ["ACH", "Wire"],
        TARGET: [0, 1],
    })


@pytest.fixture(scope="session")
def synthetic_signal_df() -> pd.DataFrame:
    """60k-row synthetic frame with a clear feature/target relationship.

    ACH transactions carry a 5% positive rate, non-ACH 0.5%. XGBoost picks
    this up easily, so a model trained on this frame clears the baseline
    quality gate (AP >= 0.035). Session-scoped: built once per test run.
    """
    rng = np.random.default_rng(0)
    N = 60_000
    ts = pd.Timestamp("2022-09-01") + pd.to_timedelta(
        rng.integers(0, 18 * 24 * 3600, N), unit="s"
    )
    pf = rng.choice(["ACH", "Wire", "Cheque", "Cash"], N)
    labels = (rng.random(N) < np.where(pf == "ACH", 0.05, 0.005)).astype(int)
    return pd.DataFrame({
        "Timestamp": ts.astype(str),
        "From Bank": rng.integers(1, 100, N),
        "Account": rng.integers(0, N // 30, N).astype(str),
        "To Bank": rng.integers(1, 100, N),
        "Account.1": rng.integers(0, N // 30, N).astype(str),
        "Amount Received": rng.exponential(500, N).round(2),
        "Receiving Currency": rng.choice(["USD", "EUR", "BRL"], N),
        "Amount Paid": rng.exponential(500, N).round(2),
        "Payment Currency": rng.choice(["USD", "EUR", "BRL"], N),
        "Payment Format": pf,
        TARGET: labels,
    })


@pytest.fixture(scope="session")
def trained_pipeline(synthetic_signal_df):
    """A fitted pipeline over ``synthetic_signal_df``. Trained once per run."""
    from src.models.train import train_model
    pipeline, _metadata = train_model(synthetic_signal_df, save_artifacts=False)
    return pipeline


@pytest.fixture
def cached_model(trained_pipeline, monkeypatch):
    """Inject ``trained_pipeline`` into ``src.models.predict``'s in-process cache.

    Predict-layer tests can then exercise the real code path without loading
    a joblib from disk. The monkeypatch reverts automatically after each test.
    """
    from src.models import predict
    predict.reset_model_cache()
    monkeypatch.setattr(predict, "_MODEL_CACHE", trained_pipeline)
    yield trained_pipeline
    predict.reset_model_cache()
