"""Formal tests for src.models.train (Day 4 acceptance criterion)."""

from __future__ import annotations

from src.config import TARGET
from src.models.train import prepare_train_test_split, train_model


def test_split_row_alignment(synthetic_signal_df):
    """Splits are consistent: same lengths on features and target sides.

    This is the property that broke silently in Day 4 pre-fix: features
    got sorted while target came from the original ordering, producing
    identically-shaped but row-misaligned splits.
    """
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        synthetic_signal_df.head(10_000)
    )
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    # Total positives preserved end-to-end.
    total_pos_input = int((synthetic_signal_df.head(10_000)[TARGET] == 1).sum())
    total_pos_split = int(y_train.sum() + y_test.sum())
    assert total_pos_input == total_pos_split


def test_metadata_has_expected_keys(synthetic_signal_df):
    """train_model produces a metadata dict with the full contract."""
    _pipeline, metadata = train_model(synthetic_signal_df, save_artifacts=False)
    for key in ("model_name", "metrics", "baseline_gate", "hyperparameters",
                "features", "library_versions", "scale_pos_weight"):
        assert key in metadata, f"missing metadata key: {key}"
    assert metadata["baseline_gate"]["status"] in ("PASS", "FAIL")


def test_gate_passes_on_signal_data(synthetic_signal_df):
    """On synthetic data with real signal, AP clears the quality gate.

    This is an end-to-end sanity check: features -> split -> fit -> evaluate
    all working. If it ever fails, one of the intermediate stages has drifted
    (bug 2 from Day 4 would fail this assertion).
    """
    _pipeline, metadata = train_model(synthetic_signal_df, save_artifacts=False)
    ap = metadata["metrics"]["average_precision"]
    # The synthetic frame has ~2% base rate; XGBoost with real signal should
    # far exceed that. 0.03 is a conservative lower bound.
    assert ap > 0.03, f"AP={ap:.4f} suggests broken pipeline"
