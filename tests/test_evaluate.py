"""Formal tests for src.models.evaluate (Day 5 acceptance criterion)."""

from __future__ import annotations

from src.models.evaluate import _compare_row, _flatten_baseline


def test_flatten_baseline_returns_expected_keys():
    """The frozen baseline JSON exposes the metrics evaluate compares against."""
    baseline = _flatten_baseline()
    for key in ("average_precision", "precision", "recall", "f1_score",
                "alerts", "alert_rate_pct", "threshold"):
        assert key in baseline, f"missing baseline key: {key}"


def test_compare_row_marks_within_tolerance_as_ok():
    """A small relative drift is marked 'ok'."""
    # AP dropped ~2.7% relative -- well within the 5% tolerance.
    row = _compare_row("average_precision", 0.037, 0.036)
    assert row["status"] == "ok"


def test_compare_row_marks_large_drift_as_regressed():
    """A drift larger than the tolerance is flagged 'regressed'."""
    # +35% is far above the 5% AP tolerance.
    row = _compare_row("average_precision", 0.037, 0.050)
    assert row["status"] == "regressed"


def test_compare_row_untracked_metric_is_info():
    """A metric without a defined tolerance is marked 'info', not regressed."""
    row = _compare_row("true_positives", 1041, 900)
    assert row["status"] == "info"
