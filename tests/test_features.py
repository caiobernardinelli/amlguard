"""Formal tests for src.features (Day 3 acceptance criterion: parity + no-target)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from src.config import MODEL_FEATURES, TARGET
from src.features import build_model_features


def test_output_has_expected_columns(synthetic_signal_df):
    """build_model_features returns exactly MODEL_FEATURES in order."""
    features = build_model_features(synthetic_signal_df.head(1000))
    assert list(features.columns) == MODEL_FEATURES


def test_row_count_preserved(synthetic_signal_df):
    """Feature building neither drops nor duplicates rows."""
    subset = synthetic_signal_df.head(1000)
    features = build_model_features(subset)
    assert len(features) == len(subset)


def test_sender_prev_count_leakage_safe(synthetic_signal_df):
    """First tx per account is 0; subsequent txs increase strictly by 1.

    This is the leakage-safety property: sender_previous_tx_count uses only
    information available BEFORE the current row.
    """
    subset = synthetic_signal_df.head(2000).copy()
    features = build_model_features(subset)
    subset_sorted = subset.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    subset_sorted = subset_sorted.assign(prev=features["sender_previous_tx_count"].values)

    for _, group in subset_sorted.groupby("Account", sort=False):
        prev = group["prev"].to_numpy()
        assert prev[0] == 0, "first observed tx per account should be 0"
        if len(prev) > 1:
            assert (np.diff(prev) == 1).all(), "cumcount must be strictly +1 per row"


def test_build_features_works_without_target_column(synthetic_signal_df):
    """The feature builder must not depend on the target being present."""
    df_no_target = synthetic_signal_df.head(1000).drop(columns=[TARGET])
    features = build_model_features(df_no_target)
    assert list(features.columns) == MODEL_FEATURES


def test_no_target_subscript_in_source():
    """AST scan: no code path reads df['Is Laundering'] in the module.

    Docstrings and comments mentioning the target string are ignored
    (AST does not walk into them), so documenting the target is fine.
    """
    module = (
        Path(__file__).resolve().parents[1]
        / "src" / "features" / "build_features.py"
    )
    tree = ast.parse(module.read_text(encoding="utf-8"))
    offending = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            key = node.slice
            if isinstance(key, ast.Constant) and key.value == TARGET:
                offending.append(node.lineno)
    assert not offending, f"target subscript found at lines {offending}"
