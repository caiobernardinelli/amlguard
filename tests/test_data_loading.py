"""Formal tests for src.data.load_data (Day 2 acceptance criterion)."""

from __future__ import annotations

import pytest

from src.config import TARGET
from src.data.load_data import (
    SchemaValidationError,
    load_raw_transactions,
    validate_schema,
)


def test_valid_schema_passes(valid_transactions_df):
    """A well-formed frame passes validation without exception."""
    validate_schema(valid_transactions_df)  # must not raise


def test_missing_column_raises_named_error(valid_transactions_df):
    """A dropped required column is named in the error message."""
    df = valid_transactions_df.drop(columns=["Payment Format"])
    with pytest.raises(SchemaValidationError, match="Payment Format"):
        validate_schema(df)


def test_non_binary_target_raises_named_error(valid_transactions_df):
    """A non-{0,1} value in the target column is flagged with the column name."""
    df = valid_transactions_df.copy()
    df.loc[0, TARGET] = 9
    with pytest.raises(SchemaValidationError, match=TARGET):
        validate_schema(df)


def test_non_numeric_amount_raises_named_error(valid_transactions_df):
    """A stringified Amount Paid column is flagged with the column name."""
    df = valid_transactions_df.copy()
    df["Amount Paid"] = df["Amount Paid"].astype(str)
    with pytest.raises(SchemaValidationError, match="Amount Paid"):
        validate_schema(df)


def test_missing_file_raises_named_path(tmp_path):
    """FileNotFoundError names the path when download is disabled."""
    with pytest.raises(FileNotFoundError, match="nowhere.csv"):
        load_raw_transactions(tmp_path / "nowhere.csv", download_if_missing=False)
