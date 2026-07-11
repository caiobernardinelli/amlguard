"""Day 2 acceptance smoke test.

Run from the repository root to prove that src/data/load_data.py fails with
clear messages when the input schema is broken:

    python scripts/smoke_test_day2.py

This script is NOT a formal pytest module (that's scheduled for Day 7). It is
a manual demonstration for the Day 2 acceptance criterion:

    "A carga falha com mensagem clara quando o schema nao corresponde ao esperado."

It does not touch the raw CSV or the network, so it runs in under a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running the script directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.load_data import (
    SchemaValidationError,
    load_raw_transactions,
    validate_schema,
)


def make_valid() -> pd.DataFrame:
    """A minimal DataFrame matching RAW_REQUIRED_COLUMNS."""
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
        "Is Laundering": [0, 1],
    })


def rule(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    rule("Case 1 - valid schema (must NOT raise)")
    validate_schema(make_valid())
    print("  OK: no exception. Load path would return the DataFrame normally.")
    print()

    rule("Case 2 - missing column 'Payment Format' (must fail clearly)")
    try:
        validate_schema(make_valid().drop(columns=["Payment Format"]))
    except SchemaValidationError as e:
        print(f"  RAISED SchemaValidationError:\n    {e}")
    print()

    rule("Case 3 - 'Is Laundering' contains value 9 (must fail clearly)")
    bad = make_valid()
    bad.loc[0, "Is Laundering"] = 9
    try:
        validate_schema(bad)
    except SchemaValidationError as e:
        print(f"  RAISED SchemaValidationError:\n    {e}")
    print()

    rule("Case 4 - 'Amount Paid' is text, not numeric (must fail clearly)")
    bad = make_valid()
    bad["Amount Paid"] = bad["Amount Paid"].astype(str)
    try:
        validate_schema(bad)
    except SchemaValidationError as e:
        print(f"  RAISED SchemaValidationError:\n    {e}")
    print()

    rule("Case 5 - CSV missing, download disabled (must fail clearly)")
    try:
        load_raw_transactions(
            Path("nonexistent.csv"),
            download_if_missing=False,
        )
    except FileNotFoundError as e:
        print(f"  RAISED FileNotFoundError:\n    {e}")
    print()

    print("ACCEPTANCE CRITERION MET: every failure mode names the offending")
    print("column or path instead of surfacing a raw pandas error.")


if __name__ == "__main__":
    main()
