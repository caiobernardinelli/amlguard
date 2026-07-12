"""Day 3 acceptance check: feature parity and target independence.

The plan's acceptance criterion for Day 3 (13/07):

    "A mesma entrada produz as mesmas features do notebook e nenhuma
     feature utiliza o alvo."

This script proves both properties:

1. PARITY  — each feature produced by ``src.features.build_model_features``
   is recomputed here with an INDEPENDENT inline implementation copied from
   the notebook cells (Section 5), and the two results must be identical.
2. NO-TARGET — the feature module's source code must never reference the
   target column, and building features from a frame WITHOUT the target
   must succeed (structural proof that the target is not consumed).

Usage
-----
Quick logic check on a synthetic frame (no CSV, <1 s)::

    python scripts/verify_features_parity.py --synthetic

Full check on the real HI-Small CSV (downloads it if absent)::

    python scripts/verify_features_parity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import MODEL_FEATURES, TARGET
from src.features import build_model_features


def make_synthetic(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Random frame with the raw schema, shuffled timestamps and repeated accounts."""
    rng = np.random.default_rng(seed)
    ts = pd.Timestamp("2022-09-01") + pd.to_timedelta(
        rng.integers(0, 18 * 24 * 3600, n), unit="s"
    )
    accounts = rng.integers(0, n // 20, n).astype(str)
    receivers = rng.integers(0, n // 20, n).astype(str)
    return pd.DataFrame({
        "Timestamp": ts.astype(str),
        "From Bank": rng.integers(1, 100, n),
        "Account": accounts,
        "To Bank": rng.integers(1, 100, n),
        "Account.1": receivers,
        "Amount Received": rng.exponential(500, n).round(2),
        "Receiving Currency": rng.choice(["USD", "EUR", "BRL"], n),
        "Amount Paid": rng.exponential(500, n).round(2),
        "Payment Currency": rng.choice(["USD", "EUR", "BRL"], n),
        "Payment Format": rng.choice(
            ["ACH", "Wire", "Cheque", "Cash", "Reinvestment"], n
        ),
        TARGET: (rng.random(n) < 0.001).astype(int),
    })


def notebook_reference_features(df: pd.DataFrame) -> pd.DataFrame:
    """INDEPENDENT re-implementation copied from the notebook (Section 5).

    Deliberately does not import from src.features — this is the reference
    the package implementation must match.
    """
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Hour"] = df["Timestamp"].dt.hour
    df["is_business_hours"] = df["Hour"].between(8, 18).astype("int8")
    df["same_account"] = (df["Account"] == df["Account.1"]).astype("int8")
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    df["sender_previous_tx_count"] = (
        df.groupby("Account", sort=False).cumcount().astype("int32")
    )
    return df[MODEL_FEATURES].copy()


def check_parity(raw: pd.DataFrame) -> None:
    print("1. PARITY - package output vs independent notebook re-implementation")
    package_out = build_model_features(raw)
    reference_out = notebook_reference_features(raw)

    assert list(package_out.columns) == list(reference_out.columns) == MODEL_FEATURES

    for col in MODEL_FEATURES:
        left = package_out[col].reset_index(drop=True)
        right = reference_out[col].reset_index(drop=True)
        if col == "Payment Format":
            equal = (left.astype(str) == right.astype(str)).all()
        elif col == "Amount Paid":
            equal = np.allclose(
                left.astype("float64"), right.astype("float64"), atol=0, rtol=0
            )
        else:
            equal = (left.to_numpy() == right.to_numpy()).all()
        status = "OK " if equal else "FAIL"
        print(f"   [{status}] {col}")
        assert equal, f"Feature mismatch: {col}"
    print(f"   All {len(MODEL_FEATURES)} features identical on {len(raw):,} rows.")


def check_no_target() -> None:
    print("2. NO-TARGET - features must not read the label")

    # 2a. AST-level: no expression in the module reads a subscript whose key
    #     equals the target column name (e.g. df["Is Laundering"]). Docstrings
    #     and comments are ignored by the AST, so mentioning the target in
    #     documentation is fine -- only real reads are caught.
    import ast

    module_path = (
        Path(__file__).resolve().parents[1]
        / "src" / "features" / "build_features.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))

    offending = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            key = node.slice
            if isinstance(key, ast.Constant) and key.value == TARGET:
                offending.append(node.lineno)
    assert not offending, (
        f"Target subscript found in build_features.py at lines {offending}"
    )
    print(f"   [OK ] no code path reads df['{TARGET}'] in build_features.py")

    # 2b. Behavioural: building features on a frame WITHOUT the target succeeds.
    #     Any hidden read of the target would raise KeyError here.
    frame = make_synthetic(2_000).drop(columns=[TARGET])
    out = build_model_features(frame)
    assert list(out.columns) == MODEL_FEATURES
    print("   [OK ] build_model_features() succeeds on a frame with no target")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="run on a synthetic 10k-row frame instead of the real CSV",
    )
    args = parser.parse_args()

    if args.synthetic:
        print("Mode: SYNTHETIC (10,000 rows, seed 42)\n")
        raw = make_synthetic()
    else:
        print("Mode: REAL CSV (HI-Small, ~5.08M rows)\n")
        from src.data.load_data import load_raw_transactions
        raw = load_raw_transactions()

    check_parity(raw)
    print()
    check_no_target()
    print()
    print("DAY 3 ACCEPTANCE CRITERION MET:")
    print("  same input -> same features as the notebook; no feature uses the target.")


if __name__ == "__main__":
    main()
