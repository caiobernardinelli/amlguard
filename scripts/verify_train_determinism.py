"""Day 4 acceptance check: training determinism.

Runs ``train_model()`` twice on the SAME stratified subsample of the raw CSV
and proves the acceptance criterion:

    "Dois treinamentos independentes produzem o mesmo hash de artefato e
     as mesmas metricas dentro de tolerancia."

Uses a stratified 200,000-row subsample so this runs in ~1-2 min rather than
the ~10-15 min a full training would take twice. The property being proved
is seed-and-code driven, not scale driven: if fitting is deterministic on
this subsample, it is deterministic on the full CSV.

Usage
-----
    python scripts/verify_train_determinism.py
"""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TARGET
from src.data.load_data import load_raw_transactions
from src.models.train import train_model

SUBSAMPLE_ROWS = 200_000


def _hash_pipeline(pipeline) -> str:
    """SHA-256 of the joblib-serialised pipeline bytes."""
    buf = io.BytesIO()
    joblib.dump(pipeline, buf, compress=3)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def main() -> int:
    print(f"Loading full CSV, drawing a stratified subsample of {SUBSAMPLE_ROWS:,} rows ...")
    df_full = load_raw_transactions()
    df_sub, _ = train_test_split(
        df_full,
        train_size=SUBSAMPLE_ROWS,
        random_state=RANDOM_STATE,
        stratify=df_full[TARGET],
    )
    df_sub = df_sub.reset_index(drop=True)
    print(f"Subsample positive rate: {(df_sub[TARGET] == 1).mean():.4%}")

    print("\nRun 1 of 2 ...")
    p1, m1 = train_model(df_sub, save_artifacts=False)
    h1 = _hash_pipeline(p1)

    print("\nRun 2 of 2 ...")
    p2, m2 = train_model(df_sub, save_artifacts=False)
    h2 = _hash_pipeline(p2)

    bytes_match = h1 == h2
    metrics_match = m1["metrics"] == m2["metrics"]

    print()
    print("=" * 72)
    print(f"Run 1 SHA-256      : {h1}")
    print(f"Run 2 SHA-256      : {h2}")
    print(f"Bytes match        : {bytes_match}")
    print()
    print(f"Run 1 AP           : {m1['metrics']['average_precision']:.9f}")
    print(f"Run 2 AP           : {m2['metrics']['average_precision']:.9f}")
    print(f"Run 1 recall       : {m1['metrics']['operating_point']['recall']:.9f}")
    print(f"Run 2 recall       : {m2['metrics']['operating_point']['recall']:.9f}")
    print(f"Metrics identical  : {metrics_match}")
    print("=" * 72)

    ok = bytes_match and metrics_match
    print("DAY 4 DETERMINISM CRITERION MET" if ok else "DAY 4 DETERMINISM FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
