"""Day 6 acceptance: predict.py contract.

Loads the persisted model (produced on Day 4) and proves five properties:

  1. predict_transaction returns exactly the expected keys with the right types
  2. Identical inputs -> identical outputs (deterministic)
  3. predict_batch on N inputs matches predict_transaction called N times
  4. predict_batch score matches pipeline.predict_proba directly (no drift)
  5. is_alert flag flips correctly around the threshold

If the model does not exist, prints an actionable error instead of failing
inside a stack trace.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import FINAL_THRESHOLD, MODEL_FEATURES, MODEL_PATH, MODEL_VERSION
from src.data.load_data import load_raw_transactions
from src.models.predict import (
    PredictionInputError,
    get_model,
    predict_batch,
    predict_transaction,
    reset_model_cache,
)
from src.models.train import prepare_train_test_split


def rule(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"ERROR: model artifact not found at {MODEL_PATH}.", file=sys.stderr)
        print("       Run `python -m src.models.train` first.", file=sys.stderr)
        return 1

    # Prime a batch of real feature vectors from the test set. Uses the same
    # split as train.py / evaluate.py so the sampled rows are representative
    # of what the served model would see.
    print("Loading a sample of feature vectors from the test set ...")
    df = load_raw_transactions()
    _, X_test, _, _ = prepare_train_test_split(df)
    sample = X_test.head(20).to_dict(orient="records")
    print(f"Loaded {len(sample)} feature vectors.\n")

    reset_model_cache()

    rule("Case 1 - predict_transaction returns the documented contract")
    result = predict_transaction(sample[0])
    expected_keys = {"risk_score", "is_alert", "threshold", "model_version"}
    assert set(result.keys()) == expected_keys, (
        f"unexpected keys: {result.keys()}"
    )
    assert isinstance(result["risk_score"], float)
    assert isinstance(result["is_alert"], bool)
    assert isinstance(result["threshold"], float)
    assert result["model_version"] == MODEL_VERSION
    print(f"  keys OK: {sorted(result.keys())}")
    print(f"  sample : risk_score={result['risk_score']:.6f}  "
          f"is_alert={result['is_alert']}  "
          f"threshold={result['threshold']:.6f}\n")

    rule("Case 2 - identical inputs -> identical outputs (determinism)")
    r_a = predict_transaction(sample[0])
    r_b = predict_transaction(sample[0])
    assert r_a == r_b, f"non-deterministic: {r_a} vs {r_b}"
    print("  identical dicts on repeat calls\n")

    rule("Case 3 - predict_batch matches predict_transaction called N times")
    batch = predict_batch(sample)
    individually = [predict_transaction(feat) for feat in sample]
    assert batch == individually, "batch and per-item outputs disagree"
    print(f"  {len(batch)} outputs, batch == [predict_transaction(...) for ...]\n")

    rule("Case 4 - risk_score matches pipeline.predict_proba directly")
    pipeline = get_model()
    df_sample = pd.DataFrame(sample)[MODEL_FEATURES]
    direct = pipeline.predict_proba(df_sample)[:, 1]
    via_batch = np.array([r["risk_score"] for r in batch])
    max_diff = float(np.max(np.abs(direct - via_batch)))
    assert max_diff == 0.0, f"drift {max_diff:.2e}"
    print(f"  max |predict_batch - predict_proba| = {max_diff:.2e}\n")

    rule("Case 5 - is_alert flag flips correctly around the threshold")
    # Pick the median-scoring row and score it at thresholds a small margin
    # below and above its own score. The margin (1e-6) is well above float
    # round-trip noise yet negligible against any realistic threshold change.
    scores = np.array([r["risk_score"] for r in batch])
    mid_idx = int(np.argsort(scores)[len(scores) // 2])
    mid_score = float(scores[mid_idx])
    epsilon = 1e-6
    below = predict_transaction(sample[mid_idx], threshold=mid_score - epsilon)
    above = predict_transaction(sample[mid_idx], threshold=mid_score + epsilon)
    assert below["is_alert"] is True, (
        f"below-threshold row not flagged (score={mid_score}, "
        f"threshold={mid_score - epsilon})"
    )
    assert above["is_alert"] is False, (
        f"above-threshold row flagged (score={mid_score}, "
        f"threshold={mid_score + epsilon})"
    )
    print(f"  row score={mid_score:.6f}")
    print(f"  threshold=score-{epsilon:.0e} -> is_alert={below['is_alert']}  (expected True)")
    print(f"  threshold=score+{epsilon:.0e} -> is_alert={above['is_alert']}  (expected False)\n")

    rule("Case 6 - malformed input raises PredictionInputError, not a raw KeyError")
    try:
        predict_transaction({"Amount Paid": 100.0})
    except PredictionInputError as e:
        print(f"  RAISED PredictionInputError:\n    {e}\n")

    print("=" * 72)
    print("DAY 6 PREDICTION CONTRACT VERIFIED")
    print(f"  threshold  : {FINAL_THRESHOLD}")
    print(f"  version    : {MODEL_VERSION}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
