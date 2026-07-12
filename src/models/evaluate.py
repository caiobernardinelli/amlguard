"""Evaluate the persisted model against the frozen baseline.

Loads ``artifacts/model.joblib``, rebuilds the exact same test set that
``train.py`` used (via :func:`src.models.train.prepare_train_test_split`),
recomputes the full set of metrics reported in
``artifacts/baseline_metrics.json``, and produces a structured comparison
report to stdout plus a machine-readable ``artifacts/metrics.json``.

Exits with:

    0  when both gates pass:
         * the absolute quality gate (AP >= MIN_AVERAGE_PRECISION,
           recall >= MIN_RECALL_AT_2PCT_PRECISION), and
         * every tracked metric is within the regression tolerance
           relative to ``baseline_metrics.json``.

    1  when any gate fails, with the offending metrics named in the
       final report so the diagnosis is on the last screen.

CLI
---
    python -m src.models.evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.config import (
    ARTIFACTS_DIR,
    BASELINE_METRICS_PATH,
    FINAL_MODEL_NAME,
    MIN_AVERAGE_PRECISION,
    MIN_RECALL_AT_2PCT_PRECISION,
    MODEL_METADATA_PATH,
    MODEL_PATH,
)
from src.data.load_data import load_raw_transactions
from src.models.train import evaluate as evaluate_pipeline
from src.models.train import prepare_train_test_split

logger = logging.getLogger(__name__)

# Per-metric relative tolerance for the regression check. A larger tolerance
# on `threshold` reflects that the exact operating-point score is a byproduct
# of a jagged PR curve and can shift more than the ranking-based metrics.
REGRESSION_TOLERANCE: dict[str, float] = {
    "average_precision": 0.05,   # 5%
    "precision":         0.05,
    "recall":            0.05,
    "f1_score":          0.10,
    "alerts":            0.10,
    "alert_rate_pct":    0.10,
    "threshold":         0.05,
}

METRICS_JSON_PATH = ARTIFACTS_DIR / "metrics.json"


def _flatten_baseline() -> dict[str, float]:
    """Extract the comparable metrics from ``baseline_metrics.json``."""
    baseline = json.loads(BASELINE_METRICS_PATH.read_text())
    fm = baseline["final_model"]
    m = fm["metrics"]
    return {
        "average_precision": m["average_precision"],
        "accuracy":          m["accuracy"],
        "precision":         m["precision"],
        "recall":            m["recall"],
        "f1_score":          m["f1_score"],
        "alerts":            float(m["alerts"]),
        "alert_rate_pct":    m["alert_rate_pct"],
        "threshold":         fm["threshold"],
        "true_positives":    float(m["true_positives"]),
        "false_positives":   float(m["false_positives"]),
        "false_negatives":   float(m["false_negatives"]),
        "true_negatives":    float(m["true_negatives"]),
    }


def _flatten_observed(metrics: dict) -> dict[str, float]:
    op = metrics["operating_point"]
    return {
        "average_precision": metrics["average_precision"],
        "accuracy":          op["accuracy"],
        "precision":         op["precision"],
        "recall":            op["recall"],
        "f1_score":          op["f1_score"],
        "alerts":            float(op["alerts"]),
        "alert_rate_pct":    op["alert_rate_pct"],
        "threshold":         op["threshold"],
        "true_positives":    float(op["true_positives"]),
        "false_positives":   float(op["false_positives"]),
        "false_negatives":   float(op["false_negatives"]),
        "true_negatives":    float(op["true_negatives"]),
    }


def _compare_row(name: str, baseline: float, observed: float) -> dict:
    delta = observed - baseline
    rel = delta / baseline if baseline != 0 else 0.0
    tol = REGRESSION_TOLERANCE.get(name)
    if tol is None:
        status = "info"
    else:
        status = "ok" if abs(rel) <= tol else "regressed"
    return {
        "metric": name,
        "baseline": baseline,
        "observed": observed,
        "delta": delta,
        "relative": rel,
        "tolerance": tol,
        "status": status,
    }


def _print_report(rows: list[dict], gate_status: str, regression_status: str) -> None:
    print()
    print("=" * 88)
    print(f"  Model evaluation vs frozen baseline ({FINAL_MODEL_NAME})")
    print("=" * 88)
    header = f"  {'metric':<22} {'baseline':>14} {'observed':>14} {'delta':>13} {'rel':>8}   status"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        rel_str = f"{r['relative']:+.2%}" if r["tolerance"] is not None else "   -   "
        status_glyph = {"ok": "OK", "regressed": "REGRESSED", "info": "info"}[r["status"]]
        print(
            f"  {r['metric']:<22} "
            f"{r['baseline']:>14.6f} "
            f"{r['observed']:>14.6f} "
            f"{r['delta']:>+13.6f} "
            f"{rel_str:>8}   {status_glyph}"
        )
    print("=" * 88)
    print(f"  Quality gate       (AP >= {MIN_AVERAGE_PRECISION}, "
          f"recall >= {MIN_RECALL_AT_2PCT_PRECISION}) : {gate_status}")
    print(f"  Regression gate    (per-metric tolerance): {regression_status}")
    print("=" * 88)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=MODEL_PATH,
        help=f"path to the fitted pipeline (default: {MODEL_PATH})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if not args.model.exists():
        print(f"ERROR: model artifact not found at {args.model}.", file=sys.stderr)
        print("       Run `python -m src.models.train` first.", file=sys.stderr)
        return 1

    logger.info("Loading model from %s", args.model)
    pipeline = joblib.load(args.model)

    df = load_raw_transactions()
    _X_train, X_test, _y_train, y_test = prepare_train_test_split(df)
    logger.info("Test set: %d rows, %d positives", len(y_test), int(y_test.sum()))

    metrics = evaluate_pipeline(pipeline, X_test, y_test)

    # Compare against baseline
    baseline = _flatten_baseline()
    observed = _flatten_observed(metrics)
    rows = [_compare_row(name, baseline[name], observed[name])
            for name in observed if name in baseline]

    # Two independent gates
    ap_ok = metrics["average_precision"] >= MIN_AVERAGE_PRECISION
    recall_ok = metrics["operating_point"]["recall"] >= MIN_RECALL_AT_2PCT_PRECISION
    gate_status = "PASS" if (ap_ok and recall_ok) else "FAIL"

    regressions = [r for r in rows if r["status"] == "regressed"]
    regression_status = "PASS" if not regressions else (
        f"FAIL ({', '.join(r['metric'] for r in regressions)})"
    )

    # Persist a structured record
    record = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model),
        "model_metadata": (
            json.loads(MODEL_METADATA_PATH.read_text())
            if MODEL_METADATA_PATH.exists() else None
        ),
        "n_test": int(len(y_test)),
        "metrics": metrics,
        "comparison": {
            "baseline_source": str(BASELINE_METRICS_PATH),
            "rows": rows,
            "quality_gate": {
                "status": gate_status,
                "min_average_precision": MIN_AVERAGE_PRECISION,
                "min_recall_at_2pct_precision": MIN_RECALL_AT_2PCT_PRECISION,
            },
            "regression_gate": {
                "status": regression_status,
                "tolerances": REGRESSION_TOLERANCE,
                "regressed_metrics": [r["metric"] for r in regressions],
            },
        },
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_JSON_PATH.write_text(json.dumps(record, indent=2))
    logger.info("Wrote %s", METRICS_JSON_PATH)

    _print_report(rows, gate_status, regression_status)

    return 0 if (gate_status == "PASS" and not regressions) else 1


if __name__ == "__main__":
    sys.exit(main())
