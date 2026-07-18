"""Evaluate an AMLGuard cloud-trained model against the frozen baseline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.config import (
    BASELINE_METRICS_PATH,
    MIN_AVERAGE_PRECISION,
    MIN_RECALL_AT_2PCT_PRECISION,
)
from src.models.train import evaluate as evaluate_pipeline

REGRESSION_TOLERANCE = {
    "average_precision": 0.05,
    "precision": 0.05,
    "recall": 0.05,
    "f1_score": 0.10,
    "alerts": 0.10,
    "alert_rate_pct": 0.10,
    "threshold": 0.05,
}


def flatten_baseline() -> dict[str, float]:
    """Load comparable values from the frozen baseline."""

    baseline = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    final_model = baseline["final_model"]
    metrics = final_model["metrics"]
    return {
        "average_precision": float(metrics["average_precision"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "alerts": float(metrics["alerts"]),
        "alert_rate_pct": float(metrics["alert_rate_pct"]),
        "threshold": float(final_model["threshold"]),
    }


def flatten_observed(metrics: dict) -> dict[str, float]:
    """Flatten observed metrics for regression comparison."""

    operating_point = metrics["operating_point"]
    return {
        "average_precision": float(metrics["average_precision"]),
        "precision": float(operating_point["precision"]),
        "recall": float(operating_point["recall"]),
        "f1_score": float(operating_point["f1_score"]),
        "alerts": float(operating_point["alerts"]),
        "alert_rate_pct": float(operating_point["alert_rate_pct"]),
        "threshold": float(operating_point["threshold"]),
    }


def main() -> int:
    """Evaluate the model and enforce quality and regression gates."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--model-input", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    args = parser.parse_args()

    bundle_path = args.prepared_data / "prepared_split.joblib"
    model_path = args.model_input / "model.joblib"

    if not bundle_path.is_file():
        raise FileNotFoundError(f"Prepared split not found: {bundle_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    bundle = joblib.load(bundle_path)
    pipeline = joblib.load(model_path)

    X_test = bundle["X_test"]
    y_test = bundle["y_test"]
    metrics = evaluate_pipeline(pipeline, X_test, y_test)

    baseline = flatten_baseline()
    observed = flatten_observed(metrics)

    comparisons = []
    regressions = []

    for metric_name, observed_value in observed.items():
        baseline_value = baseline[metric_name]
        relative_delta = (
            (observed_value - baseline_value) / baseline_value
            if baseline_value != 0
            else 0.0
        )
        tolerance = REGRESSION_TOLERANCE[metric_name]
        status = "PASS" if abs(relative_delta) <= tolerance else "FAIL"

        comparisons.append(
            {
                "metric": metric_name,
                "baseline": baseline_value,
                "observed": observed_value,
                "relative_delta": relative_delta,
                "tolerance": tolerance,
                "status": status,
            }
        )

        if status == "FAIL":
            regressions.append(metric_name)

    quality_pass = (
        metrics["average_precision"] >= MIN_AVERAGE_PRECISION
        and metrics["operating_point"]["recall"] >= MIN_RECALL_AT_2PCT_PRECISION
    )
    regression_pass = not regressions
    overall_status = "PASS" if quality_pass and regression_pass else "FAIL"

    record = {
        "status": overall_status,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_test": int(len(y_test)),
        "metrics": metrics,
        "quality_gate": {
            "status": "PASS" if quality_pass else "FAIL",
            "min_average_precision": MIN_AVERAGE_PRECISION,
            "min_recall_at_2pct_precision": MIN_RECALL_AT_2PCT_PRECISION,
        },
        "regression_gate": {
            "status": "PASS" if regression_pass else "FAIL",
            "regressed_metrics": regressions,
            "comparisons": comparisons,
        },
    }

    args.evaluation_output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.evaluation_output / "metrics.json"
    metrics_path.write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(record, indent=2))
    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
