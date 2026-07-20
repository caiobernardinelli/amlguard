"""Generate AMLGuard Day 20 data and prediction drift report.

Run from the AMLGuard repository root:

    python generate_day20_drift_report.py

Reference:
- deterministic AMLGuard X_test split from the frozen training pipeline;
- reference prediction scores produced by artifacts/model.joblib.

Current:
- Azure ML Data Collector model_inputs/model_outputs downloaded to
  .day20_collected and paired by correlationid.

The current batch is synthetic monitoring traffic. Therefore this report
demonstrates a production-style drift workflow but MUST NOT be interpreted
as evidence of real production drift.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FINAL_THRESHOLD,
    MODEL_PATH,
    NUMERIC_FEATURES,
)
from src.data.load_data import load_raw_transactions
from src.models.train import prepare_train_test_split

ROOT = Path(__file__).resolve().parents[2]
COLLECTED_DIR = ROOT / ".day20_collected"
INPUTS_PATH = COLLECTED_DIR / "model_inputs.jsonl"
OUTPUTS_PATH = COLLECTED_DIR / "model_outputs.jsonl"
EVIDENCE_PATH = ROOT / "docs" / "evidence" / "day20_drift_report.json"

EPSILON = 1e-6
PSI_LOW_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25
NUMERIC_BINS = 10


def load_collector_records(path: Path) -> list[dict[str, Any]]:
    """Load Azure ML Data Collector JSONL records."""
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            data = record.get("data")

            if not isinstance(data, list) or len(data) != 1:
                raise ValueError(
                    f"{path.name} line {line_number} must contain one data row."
                )

            row = data[0]
            correlation_id = record.get("correlationid")

            if not isinstance(row, dict):
                raise ValueError(
                    f"{path.name} line {line_number} data row is not an object."
                )
            if not isinstance(correlation_id, str) or not correlation_id:
                raise ValueError(
                    f"{path.name} line {line_number} has no correlationid."
                )

            records.append(
                {
                    "correlationid": correlation_id,
                    "data": row,
                }
            )

    return records


def pair_collected_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair collected model inputs and outputs by correlation ID."""
    input_records = load_collector_records(INPUTS_PATH)
    output_records = load_collector_records(OUTPUTS_PATH)

    inputs_by_id = {
        record["correlationid"]: record["data"]
        for record in input_records
    }
    outputs_by_id = {
        record["correlationid"]: record["data"]
        for record in output_records
    }

    if set(inputs_by_id) != set(outputs_by_id):
        raise ValueError(
            "Collected model_inputs and model_outputs correlation IDs do not match."
        )

    ordered_ids = sorted(inputs_by_id)

    current_inputs = pd.DataFrame(
        [inputs_by_id[correlation_id] for correlation_id in ordered_ids]
    )
    current_outputs = pd.DataFrame(
        [outputs_by_id[correlation_id] for correlation_id in ordered_ids]
    )

    return current_inputs, current_outputs


def safe_probabilities(counts: np.ndarray) -> np.ndarray:
    """Convert counts to smoothed probabilities."""
    counts = counts.astype(float) + EPSILON
    return counts / counts.sum()


def psi_from_probabilities(
    reference_probabilities: np.ndarray,
    current_probabilities: np.ndarray,
) -> float:
    """Calculate Population Stability Index from aligned probabilities."""
    return float(
        np.sum(
            (current_probabilities - reference_probabilities)
            * np.log(current_probabilities / reference_probabilities)
        )
    )


def classify_psi(value: float) -> str:
    """Classify PSI using AMLGuard project monitoring heuristics."""
    if value < PSI_LOW_THRESHOLD:
        return "low"
    if value < PSI_SIGNIFICANT_THRESHOLD:
        return "moderate"
    return "significant"


def categorical_psi(
    reference: pd.Series,
    current: pd.Series,
) -> dict[str, Any]:
    """Calculate PSI across the union of categorical values."""
    reference_values = reference.astype(str).fillna("<NA>")
    current_values = current.astype(str).fillna("<NA>")

    categories = sorted(
        set(reference_values.unique()) | set(current_values.unique())
    )

    reference_counts = np.array(
        [(reference_values == category).sum() for category in categories]
    )
    current_counts = np.array(
        [(current_values == category).sum() for category in categories]
    )

    reference_probabilities = safe_probabilities(reference_counts)
    current_probabilities = safe_probabilities(current_counts)

    value = psi_from_probabilities(
        reference_probabilities,
        current_probabilities,
    )

    return {
        "psi": value,
        "signal": classify_psi(value),
        "category_count": len(categories),
    }


def numeric_bin_edges(reference: pd.Series) -> np.ndarray:
    """Create stable quantile-derived bin edges from reference data."""
    values = pd.to_numeric(reference, errors="raise").to_numpy(dtype=float)

    quantiles = np.linspace(0.0, 1.0, NUMERIC_BINS + 1)
    edges = np.unique(np.quantile(values, quantiles))

    if len(edges) < 2:
        center = float(edges[0])
        return np.array([-math.inf, center, math.inf], dtype=float)

    edges = edges.astype(float)
    edges[0] = -math.inf
    edges[-1] = math.inf

    return edges


def numeric_psi(
    reference: pd.Series,
    current: pd.Series,
) -> dict[str, Any]:
    """Calculate PSI using quantile bins derived from reference values."""
    edges = numeric_bin_edges(reference)

    reference_values = pd.to_numeric(reference, errors="raise").to_numpy(dtype=float)
    current_values = pd.to_numeric(current, errors="raise").to_numpy(dtype=float)

    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)

    reference_probabilities = safe_probabilities(reference_counts)
    current_probabilities = safe_probabilities(current_counts)

    value = psi_from_probabilities(
        reference_probabilities,
        current_probabilities,
    )

    return {
        "psi": value,
        "signal": classify_psi(value),
        "bin_count": int(len(edges) - 1),
    }


def score_summary(scores: np.ndarray) -> dict[str, float]:
    """Return a compact score distribution summary."""
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "p05": float(np.quantile(scores, 0.05)),
        "p25": float(np.quantile(scores, 0.25)),
        "p75": float(np.quantile(scores, 0.75)),
        "p95": float(np.quantile(scores, 0.95)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def main() -> int:
    """Generate and persist the Day 20 drift report."""
    if not INPUTS_PATH.is_file() or not OUTPUTS_PATH.is_file():
        raise FileNotFoundError(
            "Collected model_inputs/model_outputs files are missing from "
            ".day20_collected."
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}. "
            "The frozen baseline model is required for prediction drift."
        )

    current_inputs, current_outputs = pair_collected_data()

    raw = load_raw_transactions(download_if_missing=False)
    _, reference_inputs, _, _ = prepare_train_test_split(raw)

    model = joblib.load(MODEL_PATH)
    reference_scores = model.predict_proba(reference_inputs)[:, 1]

    current_scores = pd.to_numeric(
        current_outputs["score"],
        errors="raise",
    ).to_numpy(dtype=float)

    data_drift: dict[str, Any] = {}

    for feature in CATEGORICAL_FEATURES:
        data_drift[feature] = {
            "type": "categorical",
            **categorical_psi(
                reference_inputs[feature],
                current_inputs[feature],
            ),
        }

    for feature in NUMERIC_FEATURES:
        data_drift[feature] = {
            "type": "numeric",
            **numeric_psi(
                reference_inputs[feature],
                current_inputs[feature],
            ),
        }

    for feature in BINARY_FEATURES:
        data_drift[feature] = {
            "type": "binary",
            **categorical_psi(
                reference_inputs[feature],
                current_inputs[feature],
            ),
        }

    prediction_drift = numeric_psi(
        pd.Series(reference_scores),
        pd.Series(current_scores),
    )

    reference_alerts = reference_scores >= FINAL_THRESHOLD
    current_alerts = current_scores >= FINAL_THRESHOLD

    reference_alert_rate = float(np.mean(reference_alerts))
    current_alert_rate = float(np.mean(current_alerts))

    significant_features = sorted(
        feature
        for feature, report in data_drift.items()
        if report["signal"] == "significant"
    )
    moderate_features = sorted(
        feature
        for feature, report in data_drift.items()
        if report["signal"] == "moderate"
    )

    overall_signal = "low"
    if (
        significant_features
        or prediction_drift["signal"] == "significant"
    ):
        overall_signal = "significant"
    elif (
        moderate_features
        or prediction_drift["signal"] == "moderate"
    ):
        overall_signal = "moderate"

    report = {
        "day": 20,
        "scope": "data_and_prediction_drift",
        "status": "PASS",
        "interpretation_scope": "synthetic_monitoring_demonstration",
        "warning": (
            "The current dataset is synthetic monitoring traffic generated for "
            "Day 20. Drift signals demonstrate the monitoring workflow and must "
            "not be interpreted as evidence of real production population drift."
        ),
        "reference": {
            "source": "deterministic_frozen_pipeline_test_split",
            "records": int(len(reference_inputs)),
            "model_artifact": "artifacts/model.joblib",
            "threshold": FINAL_THRESHOLD,
        },
        "current": {
            "source": "azure_ml_model_data_collector",
            "records": int(len(current_inputs)),
            "paired_input_output_records": int(len(current_inputs)),
            "traffic_type": "synthetic_monitoring_traffic",
        },
        "psi_policy": {
            "description": "AMLGuard project monitoring heuristic",
            "low": f"PSI < {PSI_LOW_THRESHOLD}",
            "moderate": (
                f"{PSI_LOW_THRESHOLD} <= PSI < "
                f"{PSI_SIGNIFICANT_THRESHOLD}"
            ),
            "significant": f"PSI >= {PSI_SIGNIFICANT_THRESHOLD}",
        },
        "data_drift": data_drift,
        "prediction_drift": {
            "metric": "PSI",
            **prediction_drift,
            "reference_score_summary": score_summary(reference_scores),
            "current_score_summary": score_summary(current_scores),
        },
        "alert_rate_shift": {
            "threshold": FINAL_THRESHOLD,
            "reference_alert_rate": reference_alert_rate,
            "current_alert_rate": current_alert_rate,
            "absolute_percentage_point_change": (
                (current_alert_rate - reference_alert_rate) * 100.0
            ),
        },
        "summary": {
            "overall_signal": overall_signal,
            "significant_drift_features": significant_features,
            "moderate_drift_features": moderate_features,
            "prediction_drift_signal": prediction_drift["signal"],
        },
        "retraining_policy_preview": {
            "automatic_retraining": False,
            "recommended_action_for_this_batch": (
                "Do not retrain from this synthetic batch. Use the result only "
                "to validate the monitoring pipeline."
            ),
            "production_principle": (
                "A production retraining decision should require sustained drift "
                "on representative traffic plus model-quality evidence when "
                "ground-truth labels become available."
            ),
        },
        "privacy": (
            "The report stores aggregate drift statistics only. Raw collected "
            "transaction records, correlation IDs, request IDs, Azure resource "
            "IDs, and credentials are excluded."
        ),
    }

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("AMLGuard Day 20 - Data & Prediction Drift Report")
    print("=" * 54)
    print(f"Reference records: {len(reference_inputs)}")
    print(f"Current collected records: {len(current_inputs)}")
    print()
    print("DATA DRIFT (PSI)")
    for feature, feature_report in data_drift.items():
        print(
            f"- {feature}: "
            f"PSI={feature_report['psi']:.6f} "
            f"signal={feature_report['signal']}"
        )

    print()
    print(
        "PREDICTION DRIFT: "
        f"PSI={prediction_drift['psi']:.6f} "
        f"signal={prediction_drift['signal']}"
    )
    print(
        "REFERENCE ALERT RATE: "
        f"{reference_alert_rate:.2%}"
    )
    print(
        "CURRENT ALERT RATE: "
        f"{current_alert_rate:.2%}"
    )
    print(
        "ALERT RATE CHANGE: "
        f"{(current_alert_rate - reference_alert_rate) * 100.0:+.2f} pp"
    )
    print()
    print(f"OVERALL DRIFT SIGNAL: {overall_signal}")
    print(
        "INTERPRETATION: synthetic monitoring demonstration; "
        "not production drift evidence"
    )
    print(f"Evidence written: {EVIDENCE_PATH.relative_to(ROOT)}")
    print("Day 20 drift report: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
