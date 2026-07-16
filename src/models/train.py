"""Training entry point — reproduces the frozen XGBoost baseline.

Loads the raw CSV via :func:`src.data.load_data.load_raw_transactions`,
builds the leakage-safe feature set from :mod:`src.features`, splits
stratified 70/30, fits the ``tree_preprocessor + XGBoost`` pipeline with
``scale_pos_weight`` derived from the training fold, evaluates on the test
set, and persists:

    artifacts/model.joblib           the fitted sklearn Pipeline
    artifacts/model_metadata.json    metrics, hyperparameters, versions,
                                     git commit, gate status

Every random source is seeded (numpy, sklearn split, XGBoost). Two
invocations on identical data must produce identical artifacts
(see ``scripts/verify_train_determinism.py``).

CLI
---
Full training on the raw CSV::

    python -m src.models.train

Fit and evaluate without writing artifacts (fast smoke test)::

    python -m src.models.train --no-save

The script exits with code 0 when the baseline gate passes
(``Average Precision >= MIN_AVERAGE_PRECISION`` and
``recall >= MIN_RECALL_AT_2PCT_PRECISION`` at the operating point),
and code 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.config import (
    ARTIFACTS_DIR,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FINAL_MODEL_NAME,
    FINAL_PRECISION_TARGET,
    MIN_AVERAGE_PRECISION,
    MIN_RECALL_AT_2PCT_PRECISION,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    XGBOOST_PARAMS,
)
from src.data.load_data import load_raw_transactions
from src.features import build_model_features
from src.models.tracking import log_training_run

logger = logging.getLogger(__name__)


def _make_tree_preprocessor() -> ColumnTransformer:
    """Preprocessor matching notebook cell 35 (``tree_preprocessor``)."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    """Build the winning pipeline: tree preprocessor + XGBoost scale_pos_weight."""
    return Pipeline(
        steps=[
            ("preprocessor", _make_tree_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    **XGBOOST_PARAMS,
                    scale_pos_weight=scale_pos_weight,
                ),
            ),
        ]
    )


def _find_threshold_for_precision(
    y_true: np.ndarray, y_score: np.ndarray, target_precision: float
) -> tuple[float, bool]:
    """Find the smallest threshold achieving ``precision >= target``.

    Returns ``(threshold, target_reached)``. If no threshold on the PR curve
    reaches the target precision, falls back to the threshold that
    **maximises** precision, and returns ``target_reached=False`` so the
    caller can log the deviation and continue instead of crashing.
    """
    precision, _recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall arrays are one entry longer than thresholds
    # (last precision entry = 1.0 boundary, no matching threshold).
    p_valid = precision[:-1]

    valid = np.where(p_valid >= target_precision)[0]
    if len(valid) > 0:
        return float(thresholds[valid[0]]), True

    # Fallback: pick the threshold that maximises precision.
    best_idx = int(np.argmax(p_valid))
    return float(thresholds[best_idx]), False


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute the metrics reported in the frozen baseline.

    Emits Average Precision and the max achievable precision as soon as the
    PR curve is computed, so a run that will not reach the target operating
    point still gives useful diagnostics before finishing.
    """
    y_score = pipeline.predict_proba(X_test)[:, 1]

    average_precision = float(average_precision_score(y_test, y_score))
    logger.info("Test Average Precision = %.6f", average_precision)

    precision_curve, _, _ = precision_recall_curve(y_test.to_numpy(), y_score)
    max_precision = float(precision_curve[:-1].max())
    logger.info(
        "Max achievable precision on test set: %.6f (target %.4f)",
        max_precision, FINAL_PRECISION_TARGET,
    )

    threshold, reached = _find_threshold_for_precision(
        y_test.to_numpy(), y_score, FINAL_PRECISION_TARGET
    )
    if not reached:
        logger.warning(
            "Target precision %.4f is unreachable on this test set. "
            "Falling back to the max-precision threshold %.6f.",
            FINAL_PRECISION_TARGET, threshold,
        )

    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    accuracy = float((tp + tn) / (tp + tn + fp + fn))
    f1_score = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall) > 0 else 0.0
    )
    n_alerts = int(y_pred.sum())

    return {
        "average_precision": average_precision,
        "max_achievable_precision": max_precision,
        "operating_point": {
            "precision_target": FINAL_PRECISION_TARGET,
            "precision_target_reached": reached,
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "alerts": n_alerts,
            "alert_rate_pct": 100.0 * n_alerts / len(y_test),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        },
    }


def _git_commit() -> str:
    """Best-effort short git SHA for provenance; ``unknown`` if the repo is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def prepare_train_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Deterministic (X_train, X_test, y_train, y_test) from a raw DataFrame.

    Sorts by ``Timestamp`` first (matching ``build_model_features``' internal
    sort) so that features and target come from identically-ordered rows,
    then stratified-splits with ``config.TEST_SIZE`` and ``config.RANDOM_STATE``.

    Both ``train.py`` and ``evaluate.py`` call this so their test sets are
    guaranteed to be identical row-by-row: any comparison against
    ``baseline_metrics.json`` is only meaningful if the test set matches.
    """
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)

    features = build_model_features(df).reset_index(drop=True)
    y = df[TARGET].astype("int8").reset_index(drop=True)

    # Defensive: features and target must have identical length and
    # the same total number of positives as the raw df.
    assert len(features) == len(y), (
        f"feature/target length mismatch: {len(features)} vs {len(y)}"
    )
    assert int(y.sum()) == int(df[TARGET].sum()), (
        "target sum drifted between raw df and post-sort extraction"
    )

    return train_test_split(
        features, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_model(
    df: pd.DataFrame,
    save_artifacts: bool = True,
    track_experiment: bool = False,
) -> tuple[Pipeline, dict]:
    """End-to-end: features -> stratified split -> fit -> evaluate.

    Parameters
    ----------
    df
        Raw transaction table (post schema validation).
    save_artifacts
        When True, writes ``artifacts/model.joblib`` and
        ``artifacts/model_metadata.json``. Set to False for smoke tests
        and the determinism check.
    track_experiment
        When True, logs the completed run to MLflow. Tracking requires ``save_artifacts=True``.

    Returns
    -------
    (fitted_pipeline, metadata_dict)
    """
    # === AMLGUARD DAY 13 VALIDATION ===
    if track_experiment and not save_artifacts:
        raise ValueError(
            "MLflow tracking requires save_artifacts=True."
        )

    logger.info("Preparing train/test split on %d rows", len(df))
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    logger.info("Split: train=%d, test=%d", len(X_train), len(X_test))

    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / positive
    logger.info(
        "scale_pos_weight = %.4f (train neg=%d, pos=%d)",
        scale_pos_weight, negative, positive,
    )

    pipeline = build_pipeline(scale_pos_weight)
    logger.info("Fitting %s ...", FINAL_MODEL_NAME)
    pipeline.fit(X_train, y_train)

    # Persist the trained model BEFORE evaluation so any downstream crash
    # (e.g., threshold search on a difficult PR curve) does not force a
    # full retrain. Metadata is written only after evaluation succeeds.
    if save_artifacts:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH, compress=3)
        logger.info(
            "Wrote %s (%.1f MB) - pre-evaluation checkpoint",
            MODEL_PATH, MODEL_PATH.stat().st_size / 1e6,
        )

    metrics = evaluate(pipeline, X_test, y_test)

    op = metrics["operating_point"]
    passed_ap = metrics["average_precision"] >= MIN_AVERAGE_PRECISION
    passed_recall = op["recall"] >= MIN_RECALL_AT_2PCT_PRECISION
    gate_status = "PASS" if (passed_ap and passed_recall) else "FAIL"

    metadata = {
        "model_name": FINAL_MODEL_NAME,
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_train_positive": positive,
        "n_train_negative": negative,
        "scale_pos_weight": scale_pos_weight,
        "hyperparameters": XGBOOST_PARAMS,
        "features": list(X_train.columns),
        "target": TARGET,
        "metrics": metrics,
        "baseline_gate": {
            "min_average_precision": MIN_AVERAGE_PRECISION,
            "min_recall_at_2pct_precision": MIN_RECALL_AT_2PCT_PRECISION,
            "status": gate_status,
        },
        "library_versions": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"),
            "xgboost": version("xgboost"),
            "joblib": version("joblib"),
        },
    }

    if save_artifacts:
        MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
        logger.info("Wrote %s", MODEL_METADATA_PATH)

    # === AMLGUARD DAY 13 TRACKING START ===
    if track_experiment:
        run_id = log_training_run(metadata)
        metadata["mlflow_run_id"] = run_id
        MODEL_METADATA_PATH.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        logger.info("Logged MLflow run %s", run_id)
    # === AMLGUARD DAY 13 TRACKING END ===

    return pipeline, metadata


def _summarise(metadata: dict) -> None:
    op = metadata["metrics"]["operating_point"]
    reached_flag = "OK" if op["precision_target_reached"] else "FALLBACK"
    print()
    print("=" * 72)
    print(f"  {metadata['model_name']}")
    print("=" * 72)
    print(f"  Average Precision           : {metadata['metrics']['average_precision']:.6f}")
    print(f"  Max achievable precision    : {metadata['metrics']['max_achievable_precision']:.6f}")
    print(f"  Precision target ({op['precision_target']:.0%})       : {reached_flag}")
    print(f"  Threshold used              : {op['threshold']:.6f}")
    print(f"  Precision                   : {op['precision']:.4f}")
    print(f"  Recall                      : {op['recall']:.4f}")
    print(f"  Alerts / alert rate         : {op['alerts']:,} ({op['alert_rate_pct']:.2f}%)")
    print(f"  True positives              : {op['true_positives']:,}")
    print(f"  False positives             : {op['false_positives']:,}")
    print(f"  False negatives             : {op['false_negatives']:,}")
    print(f"  Baseline gate               : {metadata['baseline_gate']['status']}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-save", action="store_true",
        help="fit and evaluate without writing artifacts",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="log the completed persisted training run to MLflow",
    )
    args = parser.parse_args()

    if args.no_save and args.track:
        parser.error("--track cannot be combined with --no-save")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    df = load_raw_transactions()
    _, metadata = train_model(
        df,
        save_artifacts=not args.no_save,
        track_experiment=args.track,
    )
    _summarise(metadata)

    return 0 if metadata["baseline_gate"]["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
