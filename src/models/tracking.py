"""MLflow experiment tracking for AMLGuard training runs.

The tracking layer records the already-computed training contract without
changing the model, threshold, or frozen quality gates.

CLI
---
Log the currently persisted model and metadata without retraining::

    python -m src.models.tracking
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from src.config import (
    BASELINE_METRICS_PATH,
    MLFLOW_DB_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
)

METRICS_JSON_PATH = MODEL_PATH.parent / "metrics.json"


def _import_mlflow() -> ModuleType:
    """Import MLflow lazily so core training and CI stay lightweight."""

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is not installed. Run "
            '`python -m pip install -e ".[mlops]"` first.'
        ) from exc
    return mlflow


def _sqlite_tracking_uri(database_path: Path) -> str:
    """Build a cross-platform absolute SQLite URI."""

    return f"sqlite:///{database_path.resolve().as_posix()}"


def resolve_tracking_uri(explicit_uri: str | None = None) -> str:
    """Resolve a CLI override, environment variable, or local SQLite backend."""

    return (
        explicit_uri
        or os.getenv("MLFLOW_TRACKING_URI")
        or _sqlite_tracking_uri(MLFLOW_DB_PATH)
    )


def _build_params(metadata: dict[str, Any]) -> dict[str, Any]:
    """Flatten reproducibility settings and XGBoost parameters for MLflow."""

    params: dict[str, Any] = {
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "random_state": metadata["random_state"],
        "test_size": metadata["test_size"],
        "n_train": metadata["n_train"],
        "n_test": metadata["n_test"],
        "n_train_positive": metadata["n_train_positive"],
        "n_train_negative": metadata["n_train_negative"],
        "scale_pos_weight": metadata["scale_pos_weight"],
        "feature_count": len(metadata["features"]),
        "target": metadata["target"],
    }
    params.update(
        {
            f"xgb_{name}": value
            for name, value in metadata["hyperparameters"].items()
        }
    )
    return params


def _build_metrics(metadata: dict[str, Any]) -> dict[str, float]:
    """Flatten every available metric and derive compatible legacy fields."""

    metrics = metadata.get("metrics", {})
    operating_point = metrics.get("operating_point", {})
    flattened: dict[str, float] = {}

    top_level_names = (
        "average_precision",
        "max_achievable_precision",
    )
    operating_names = (
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "alerts",
        "alert_rate_pct",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
    )

    for name in top_level_names:
        value = metrics.get(name)
        if value is not None:
            flattened[name] = float(value)

    for name in operating_names:
        value = operating_point.get(name)
        if value is not None:
            flattened[name] = float(value)

    confusion_names = (
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
    )
    if "accuracy" not in flattened and all(
        name in flattened for name in confusion_names
    ):
        total = sum(flattened[name] for name in confusion_names)
        if total > 0:
            flattened["accuracy"] = (
                flattened["true_positives"]
                + flattened["true_negatives"]
            ) / total

    if (
        "f1_score" not in flattened
        and "precision" in flattened
        and "recall" in flattened
    ):
        precision = flattened["precision"]
        recall = flattened["recall"]
        if precision + recall > 0:
            flattened["f1_score"] = (
                2.0 * precision * recall / (precision + recall)
            )

    if (
        "alerts" not in flattened
        and "true_positives" in flattened
        and "false_positives" in flattened
    ):
        flattened["alerts"] = (
            flattened["true_positives"]
            + flattened["false_positives"]
        )

    if (
        "alert_rate_pct" not in flattened
        and "alerts" in flattened
        and metadata.get("n_test")
    ):
        flattened["alert_rate_pct"] = (
            100.0 * flattened["alerts"] / float(metadata["n_test"])
        )

    if not flattened:
        raise ValueError(
            "No numeric training metrics were found in model metadata."
        )

    return flattened

def _build_tags(metadata: dict[str, Any]) -> dict[str, str]:
    """Build searchable governance and provenance tags."""

    return {
        "project": "AMLGuard",
        "stage": "training",
        "model_name": str(metadata["model_name"]),
        "model_version": str(metadata["model_version"]),
        "git_commit": str(metadata["git_commit"]),
        "baseline_gate": str(metadata["baseline_gate"]["status"]),
    }


def _default_run_name(metadata: dict[str, Any]) -> str:
    """Create a readable run name tied to model version and commit."""

    commit = str(metadata.get("git_commit", "unknown"))
    short_commit = commit[:7] if commit != "unknown" else "unknown"
    return f"amlguard-{metadata['model_version']}-{short_commit}"


def log_training_run(
    metadata: dict[str, Any],
    *,
    tracking_uri: str | None = None,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    run_name: str | None = None,
) -> str:
    """Log one completed training run and return its MLflow run ID."""

    mlflow = _import_mlflow()
    resolved_uri = resolve_tracking_uri(tracking_uri)

    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=run_name or _default_run_name(metadata)
    ) as run:
        run_id = str(run.info.run_id)
        logged_metadata = {**metadata, "mlflow_run_id": run_id}

        mlflow.log_params(_build_params(metadata))
        mlflow.log_metrics(_build_metrics(metadata))
        mlflow.set_tags(_build_tags(metadata))

        mlflow.log_dict(
            logged_metadata,
            "metadata/model_metadata.json",
        )
        mlflow.log_dict(
            metadata["metrics"]["operating_point"],
            "metrics/operating_point.json",
        )

        artifact_sources = (
            (MODEL_PATH, "model"),
            (MODEL_METADATA_PATH, "metadata"),
            (BASELINE_METRICS_PATH, "baseline"),
            (METRICS_JSON_PATH, "evaluation"),
            (PROJECT_ROOT / "src" / "config.py", "configuration"),
        )
        for path, artifact_path in artifact_sources:
            if path.exists():
                mlflow.log_artifact(
                    str(path),
                    artifact_path=artifact_path,
                )

        return run_id


def log_existing_artifacts(
    *,
    tracking_uri: str | None = None,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    run_name: str | None = None,
) -> str:
    """Log the persisted AMLGuard model without retraining it."""

    missing = [
        path
        for path in (MODEL_PATH, MODEL_METADATA_PATH)
        if not path.exists()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Required training artifacts are missing: {missing_text}"
        )

    metadata = json.loads(
        MODEL_METADATA_PATH.read_text(encoding="utf-8")
    )
    run_id = log_training_run(
        metadata,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
    )
    metadata["mlflow_run_id"] = run_id
    MODEL_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return run_id


def main() -> int:
    """Log existing artifacts through the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI; defaults to local SQLite.",
    )
    parser.add_argument(
        "--experiment-name",
        default=MLFLOW_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional human-readable run name.",
    )
    args = parser.parse_args()

    try:
        run_id = log_existing_artifacts(
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("MLflow run logged successfully.")
    print(f"Run ID: {run_id}")
    print(f"Tracking URI: {resolve_tracking_uri(args.tracking_uri)}")
    print("Start the local UI from the repository root with:")
    print("  mlflow server --backend-store-uri "
      "sqlite:///mlflow.db --port 5000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
