"""Train the frozen AMLGuard XGBoost pipeline from prepared data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib

from src.config import FINAL_MODEL_NAME, MODEL_FEATURES, MODEL_VERSION, RANDOM_STATE, XGBOOST_PARAMS
from src.models.train import build_pipeline


def main() -> int:
    """Train and persist the weighted XGBoost pipeline."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()

    bundle_path = args.prepared_data / "prepared_split.joblib"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Prepared split not found: {bundle_path}")

    bundle = joblib.load(bundle_path)
    X_train = bundle["X_train"]
    y_train = bundle["y_train"]

    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    if positive == 0:
        raise RuntimeError("Training split contains no positive examples.")

    scale_pos_weight = negative / positive
    pipeline = build_pipeline(scale_pos_weight)

    print(
        f"Training {FINAL_MODEL_NAME} on {len(X_train):,} rows "
        f"with scale_pos_weight={scale_pos_weight:.6f}"
    )
    pipeline.fit(X_train, y_train)

    args.model_output.mkdir(parents=True, exist_ok=True)
    model_path = args.model_output / "model.joblib"
    metadata_path = args.model_output / "training_metadata.json"

    joblib.dump(pipeline, model_path, compress=3)

    metadata = {
        "status": "PASS",
        "model_name": FINAL_MODEL_NAME,
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "n_train": int(len(X_train)),
        "n_train_positive": positive,
        "n_train_negative": negative,
        "scale_pos_weight": scale_pos_weight,
        "features": list(MODEL_FEATURES),
        "hyperparameters": XGBOOST_PARAMS,
        "library_versions": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "joblib": version("joblib"),
            "scikit-learn": version("scikit-learn"),
            "xgboost": version("xgboost"),
        },
        "model_file": model_path.name,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
