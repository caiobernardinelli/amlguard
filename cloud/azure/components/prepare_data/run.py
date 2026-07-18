"""Prepare AMLGuard data for Azure ML component execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_FEATURES, RANDOM_STATE, TARGET, TEST_SIZE
from src.data.load_data import validate_schema
from src.models.train import prepare_train_test_split


def resolve_csv(input_path: Path) -> Path:
    """Resolve a single CSV from an Azure ML uri_file input."""

    if input_path.is_file():
        return input_path

    if input_path.is_dir():
        preferred = input_path / "HI-Small_Trans.csv"
        if preferred.is_file():
            return preferred

        candidates = sorted(input_path.glob("*.csv"))
        if len(candidates) == 1:
            return candidates[0]

    raise FileNotFoundError(f"Could not resolve one CSV from: {input_path}")


def main() -> int:
    """Validate raw data, build features, split, and persist the prepared bundle."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    args = parser.parse_args()

    csv_path = resolve_csv(args.raw_data)
    raw = pd.read_csv(csv_path)
    validate_schema(raw)

    X_train, X_test, y_train, y_test = prepare_train_test_split(raw)

    args.prepared_data.mkdir(parents=True, exist_ok=True)
    bundle_path = args.prepared_data / "prepared_split.joblib"
    manifest_path = args.prepared_data / "manifest.json"

    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        },
        bundle_path,
        compress=3,
    )

    manifest = {
        "status": "PASS",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "features": list(MODEL_FEATURES),
        "target": TARGET,
        "n_total": int(len(raw)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_train_positive": int(y_train.sum()),
        "n_test_positive": int(y_test.sum()),
        "bundle_file": bundle_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
