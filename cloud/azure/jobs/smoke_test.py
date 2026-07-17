"""Azure ML Day 16 smoke test for data access and project imports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import RAW_REQUIRED_COLUMNS
from src.data.load_data import validate_schema

EXPECTED_BYTES = 475_664_283
EXPECTED_SHA256 = (
    "b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040"
)


def _resolve_csv(input_path: Path) -> Path:
    """Return the CSV represented by an Azure ML file input."""

    if input_path.is_file():
        return input_path

    if input_path.is_dir():
        preferred = input_path / "HI-Small_Trans.csv"
        if preferred.is_file():
            return preferred

        candidates = sorted(input_path.glob("*.csv"))
        if len(candidates) == 1:
            return candidates[0]

    raise FileNotFoundError(
        f"Could not resolve one CSV file from Azure ML input: {input_path}"
    )


def _sha256(path: Path) -> str:
    """Calculate a streaming SHA-256 without loading the dataset into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_smoke_test(data_path: Path, output_path: Path) -> dict[str, Any]:
    """Validate the cloud copy and write machine-readable evidence."""

    csv_path = _resolve_csv(data_path)
    size_bytes = csv_path.stat().st_size
    sha256 = _sha256(csv_path)

    if size_bytes != EXPECTED_BYTES:
        raise RuntimeError(
            f"Dataset size mismatch: expected {EXPECTED_BYTES}, got {size_bytes}."
        )
    if sha256 != EXPECTED_SHA256:
        raise RuntimeError(
            f"Dataset SHA-256 mismatch: expected {EXPECTED_SHA256}, got {sha256}."
        )

    sample = pd.read_csv(csv_path, nrows=1_000)
    validate_schema(sample)

    summary: dict[str, Any] = {
        "status": "PASS",
        "data_asset": "azureml:ibm-aml-hi-small:1",
        "environment": "azureml:amlguard-training:1",
        "file_name": csv_path.name,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "sample_rows_validated": len(sample),
        "required_columns": list(RAW_REQUIRED_COLUMNS),
        "project_import": "PASS",
        "schema_validation": "PASS",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    """Run the Azure ML smoke test from CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    run_smoke_test(args.data_path, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
