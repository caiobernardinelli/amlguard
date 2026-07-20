"""Validate AMLGuard Day 20 collected input/output pairing.

Run from the AMLGuard repository root:

    python validate_day20_data_collection.py

The script:
- parses collected model_inputs/model_outputs JSONL;
- validates expected schemas;
- verifies unique correlation IDs;
- pairs inputs and outputs by correlation ID;
- writes sanitized evidence with counts only.

It does not print or persist transaction feature values or prediction values.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
COLLECTED_DIR = ROOT / ".day20_collected"
INPUTS_PATH = COLLECTED_DIR / "model_inputs.jsonl"
OUTPUTS_PATH = COLLECTED_DIR / "model_outputs.jsonl"
EVIDENCE_PATH = ROOT / "docs" / "evidence" / "day20_data_collection_validation.json"

EXPECTED_INPUT_FIELDS = {
    "Payment Format",
    "Amount Paid",
    "sender_previous_tx_count",
    "is_business_hours",
    "same_account",
}

EXPECTED_OUTPUT_FIELDS = {
    "score",
    "alert",
    "threshold",
    "model_version",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path.name} at line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path.name} line {line_number} is not a JSON object."
                )

            records.append(record)

    return records


def validate_records(
    records: list[dict[str, Any]],
    expected_fields: set[str],
    label: str,
) -> dict[str, Any]:
    """Validate collector envelopes and one-row tabular payloads."""
    correlation_ids: list[str] = []
    schema_mismatches = 0
    invalid_data_shapes = 0
    missing_correlation_ids = 0

    for record in records:
        correlation_id = record.get("correlationid")
        if not isinstance(correlation_id, str) or not correlation_id:
            missing_correlation_ids += 1
        else:
            correlation_ids.append(correlation_id)

        data = record.get("data")
        if not isinstance(data, list) or len(data) != 1:
            invalid_data_shapes += 1
            continue

        row = data[0]
        if not isinstance(row, dict):
            invalid_data_shapes += 1
            continue

        if set(row) != expected_fields:
            schema_mismatches += 1

    counts = Counter(correlation_ids)
    duplicate_ids = sorted(
        correlation_id
        for correlation_id, count in counts.items()
        if count > 1
    )

    return {
        "label": label,
        "records": len(records),
        "valid_correlation_ids": len(correlation_ids),
        "missing_correlation_ids": missing_correlation_ids,
        "duplicate_correlation_id_count": len(duplicate_ids),
        "invalid_data_shapes": invalid_data_shapes,
        "schema_mismatches": schema_mismatches,
        "_correlation_ids": set(correlation_ids),
    }


def main() -> int:
    """Validate pairing and persist sanitized Day 20 evidence."""
    inputs = load_jsonl(INPUTS_PATH)
    outputs = load_jsonl(OUTPUTS_PATH)

    input_report = validate_records(
        inputs,
        EXPECTED_INPUT_FIELDS,
        "model_inputs",
    )
    output_report = validate_records(
        outputs,
        EXPECTED_OUTPUT_FIELDS,
        "model_outputs",
    )

    input_ids = input_report.pop("_correlation_ids")
    output_ids = output_report.pop("_correlation_ids")

    matched_ids = input_ids & output_ids
    input_only_ids = input_ids - output_ids
    output_only_ids = output_ids - input_ids

    checks = {
        "input_records_present": input_report["records"] > 0,
        "output_records_present": output_report["records"] > 0,
        "input_output_record_counts_match": (
            input_report["records"] == output_report["records"]
        ),
        "input_correlation_ids_unique": (
            input_report["duplicate_correlation_id_count"] == 0
        ),
        "output_correlation_ids_unique": (
            output_report["duplicate_correlation_id_count"] == 0
        ),
        "all_inputs_have_correlation_id": (
            input_report["missing_correlation_ids"] == 0
        ),
        "all_outputs_have_correlation_id": (
            output_report["missing_correlation_ids"] == 0
        ),
        "input_schema_valid": (
            input_report["invalid_data_shapes"] == 0
            and input_report["schema_mismatches"] == 0
        ),
        "output_schema_valid": (
            output_report["invalid_data_shapes"] == 0
            and output_report["schema_mismatches"] == 0
        ),
        "all_input_output_pairs_matched": (
            not input_only_ids and not output_only_ids
        ),
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    evidence = {
        "day": 20,
        "scope": "azure_ml_model_data_collection_validation",
        "status": status,
        "inputs": input_report,
        "outputs": output_report,
        "pairing": {
            "matched_pairs": len(matched_ids),
            "input_only_count": len(input_only_ids),
            "output_only_count": len(output_only_ids),
        },
        "acceptance_checks": checks,
        "privacy": (
            "Evidence contains structural counts only. Collected transaction "
            "feature values, prediction scores, request IDs, correlation IDs, "
            "Azure resource IDs, and credentials are intentionally excluded."
        ),
    }

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("AMLGuard Day 20 - Data Collection Validation")
    print("=" * 52)
    print(f"model_inputs records: {input_report['records']}")
    print(f"model_outputs records: {output_report['records']}")
    print(f"matched input/output pairs: {len(matched_ids)}")
    print(f"input-only records: {len(input_only_ids)}")
    print(f"output-only records: {len(output_only_ids)}")
    print(
        "input schema: "
        f"{'PASS' if checks['input_schema_valid'] else 'FAIL'}"
    )
    print(
        "output schema: "
        f"{'PASS' if checks['output_schema_valid'] else 'FAIL'}"
    )
    print(
        "correlation pairing: "
        f"{'PASS' if checks['all_input_output_pairs_matched'] else 'FAIL'}"
    )
    print(f"Evidence written: {EVIDENCE_PATH.relative_to(ROOT)}")
    print(f"Day 20 data collection validation: {status}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
