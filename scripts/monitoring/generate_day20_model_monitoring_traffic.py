"""Generate deterministic synthetic monitoring traffic for AMLGuard Day 20.

Run from the AMLGuard repository root:

    python generate_day20_model_monitoring_traffic.py

The script sends 70 varied synthetic inference requests to the deployed Azure ML
endpoint and writes a sanitized local evidence summary.

It does not read or persist endpoint credentials and does not modify Azure
resources. The requests are synthetic monitoring traffic, not production data.
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = ROOT / "docs" / "evidence" / "day20_model_monitoring_traffic.json"

ENDPOINT_NAME = "amlguard-realtime"

PAYMENT_FORMATS = (
    "Cheque",
    "Credit Card",
    "ACH",
    "Cash",
    "Reinvestment",
    "Wire",
    "Bitcoin",
)

AMOUNTS = (
    25.0,
    100.0,
    350.0,
    1000.0,
    3000.0,
    7500.0,
    15000.0,
    50000.0,
    125000.0,
    500000.0,
)

PREVIOUS_COUNTS = (
    0,
    1,
    2,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
)


def find_azure_cli() -> str:
    """Locate the Azure CLI executable."""
    for candidate in ("az.cmd", "az.exe", "az"):
        executable = shutil.which(candidate)
        if executable:
            return executable

    raise FileNotFoundError(
        "Azure CLI was not found in PATH. Confirm that `az --version` works."
    )


def percentile(values: list[float], probability: float) -> float:
    """Compute a linear-interpolated percentile without third-party packages."""
    if not values:
        raise ValueError("Cannot compute percentile of an empty list.")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def parse_invoke_output(raw_output: str) -> dict[str, Any]:
    """Parse Azure CLI invoke output, including JSON encoded as a JSON string."""
    parsed: Any = json.loads(raw_output)

    if isinstance(parsed, str):
        parsed = json.loads(parsed)

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Unexpected Azure ML response type: {type(parsed).__name__}"
        )

    return parsed


def build_requests() -> list[dict[str, Any]]:
    """Create a deterministic 70-record synthetic monitoring batch."""
    requests: list[dict[str, Any]] = []

    for format_index, payment_format in enumerate(PAYMENT_FORMATS):
        for sample_index, (amount, previous_count) in enumerate(
            zip(AMOUNTS, PREVIOUS_COUNTS, strict=True)
        ):
            requests.append(
                {
                    "data": {
                        "Payment Format": payment_format,
                        "Amount Paid": amount,
                        "sender_previous_tx_count": previous_count,
                        "is_business_hours": (sample_index + format_index) % 2,
                        "same_account": 1
                        if (sample_index + format_index) % 3 == 0
                        else 0,
                    }
                }
            )

    return requests


def invoke(
    executable: str,
    request_file: Path,
) -> tuple[dict[str, Any], float]:
    """Invoke the endpoint and return parsed response plus client latency."""
    started = time.perf_counter()

    completed = subprocess.run(
        [
            executable,
            "ml",
            "online-endpoint",
            "invoke",
            "--name",
            ENDPOINT_NAME,
            "--request-file",
            str(request_file),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    latency_ms = (time.perf_counter() - started) * 1000.0

    if completed.returncode != 0:
        diagnostic = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or "Azure CLI returned no diagnostic output."
        )
        raise RuntimeError(
            f"Endpoint invocation failed with exit code "
            f"{completed.returncode}: {diagnostic}"
        )

    return parse_invoke_output(completed.stdout.strip()), latency_ms


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build sanitized model-behavior and client-latency summaries."""
    scores = [float(item["response"]["score"]) for item in results]
    latencies = [float(item["client_latency_ms"]) for item in results]
    alert_count = sum(bool(item["response"]["alert"]) for item in results)

    by_format: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        by_format[item["request"]["data"]["Payment Format"]].append(item)

    format_summary: dict[str, Any] = {}

    for payment_format, items in sorted(by_format.items()):
        format_scores = [
            float(item["response"]["score"])
            for item in items
        ]
        format_alerts = sum(
            bool(item["response"]["alert"])
            for item in items
        )

        format_summary[payment_format] = {
            "requests": len(items),
            "alerts": format_alerts,
            "alert_rate": format_alerts / len(items),
            "score_mean": statistics.fmean(format_scores),
            "score_min": min(format_scores),
            "score_max": max(format_scores),
        }

    return {
        "request_count": len(results),
        "alert_count": alert_count,
        "alert_rate": alert_count / len(results),
        "score_distribution": {
            "min": min(scores),
            "p05": percentile(scores, 0.05),
            "p25": percentile(scores, 0.25),
            "median": statistics.median(scores),
            "p75": percentile(scores, 0.75),
            "p95": percentile(scores, 0.95),
            "max": max(scores),
            "mean": statistics.fmean(scores),
        },
        "client_latency_ms": {
            "min": min(latencies),
            "median": statistics.median(latencies),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies),
            "mean": statistics.fmean(latencies),
            "note": (
                "Client-observed latency includes Azure CLI process overhead "
                "and network time; it is not the Azure Monitor server-side "
                "RequestLatency metric."
            ),
        },
        "by_payment_format": format_summary,
    }


def main() -> int:
    """Send synthetic monitoring traffic and persist sanitized evidence."""
    executable = find_azure_cli()
    requests = build_requests()

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    print(
        f"Sending {len(requests)} synthetic monitoring requests "
        f"to {ENDPOINT_NAME}..."
    )

    with tempfile.TemporaryDirectory(prefix="amlguard_day20_") as temp_dir:
        request_file = Path(temp_dir) / "request.json"

        for index, payload in enumerate(requests, start=1):
            request_file.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

            try:
                response, latency_ms = invoke(executable, request_file)

                required_fields = {
                    "score",
                    "alert",
                    "threshold",
                    "model_version",
                }
                missing = required_fields - set(response)
                if missing:
                    raise ValueError(
                        f"Response missing field(s): {sorted(missing)}"
                    )

                results.append(
                    {
                        "request_index": index,
                        "request": payload,
                        "response": response,
                        "client_latency_ms": latency_ms,
                    }
                )

                print(
                    f"[{index:02d}/{len(requests)}] PASS | "
                    f"{payload['data']['Payment Format']:<12} | "
                    f"score={float(response['score']):.6f} | "
                    f"alert={bool(response['alert'])}"
                )

            except Exception as error:
                failures.append(
                    {
                        "request_index": index,
                        "error": str(error),
                    }
                )
                print(
                    f"[{index:02d}/{len(requests)}] FAIL | {error}"
                )

            time.sleep(0.25)

    status = (
        "PASS"
        if len(results) == len(requests) and not failures
        else "FAIL"
    )

    evidence: dict[str, Any] = {
        "day": 20,
        "scope": "model_observability_traffic",
        "status": status,
        "traffic_type": "synthetic_monitoring_traffic",
        "endpoint": ENDPOINT_NAME,
        "payment_formats": list(PAYMENT_FORMATS),
        "requested_inferences": len(requests),
        "successful_inferences": len(results),
        "failed_inferences": len(failures),
        "failures": failures,
        "notes": [
            (
                "This batch is deterministic synthetic traffic created only "
                "to exercise model observability and Azure ML data collection."
            ),
            (
                "It is not representative production traffic and must not be "
                "used to claim production model performance."
            ),
        ],
    }

    if results:
        evidence["summary"] = summarize(results)

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Evidence written: {EVIDENCE_PATH.relative_to(ROOT)}")
    print(f"Day 20 synthetic monitoring traffic: {status}")
    print(f"- requested: {len(requests)}")
    print(f"- succeeded: {len(results)}")
    print(f"- failed: {len(failures)}")

    if results:
        summary = evidence["summary"]
        print(
            f"- alert rate: {summary['alert_rate']:.2%} "
            f"({summary['alert_count']}/{summary['request_count']})"
        )
        print(
            "- score range: "
            f"{summary['score_distribution']['min']:.6f} to "
            f"{summary['score_distribution']['max']:.6f}"
        )

    print("- Azure credentials and resource IDs were not persisted")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
