from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEMO_APP_PATH = ROOT / "demo" / "app.py"
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
EVIDENCE_PATH = ROOT / "docs" / "evidence" / "day21_recruiter_demo.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FINAL_THRESHOLD, MODEL_VERSION  # noqa: E402
from src.models.predict import predict_transaction  # noqa: E402

EXPECTED_MODEL_SHA256 = (
    "b9c069b8fe40f7d905b59a1af6ba2d9c21b1c848daf5c78fbbd50ad94cfe1d2e"
)
EXPECTED_SCORE = 0.985365629196167
SCORE_TOLERANCE = 1e-12

DEFAULT_FORM_VALUES: dict[str, Any] = {
    "payment_format": "ACH",
    "amount_paid": 13701.30,
    "sender_previous_tx_count": 238,
    "is_business_hours": False,
    "same_account": False,
}

EXPECTED_FEATURES: dict[str, Any] = {
    "Payment Format": "ACH",
    "Amount Paid": 13701.30,
    "sender_previous_tx_count": 238,
    "is_business_hours": 0,
    "same_account": 0,
}


def load_demo_module():
    """Load demo/app.py directly by file path."""
    spec = importlib.util.spec_from_file_location("amlguard_demo_app", DEMO_APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load demo module from {DEMO_APP_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if not DEMO_APP_PATH.is_file():
        raise FileNotFoundError(f"Demo app not found: {DEMO_APP_PATH}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")

    demo_app = load_demo_module()
    build_model_features = demo_app.build_model_features

    model_sha256 = sha256_file(MODEL_PATH)
    features = build_model_features(**DEFAULT_FORM_VALUES)
    result = predict_transaction(features)

    checks = {
        "demo_app_exists": True,
        "model_artifact_exists": True,
        "model_sha256_matches_frozen_demo_artifact": (
            model_sha256 == EXPECTED_MODEL_SHA256
        ),
        "form_mapping_matches_model_contract": features == EXPECTED_FEATURES,
        "risk_score_matches_validated_local_inference": (
            abs(float(result["risk_score"]) - EXPECTED_SCORE) <= SCORE_TOLERANCE
        ),
        "alert_decision_is_true": bool(result["is_alert"]) is True,
        "threshold_matches_frozen_operating_point": (
            abs(float(result["threshold"]) - float(FINAL_THRESHOLD)) <= 1e-12
        ),
        "model_version_matches_project_config": (
            str(result["model_version"]) == str(MODEL_VERSION)
        ),
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    evidence = {
        "day": 21,
        "scope": "recruiter_demo_local_validation",
        "status": status,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "demo_entrypoint": "demo/app.py",
        "model_artifact": {
            "path": "artifacts/model.joblib",
            "sha256": model_sha256,
            "size_bytes": MODEL_PATH.stat().st_size,
        },
        "validated_input": EXPECTED_FEATURES,
        "validated_output": {
            "risk_score": float(result["risk_score"]),
            "is_alert": bool(result["is_alert"]),
            "threshold": float(result["threshold"]),
            "model_version": str(result["model_version"]),
        },
        "acceptance_checks": checks,
        "notes": [
            "The Streamlit interface uses the same persisted AMLGuard prediction contract.",
            "The displayed score is rounded visually; the evidence stores the full model output.",
            "The demo is a portfolio interface and must use synthetic or non-sensitive inputs.",
            "Public hosting is not part of this Day 21 local validation.",
        ],
    }

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"Day 21 recruiter demo validation: {status}")
    print(f"- model SHA256: {model_sha256}")
    print(f"- risk score: {result['risk_score']}")
    print(f"- alert: {result['is_alert']}")
    print(f"- threshold: {result['threshold']}")
    print(f"- model version: {result['model_version']}")
    print(f"- evidence: {EVIDENCE_PATH.relative_to(ROOT)}")

    if status != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        print(f"- failed checks: {failed}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
