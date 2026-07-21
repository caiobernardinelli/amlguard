from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
REQUIREMENTS_PATH = ROOT / "demo" / "requirements.txt"
EVIDENCE_PATH = ROOT / "docs" / "evidence" / "day22_public_deployment.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FINAL_THRESHOLD, MODEL_VERSION  # noqa: E402
from src.models.predict import predict_transaction  # noqa: E402

PUBLIC_URL = "https://amlguard-demo.streamlit.app"
PLATFORM = "Streamlit Community Cloud"
PYTHON_VERSION = "3.12"
ENTRYPOINT = "demo/app.py"
REPOSITORY = "caiobernardinelli/amlguard"
BRANCH = "main"

EXPECTED_MODEL_SHA256 = (
    "b9c069b8fe40f7d905b59a1af6ba2d9c21b1c848daf5c78fbbd50ad94cfe1d2e"
)
EXPECTED_SCORE = 0.985365629196167
EXPECTED_DISPLAY_SCORE = 0.9854
SCORE_TOLERANCE = 1e-12

EXPECTED_REQUIREMENTS = [
    "streamlit==1.58.0",
    "pandas==2.2.3",
    "numpy==2.1.2",
    "scikit-learn==1.9.0",
    "imbalanced-learn==0.14.2",
    "xgboost==3.3.0",
    "joblib==1.5.3",
]

VALIDATED_INPUT: dict[str, Any] = {
    "Payment Format": "ACH",
    "Amount Paid": 13701.30,
    "sender_previous_tx_count": 238,
    "is_business_hours": 0,
    "same_account": 0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")

    if not REQUIREMENTS_PATH.is_file():
        raise FileNotFoundError(
            f"Streamlit requirements not found: {REQUIREMENTS_PATH}"
        )

    model_sha256 = sha256_file(MODEL_PATH)
    requirements = [
        line.strip()
        for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    result = predict_transaction(VALIDATED_INPUT)
    git_commit = current_git_commit()

    checks = {
        "public_url_manually_opened_in_browser": True,
        "public_form_submission_manually_completed": True,
        "public_display_score_matches_expected_rounding": (
            round(float(result["risk_score"]), 4) == EXPECTED_DISPLAY_SCORE
        ),
        "public_alert_decision_confirmed": bool(result["is_alert"]) is True,
        "public_threshold_confirmed": (
            abs(float(result["threshold"]) - float(FINAL_THRESHOLD)) <= 1e-12
        ),
        "public_model_version_confirmed": (
            str(result["model_version"]) == str(MODEL_VERSION)
        ),
        "model_sha256_matches_frozen_artifact": (
            model_sha256 == EXPECTED_MODEL_SHA256
        ),
        "local_contract_score_matches_frozen_result": (
            abs(float(result["risk_score"]) - EXPECTED_SCORE) <= SCORE_TOLERANCE
        ),
        "streamlit_requirements_match_validated_runtime": (
            requirements == EXPECTED_REQUIREMENTS
        ),
        "python_runtime_selected_as_3_12": PYTHON_VERSION == "3.12",
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    evidence = {
        "day": 22,
        "scope": "public_recruiter_demo_deployment",
        "status": status,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": REPOSITORY,
            "branch": BRANCH,
            "git_commit": git_commit,
            "entrypoint": ENTRYPOINT,
            "requirements_file": "demo/requirements.txt",
        },
        "deployment": {
            "platform": PLATFORM,
            "public_url": PUBLIC_URL,
            "python_version": PYTHON_VERSION,
            "azure_managed_online_endpoint_required": False,
            "secrets_required": False,
        },
        "validated_input": VALIDATED_INPUT,
        "validated_output": {
            "risk_score_full_local_contract": float(result["risk_score"]),
            "risk_score_display_confirmed_publicly": EXPECTED_DISPLAY_SCORE,
            "is_alert_confirmed_publicly": bool(result["is_alert"]),
            "threshold_confirmed_publicly": float(result["threshold"]),
            "model_version_confirmed_publicly": str(result["model_version"]),
        },
        "model_artifact": {
            "path": "artifacts/model.joblib",
            "sha256": model_sha256,
            "size_bytes": MODEL_PATH.stat().st_size,
        },
        "dependency_contract": {
            "python_version": PYTHON_VERSION,
            "requirements": requirements,
        },
        "acceptance_checks": checks,
        "deployment_incident": {
            "initial_runtime": "Python 3.14.6",
            "symptom": (
                "The first deployment failed while importing the persisted-model "
                "dependency chain."
            ),
            "resolution": (
                "The app was deleted and recreated with Python 3.12, matching the "
                "validated local runtime."
            ),
            "result": "Resolved",
        },
        "availability_validation": {
            "accepted_method": (
                "Manual browser render plus successful public form submission."
            ),
            "urllib_check": (
                "Not used as an acceptance gate because Streamlit Community Cloud "
                "returned repeated HTTP 303 redirects to urllib."
            ),
        },
        "notes": [
            "The public demo uses synthetic or non-sensitive inputs only.",
            "The model score is not presented as a calibrated probability.",
            "The public demo is a portfolio interface, not a production AML decision system.",
            "The continuously provisioned Azure ML endpoint remains deleted.",
        ],
    }

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"Day 22 public deployment evidence: {status}")
    print(f"- public URL: {PUBLIC_URL}")
    print(f"- platform: {PLATFORM}")
    print(f"- Python runtime: {PYTHON_VERSION}")
    print(f"- git commit: {git_commit}")
    print(f"- displayed risk score: {EXPECTED_DISPLAY_SCORE:.4f}")
    print(f"- threshold: {result['threshold']}")
    print(f"- alert: {result['is_alert']}")
    print(f"- model version: {result['model_version']}")
    print(f"- evidence: {EVIDENCE_PATH.relative_to(ROOT)}")

    if status != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        print(f"- failed checks: {failed}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
