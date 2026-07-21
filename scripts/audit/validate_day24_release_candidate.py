from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = ROOT / "docs" / "evidence" / "day24_release_candidate.json"

RELEASE_VERSION = "1.0.0"
RELEASE_TAG = "v1.0.0"
MODEL_VERSION = "0.1.0"
AZURE_MODEL_ASSET = "AMLGuard:1"
PUBLIC_URL = "https://amlguard-demo.streamlit.app"
EXPECTED_MODEL_SHA256 = (
    "b9c069b8fe40f7d905b59a1af6ba2d9c21b1c848daf5c78fbbd50ad94cfe1d2e"
)

MANUALS = [
    "docs/manuals/AMLGuard_Manual_Estudo_Vol1_Notebook_ao_Pipeline_MLOps.pdf",
    "docs/manuals/AMLGuard_Manual_Estudo_Vol2_FastAPI_Pydantic_Docker.pdf",
    "docs/manuals/AMLGuard_Manual_Estudo_Vol3_CI_CD_GitHub_Actions_GHCR.pdf",
    "docs/manuals/AMLGuard_Manual_Estudo_Vol4_MLflow_Tracking_Model_Registry.pdf",
    "docs/manuals/AMLGuard_Manual_Estudo_Vol5_Azure_ML_Cloud_Deploy.pdf",
    "docs/manuals/AMLGuard_Manual_Estudo_Vol6_Observability_Drift_Retraining.pdf",
    "docs/manuals/AMLGuard_Manual_Estudo_Vol7_Demo_Portfolio_Release.pdf",
]

PRIOR_EVIDENCE = [
    "docs/evidence/day18_evaluation_metrics.json",
    "docs/evidence/day19_online_endpoint.json",
    "docs/evidence/day20_data_collection_validation.json",
    "docs/evidence/day20_drift_report.json",
    "docs/evidence/day20_retraining_strategy.json",
    "docs/evidence/day21_recruiter_demo.json",
    "docs/evidence/day22_public_deployment.json",
    "docs/evidence/day23_portfolio_readiness.json",
]

EXPECTED_WORKTREE_PATHS = {
    ".gitattributes",
    "README.md",
    "pyproject.toml",
    "CHANGELOG.md",
    "docs/FINAL_RELEASE.md",
    "docs/RELEASE_NOTES_v1.0.0.md",
    "docs/evidence/day24_release_candidate.json",
    "docs/manuals/README.md",
    "scripts/audit/validate_day24_release_candidate.py",
    *MANUALS,
}

MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
PASSED_RE = re.compile(r"(?P<count>\d+)\s+passed")


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part.strip()
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "output": output,
    }


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    # Preserve the leading status column from the first line of `git status`.
    # Only line-ending characters are removed.
    return completed.stdout.rstrip("\r\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pyproject_version() -> str | None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(
        r'(?ms)^\[project\].*?^version\s*=\s*"([^"]+)"',
        text,
    )
    return match.group(1) if match else None


def config_model_version() -> str | None:
    text = (ROOT / "src" / "config.py").read_text(encoding="utf-8")
    match = re.search(
        r'^MODEL_VERSION(?:\s*:\s*[^=]+)?\s*=\s*"([^"]+)"',
        text,
        re.MULTILINE,
    )
    return match.group(1) if match else None


def parse_status_path(line: str) -> str:
    if len(line) < 4:
        return ""
    raw = line[3:].strip()
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.replace("\\", "/")


def unexpected_worktree_entries() -> list[str]:
    entries = git_output("status", "--short", "--untracked-files=all").splitlines()
    unexpected: list[str] = []

    for entry in entries:
        path = parse_status_path(entry)
        if path and path not in EXPECTED_WORKTREE_PATHS:
            unexpected.append(entry)

    return unexpected


def parse_local_target(raw_target: str) -> str | None:
    target = raw_target.strip()

    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]

    if ' "' in target:
        target = target.split(' "', 1)[0]
    elif " '" in target:
        target = target.split(" '", 1)[0]

    if not target or target.startswith("#"):
        return None

    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return None

    clean = unquote(parsed.path)
    return clean or None


def markdown_link_failures() -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    markdown_files = [
        ROOT / "README.md",
        ROOT / "CHANGELOG.md",
        *sorted((ROOT / "docs").rglob("*.md")),
    ]

    for markdown_file in markdown_files:
        if not markdown_file.is_file():
            continue

        text = markdown_file.read_text(encoding="utf-8", errors="replace")
        for match in MARKDOWN_LINK_RE.finditer(text):
            raw_target = match.group(1)
            local_target = parse_local_target(raw_target)
            if local_target is None:
                continue

            resolved = (markdown_file.parent / local_target).resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                failures.append(
                    {
                        "source": markdown_file.relative_to(ROOT).as_posix(),
                        "target": raw_target,
                        "reason": "link escapes repository root",
                    }
                )
                continue

            if not resolved.exists():
                failures.append(
                    {
                        "source": markdown_file.relative_to(ROOT).as_posix(),
                        "target": raw_target,
                        "reason": "target does not exist",
                    }
                )

    return failures


def tracked_hygiene_failures() -> list[str]:
    tracked = git_output("ls-files").splitlines()
    failures: list[str] = []

    for path in tracked:
        normalized = path.replace("\\", "/")
        name = Path(normalized).name.lower()
        suffix = Path(normalized).suffix.lower()

        if normalized.startswith("data/raw/") and name != ".gitkeep":
            failures.append(f"{normalized}: raw dataset is tracked")
        elif name == ".env":
            failures.append(f"{normalized}: secrets file is tracked")
        elif name.startswith(".env.") and name != ".env.example":
            failures.append(f"{normalized}: environment-specific secrets file is tracked")
        elif name == "mlflow.db":
            failures.append(f"{normalized}: local MLflow database is tracked")
        elif suffix in {".pem", ".key", ".pfx", ".p12"}:
            failures.append(f"{normalized}: credential-like file is tracked")
        elif suffix == ".jsonl" and (
            "model_inputs" in normalized.lower()
            or "model_outputs" in normalized.lower()
            or "collector" in normalized.lower()
        ):
            failures.append(f"{normalized}: raw monitoring collection is tracked")

    return failures


def evidence_status(path: str) -> dict[str, Any]:
    target = ROOT / path
    if not target.is_file():
        return {
            "path": path,
            "exists": False,
            "status": None,
            "passed": False,
        }

    payload = read_json(target)
    status = payload.get("status")
    return {
        "path": path,
        "exists": True,
        "status": status,
        "passed": status == "PASS",
    }


def manuals_index_checks() -> dict[str, bool]:
    index_path = ROOT / "docs" / "manuals" / "README.md"
    text = index_path.read_text(encoding="utf-8") if index_path.is_file() else ""
    return {
        manual: Path(manual).name in text
        for manual in MANUALS
    }


def stale_volume_references() -> list[dict[str, str]]:
    checked_files = [
        ROOT / "README.md",
        ROOT / "CHANGELOG.md",
        ROOT / "docs" / "FINAL_RELEASE.md",
        ROOT / "docs" / "RELEASE_NOTES_v1.0.0.md",
        ROOT / "docs" / "manuals" / "README.md",
    ]
    stale_tokens = [
        "Volume 8",
        "Vol8",
        "AMLGuard_Manual_Estudo_Vol8",
    ]
    failures: list[dict[str, str]] = []

    for path in checked_files:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in stale_tokens:
            if token in text:
                failures.append(
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "token": token,
                    }
                )

    return failures


def main() -> int:
    required_files = [
        "README.md",
        "CHANGELOG.md",
        "pyproject.toml",
        "artifacts/model.joblib",
        "artifacts/baseline_metrics.json",
        "docs/FINAL_RELEASE.md",
        "docs/PORTFOLIO_READINESS.md",
        "docs/RECRUITER_DEMO.md",
        "docs/RELEASE_NOTES_v1.0.0.md",
        "docs/evidence/day23_portfolio_readiness.json",
        "docs/manuals/README.md",
        "scripts/audit/validate_day23_portfolio_readiness.py",
        "scripts/audit/validate_day24_release_candidate.py",
        *MANUALS,
    ]
    missing_files = [path for path in required_files if not (ROOT / path).is_file()]

    prior_evidence = [evidence_status(path) for path in PRIOR_EVIDENCE]

    day23_path = ROOT / "docs" / "evidence" / "day23_portfolio_readiness.json"
    day23_payload = read_json(day23_path) if day23_path.is_file() else {}
    day23_days = day23_payload.get("days", [])
    day23_contract = {
        "status_pass": day23_payload.get("status") == "PASS",
        "contains_22_days": isinstance(day23_days, list) and len(day23_days) == 22,
        "all_days_pass": (
            isinstance(day23_days, list)
            and len(day23_days) == 22
            and all(day.get("status") == "PASS" for day in day23_days)
        ),
    }

    model_path = ROOT / "artifacts" / "model.joblib"
    actual_model_sha256 = sha256(model_path) if model_path.is_file() else None

    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    release_doc_text = (ROOT / "docs" / "FINAL_RELEASE.md").read_text(
        encoding="utf-8"
    )
    release_notes_text = (ROOT / "docs" / "RELEASE_NOTES_v1.0.0.md").read_text(
        encoding="utf-8"
    )

    manual_index = manuals_index_checks()
    stale_references = stale_volume_references()

    documentation_checks = {
        "readme_public_url": PUBLIC_URL in readme_text,
        "readme_release_version": RELEASE_TAG in readme_text,
        "readme_24_day_delivery": "24-day" in readme_text,
        "readme_final_release_link": "./docs/FINAL_RELEASE.md" in readme_text,
        "release_doc_version": RELEASE_TAG in release_doc_text,
        "release_doc_model_version": MODEL_VERSION in release_doc_text,
        "release_doc_azure_asset": AZURE_MODEL_ASSET in release_doc_text,
        "release_doc_manual_index": "./manuals/README.md" in release_doc_text,
        "release_notes_public_url": PUBLIC_URL in release_notes_text,
        "manual_index_complete": all(manual_index.values()),
        "no_stale_volume_8_references": not stale_references,
    }

    tag_present = bool(git_output("tag", "--list", RELEASE_TAG))
    tag_available = not tag_present

    lint = run_command(
        [sys.executable, "-m", "ruff", "check", "src", "tests", "demo", "scripts"]
    )
    tests = run_command(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "-q"]
    )
    test_match = PASSED_RE.search(tests["output"])
    test_count = int(test_match.group("count")) if test_match else None
    tests["test_count"] = test_count
    tests["expected_test_count"] = 56
    tests["passed"] = tests["passed"] and test_count == 56

    diff_check = run_command(["git", "diff", "--check"])
    link_failures = markdown_link_failures()
    hygiene_failures = tracked_hygiene_failures()
    unexpected_entries = unexpected_worktree_entries()

    checks = {
        "branch_is_main": git_output("branch", "--show-current") == "main",
        "required_files_present": not missing_files,
        "package_version_is_1_0_0": pyproject_version() == RELEASE_VERSION,
        "model_version_remains_0_1_0": config_model_version() == MODEL_VERSION,
        "model_sha256_unchanged": actual_model_sha256 == EXPECTED_MODEL_SHA256,
        "prior_evidence_pass": all(item["passed"] for item in prior_evidence),
        "day23_contract_pass": all(day23_contract.values()),
        "documentation_contract_pass": all(documentation_checks.values()),
        "seven_manuals_present": all((ROOT / path).is_file() for path in MANUALS),
        "release_tag_available": tag_available,
        "ruff_pass": lint["passed"],
        "pytest_56_pass": tests["passed"],
        "git_diff_check_pass": diff_check["passed"],
        "local_markdown_links_pass": not link_failures,
        "tracked_hygiene_pass": not hygiene_failures,
        "no_unexpected_worktree_entries": not unexpected_entries,
    }
    overall_status = "PASS" if all(checks.values()) else "FAIL"

    evidence = {
        "day": 24,
        "scope": "final_release_candidate",
        "status": overall_status,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "release": {
            "version": RELEASE_VERSION,
            "planned_tag": RELEASE_TAG,
            "tag_available_before_release": tag_available,
        },
        "version_domains": {
            "project_release": RELEASE_TAG,
            "python_package": pyproject_version(),
            "frozen_model_contract": config_model_version(),
            "azure_ml_model_asset": AZURE_MODEL_ASSET,
        },
        "repository": {
            "branch": git_output("branch", "--show-current"),
            "audited_commit": git_output("rev-parse", "HEAD"),
            "unexpected_worktree_entries": unexpected_entries,
        },
        "artifact": {
            "path": "artifacts/model.joblib",
            "sha256": actual_model_sha256,
            "expected_sha256": EXPECTED_MODEL_SHA256,
        },
        "manuals": {
            "count": len(MANUALS),
            "paths": MANUALS,
            "index_checks": manual_index,
            "stale_volume_references": stale_references,
        },
        "prior_evidence": prior_evidence,
        "day23_contract": day23_contract,
        "documentation_checks": documentation_checks,
        "quality_commands": {
            "ruff": lint,
            "pytest": tests,
            "git_diff_check": diff_check,
        },
        "repository_checks": {
            "missing_required_files": missing_files,
            "broken_local_markdown_links": link_failures,
            "tracked_hygiene_failures": hygiene_failures,
        },
        "checks": checks,
        "limitations": [
            "The IBM AML dataset is synthetic.",
            "The model score is not a calibrated probability.",
            "Predictions require human review.",
            "Day 20 monitoring traffic was synthetic.",
            "The Azure Managed Online Endpoint remains deleted for cost control.",
            "The public Streamlit demo is not covered by a production SLA.",
            (
                "Release-candidate evidence is generated before the final commit "
                "and annotated tag are created."
            ),
        ],
    }

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("AMLGuard Day 24 final release candidate")
    print(f"- package version: {pyproject_version()}")
    print(f"- frozen model version: {config_model_version()}")
    print(f"- model SHA-256 unchanged: {checks['model_sha256_unchanged']}")
    print(f"- prior evidence passed: {checks['prior_evidence_pass']}")
    print(f"- manuals present: {sum((ROOT / path).is_file() for path in MANUALS)}/7")
    print(f"- manual index complete: {documentation_checks['manual_index_complete']}")
    print(f"- stale Volume 8 references: {len(stale_references)}")
    print(f"- Ruff passed: {checks['ruff_pass']}")
    print(f"- pytest passed: {tests['test_count']}/56")
    print(f"- local Markdown links passed: {checks['local_markdown_links_pass']}")
    print(f"- tracked hygiene passed: {checks['tracked_hygiene_pass']}")
    print(f"- tag {RELEASE_TAG} available: {checks['release_tag_available']}")
    print(f"- unexpected worktree entries: {len(unexpected_entries)}")
    print(f"- status: {overall_status}")
    print(f"- evidence: {EVIDENCE_PATH.relative_to(ROOT)}")

    if overall_status != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        print(f"- failed checks: {failed}")

    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
