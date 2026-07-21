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
EVIDENCE_PATH = ROOT / "docs" / "evidence" / "v1_0_1_hotfix.json"

RELEASE_VERSION = "1.0.1"
RELEASE_TAG = "v1.0.1"
MODEL_VERSION = "0.1.0"
EXPECTED_MODEL_SHA256 = "b9c069b8fe40f7d905b59a1af6ba2d9c21b1c848daf5c78fbbd50ad94cfe1d2e"
EXPECTED_TEST_COUNT = 60

EXPECTED_WORKTREE_PATHS = {
    "README.md",
    "pyproject.toml",
    "CHANGELOG.md",
    "src/api/main.py",
    "tests/test_api.py",
    "tests/test_release_integrity.py",
    "docs/RELEASE_NOTES_v1.0.1.md",
    "docs/manuals/README.md",
    "docs/evidence/v1_0_1_hotfix.json",
    "scripts/audit/validate_v1_0_1_hotfix.py",
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
    return completed.stdout.rstrip("\r\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_version(path: Path, name: str) -> str | None:
    text = path.read_text(encoding="utf-8")
    if name == "project":
        pattern = r'(?ms)^\[project\].*?^version\s*=\s*"([^"]+)"'
    else:
        pattern = rf'^{re.escape(name)}(?:\s*:\s*[^=]+)?\s*=\s*"([^"]+)"'
    match = re.search(pattern, text, re.MULTILINE)
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
    return [
        entry
        for entry in entries
        if (path := parse_status_path(entry))
        and path not in EXPECTED_WORKTREE_PATHS
    ]


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
    return unquote(parsed.path) or None


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
    failures: list[str] = []
    for path in git_output("ls-files").splitlines():
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

    return failures


def main() -> int:
    required_files = [
        "README.md",
        "CHANGELOG.md",
        "pyproject.toml",
        "src/api/main.py",
        "src/config.py",
        "tests/test_api.py",
        "tests/test_release_integrity.py",
        "artifacts/model.joblib",
        "artifacts/baseline_metrics.json",
        "docs/RELEASE_NOTES_v1.0.1.md",
        "docs/manuals/README.md",
        "docs/evidence/day24_release_candidate.json",
        "scripts/audit/validate_v1_0_1_hotfix.py",
    ]
    missing_files = [path for path in required_files if not (ROOT / path).is_file()]

    package_version = extract_version(ROOT / "pyproject.toml", "project")
    service_version = extract_version(ROOT / "src" / "api" / "main.py", "SERVICE_VERSION")
    model_version = extract_version(ROOT / "src" / "config.py", "MODEL_VERSION")

    model_path = ROOT / "artifacts" / "model.joblib"
    actual_model_sha256 = sha256(model_path) if model_path.is_file() else None

    baseline_path = ROOT / "artifacts" / "baseline_metrics.json"
    baseline = (
        json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_path.is_file()
        else {}
    )
    baseline_ap = (
        baseline.get("final_model", {})
        .get("metrics", {})
        .get("average_precision")
    )
    baseline_ap_4dp = f"{baseline_ap:.4f}" if isinstance(baseline_ap, int | float) else None

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release_notes = (ROOT / "docs" / "RELEASE_NOTES_v1.0.1.md").read_text(
        encoding="utf-8"
    )
    manual_index = (ROOT / "docs" / "manuals" / "README.md").read_text(
        encoding="utf-8"
    )
    day24 = json.loads(
        (ROOT / "docs" / "evidence" / "day24_release_candidate.json").read_text(
            encoding="utf-8"
        )
    )

    tag_present = bool(git_output("tag", "--list", RELEASE_TAG))

    lint = run_command(
        [sys.executable, "-m", "ruff", "check", "src", "tests", "demo", "scripts"]
    )
    tests = run_command(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "-q"]
    )
    test_match = PASSED_RE.search(tests["output"])
    test_count = int(test_match.group("count")) if test_match else None
    tests["test_count"] = test_count
    tests["expected_test_count"] = EXPECTED_TEST_COUNT
    tests["passed"] = tests["passed"] and test_count == EXPECTED_TEST_COUNT

    diff_check = run_command(["git", "diff", "--check"])
    link_failures = markdown_link_failures()
    hygiene_failures = tracked_hygiene_failures()
    unexpected_entries = unexpected_worktree_entries()

    documentation_checks = {
        "headline_ap_is_0_0368": (
            "| **Average Precision** | **0.0368** |" in readme
        ),
        "headline_ap_0_0369_absent": (
            "| **Average Precision** | **0.0369** |" not in readme
        ),
        "readme_release_is_v1_0_1": "release-v1.0.1" in readme,
        "readme_test_count_is_60": (
            "pytest (60 tests)" in readme
            and "expected: 60 passed" in readme
        ),
        "readme_manual_disclosure": (
            "Portuguese-language learning records" in readme
        ),
        "manual_index_disclosure": (
            "Portuguese-language learning records" in manual_index
        ),
        "release_notes_version": "# AMLGuard v1.0.1" in release_notes,
        "release_notes_model_unchanged": (
            "The model was not retrained." in release_notes
        ),
        "changelog_hotfix_first": (
            changelog.find("## [1.0.1]") < changelog.find("## [1.0.0]")
        ),
    }

    checks = {
        "branch_is_main": git_output("branch", "--show-current") == "main",
        "required_files_present": not missing_files,
        "package_version_is_1_0_1": package_version == RELEASE_VERSION,
        "service_version_is_1_0_1": service_version == RELEASE_VERSION,
        "model_version_remains_0_1_0": model_version == MODEL_VERSION,
        "version_domains_are_separate": service_version != model_version,
        "model_sha256_unchanged": actual_model_sha256 == EXPECTED_MODEL_SHA256,
        "baseline_average_precision_unchanged": baseline_ap == 0.036833,
        "baseline_rounds_to_0_0368": baseline_ap_4dp == "0.0368",
        "day24_evidence_remains_pass": day24.get("status") == "PASS",
        "documentation_contract_pass": all(documentation_checks.values()),
        "release_tag_available": not tag_present,
        "ruff_pass": lint["passed"],
        "pytest_60_pass": tests["passed"],
        "git_diff_check_pass": diff_check["passed"],
        "local_markdown_links_pass": not link_failures,
        "tracked_hygiene_pass": not hygiene_failures,
        "no_unexpected_worktree_entries": not unexpected_entries,
    }
    overall_status = "PASS" if all(checks.values()) else "FAIL"

    evidence = {
        "release": RELEASE_TAG,
        "scope": "maintenance_hotfix",
        "status": overall_status,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "version_domains": {
            "project_release": RELEASE_TAG,
            "python_package": package_version,
            "fastapi_service": service_version,
            "frozen_model_contract": model_version,
            "azure_ml_model_asset": "AMLGuard:1",
        },
        "repository": {
            "branch": git_output("branch", "--show-current"),
            "audited_commit": git_output("rev-parse", "HEAD"),
            "v1_0_0_commit": git_output("rev-list", "-n", "1", "v1.0.0"),
            "unexpected_worktree_entries": unexpected_entries,
        },
        "model_artifact": {
            "path": "artifacts/model.joblib",
            "sha256": actual_model_sha256,
            "expected_sha256": EXPECTED_MODEL_SHA256,
        },
        "metric_contract": {
            "frozen_average_precision": baseline_ap,
            "headline_four_decimal_display": baseline_ap_4dp,
        },
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
            "The public Streamlit demo is not covered by a production SLA.",
            (
                "Hotfix evidence is generated before the final commit and "
                "annotated v1.0.1 tag are created."
            ),
        ],
    }

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("AMLGuard v1.0.1 maintenance hotfix")
    print(f"- package version: {package_version}")
    print(f"- service version: {service_version}")
    print(f"- frozen model version: {model_version}")
    print(f"- model SHA-256 unchanged: {checks['model_sha256_unchanged']}")
    print(f"- baseline AP display: {baseline_ap} -> {baseline_ap_4dp}")
    print(f"- documentation contract passed: {checks['documentation_contract_pass']}")
    print(f"- Ruff passed: {checks['ruff_pass']}")
    print(f"- pytest passed: {tests['test_count']}/{EXPECTED_TEST_COUNT}")
    print(f"- local Markdown links passed: {checks['local_markdown_links_pass']}")
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
