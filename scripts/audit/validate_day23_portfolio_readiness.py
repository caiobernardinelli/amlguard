from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = ROOT / "docs" / "evidence" / "day23_portfolio_readiness.json"

PUBLIC_URL = "https://amlguard-demo.streamlit.app"
DAY7_COMMIT = "8f20dacb59c43601d826365d66e68726ef147c3c"

MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def file_check(path: str) -> dict[str, Any]:
    exists = (ROOT / path).is_file()
    return {
        "kind": "file",
        "target": path,
        "passed": exists,
        "detail": "present" if exists else "missing",
    }


def directory_check(path: str) -> dict[str, Any]:
    exists = (ROOT / path).is_dir()
    return {
        "kind": "directory",
        "target": path,
        "passed": exists,
        "detail": "present" if exists else "missing",
    }


def contains_check(path: str, tokens: list[str]) -> dict[str, Any]:
    target = ROOT / path
    if not target.is_file():
        return {
            "kind": "content",
            "target": path,
            "passed": False,
            "detail": "file missing",
            "tokens": tokens,
        }

    text = target.read_text(encoding="utf-8", errors="replace")
    missing = [token for token in tokens if token not in text]

    return {
        "kind": "content",
        "target": path,
        "passed": not missing,
        "detail": "all tokens present" if not missing else f"missing tokens: {missing}",
        "tokens": tokens,
    }


def git_commit_exists(commit: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    passed = completed.returncode == 0

    return {
        "kind": "git_commit",
        "target": commit,
        "passed": passed,
        "detail": "commit present" if passed else "commit missing",
    }


def day_commits(day: int) -> list[dict[str, str]]:
    completed = subprocess.run(
        ["git", "log", "--all", "--format=%H%x09%s"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    day_pattern = re.compile(rf"\bDay\s*{day}\b", re.IGNORECASE)
    commits: list[dict[str, str]] = []

    for line in completed.stdout.splitlines():
        if not line.strip():
            continue

        sha, _, message = line.partition("\t")
        if day_pattern.search(message):
            commits.append({"sha": sha, "message": message})

    return commits


def commit_history_check(day: int) -> dict[str, Any]:
    commits = day_commits(day)
    return {
        "kind": "git_history",
        "target": f"Day {day}",
        "passed": bool(commits),
        "detail": f"{len(commits)} matching commit(s)" if commits else "no matching commit",
        "commits": commits,
    }


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
    markdown_files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]

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


def build_day_specs() -> list[dict[str, Any]]:
    return [
        {
            "day": 1,
            "source": "Manual principal, Parte 3 / D1",
            "delivery": "Professional package structure and frozen baseline.",
            "checks": [
                commit_history_check(1),
                directory_check("src/data"),
                directory_check("src/features"),
                directory_check("src/models"),
                directory_check("src/api"),
                directory_check("tests"),
                directory_check("artifacts"),
                directory_check("docs"),
                file_check("src/config.py"),
                file_check("artifacts/baseline_metrics.json"),
                file_check("docs/BASELINE.md"),
                contains_check(
                    "src/config.py",
                    [
                        "RANDOM_STATE",
                        "MODEL_FEATURES",
                        "MIN_AVERAGE_PRECISION",
                        "MIN_RECALL_AT_2PCT_PRECISION",
                    ],
                ),
            ],
        },
        {
            "day": 2,
            "source": "Manual principal, Parte 3 / D2",
            "delivery": "Schema-validated raw-data loading with actionable errors.",
            "checks": [
                commit_history_check(2),
                file_check("src/data/load_data.py"),
                file_check("tests/test_data_loading.py"),
                file_check("scripts/smoke_test_day2.py"),
                contains_check(
                    "src/data/load_data.py",
                    ["SchemaValidationError", "validate_schema"],
                ),
            ],
        },
        {
            "day": 3,
            "source": "Manual principal, Parte 3 / D3",
            "delivery": "Feature parity, no-target proof and leakage-safe features.",
            "checks": [
                commit_history_check(3),
                file_check("src/features/build_features.py"),
                file_check("scripts/verify_features_parity.py"),
                file_check("tests/test_features.py"),
                contains_check(
                    "src/features/build_features.py",
                    ["sender_previous_tx_count", "cumcount"],
                ),
            ],
        },
        {
            "day": 4,
            "source": "Manual principal, Parte 3 / D4",
            "delivery": "Reproducible training with frozen hyperparameters.",
            "checks": [
                commit_history_check(4),
                file_check("src/models/train.py"),
                file_check("tests/test_train.py"),
                contains_check(
                    "src/models/train.py",
                    ["prepare_train_test_split", "scale_pos_weight"],
                ),
            ],
        },
        {
            "day": 5,
            "source": "Manual principal, Parte 3 / D5",
            "delivery": "Absolute quality gate and relative regression gate.",
            "checks": [
                commit_history_check(5),
                file_check("src/models/evaluate.py"),
                file_check("artifacts/baseline_metrics.json"),
                contains_check(
                    "src/models/evaluate.py",
                    ["quality_gate", "regression_gate"],
                ),
            ],
        },
        {
            "day": 6,
            "source": "Manual principal, Parte 3 / D6",
            "delivery": "Unified cached prediction contract.",
            "checks": [
                commit_history_check(6),
                file_check("src/models/predict.py"),
                file_check("scripts/verify_predict.py"),
                file_check("tests/test_predict.py"),
                contains_check(
                    "src/models/predict.py",
                    ["predict_transaction", "predict_batch", "risk_score"],
                ),
            ],
        },
        {
            "day": 7,
            "source": "Git history between Volumes 1 and 3",
            "delivery": "Formal pytest suite and Ruff hygiene.",
            "checks": [
                git_commit_exists(DAY7_COMMIT),
                commit_history_check(7),
                directory_check("tests"),
                file_check("pyproject.toml"),
            ],
        },
        {
            "day": 8,
            "source": "Manual Volume 3",
            "delivery": "FastAPI administrative endpoints and model lifecycle.",
            "checks": [
                commit_history_check(8),
                file_check("src/api/main.py"),
                file_check("tests/test_api.py"),
                contains_check("src/api/main.py", ["/health", "/model-info"]),
            ],
        },
        {
            "day": 9,
            "source": "Manual Volume 3",
            "delivery": "Validated single and batch prediction endpoints.",
            "checks": [
                commit_history_check(9),
                file_check("src/api/main.py"),
                file_check("tests/test_api.py"),
                contains_check("src/api/main.py", ["/predict", "/predict-batch"]),
            ],
        },
        {
            "day": 10,
            "source": "Manual Volume 3",
            "delivery": "Portable Docker image and Docker Compose service.",
            "checks": [
                commit_history_check(10),
                file_check("Dockerfile"),
                file_check("docker-compose.yml"),
                file_check(".dockerignore"),
                contains_check("Dockerfile", ["HEALTHCHECK"]),
            ],
        },
        {
            "day": 11,
            "source": "Manual Volume 4",
            "delivery": "GitHub Actions CI with Python and Dockerfile checks.",
            "checks": [
                commit_history_check(11),
                file_check(".github/workflows/ci.yml"),
                contains_check(
                    ".github/workflows/ci.yml",
                    ["pytest", "ruff", "hadolint"],
                ),
            ],
        },
        {
            "day": 12,
            "source": "Manual Volume 4",
            "delivery": "Quality-gated Docker publication to GHCR.",
            "checks": [
                commit_history_check(12),
                file_check(".github/workflows/ci.yml"),
                contains_check(
                    ".github/workflows/ci.yml",
                    ["ghcr.io", "docker/build-push-action"],
                ),
                contains_check(
                    "README.md",
                    ["ghcr.io/caiobernardinelli/amlguard"],
                ),
            ],
        },
        {
            "day": 13,
            "source": "Manual Volume 5",
            "delivery": "MLflow experiment tracking with SQLite provenance.",
            "checks": [
                commit_history_check(13),
                file_check("src/models/tracking.py"),
                file_check("tests/test_tracking.py"),
                contains_check("src/models/tracking.py", ["sqlite", "mlflow"]),
            ],
        },
        {
            "day": 14,
            "source": "Manual Volume 5",
            "delivery": "MLflow PyFunc packaging and candidate/champion promotion.",
            "checks": [
                commit_history_check(14),
                file_check("src/models/mlflow_model.py"),
                file_check("tests/test_mlflow_model.py"),
                file_check("docs/MODEL_PROMOTION.md"),
                contains_check(
                    "docs/MODEL_PROMOTION.md",
                    ["candidate", "champion"],
                ),
            ],
        },
        {
            "day": 15,
            "source": "Manual Volume 6",
            "delivery": "Azure ML workspace foundation and resource governance.",
            "checks": [
                commit_history_check(15),
                file_check(".env.example"),
                file_check("docs/AZURE_RESOURCES.md"),
                contains_check(
                    "docs/AZURE_RESOURCES.md",
                    ["rg-amlguard-dev-brazilsouth", "mlw-amlguard-dev"],
                ),
            ],
        },
        {
            "day": 16,
            "source": "Manual Volume 6",
            "delivery": "Versioned Azure Data Asset and training Environment.",
            "checks": [
                commit_history_check(16),
                file_check("cloud/azure/data/ibm_aml_hi_small.yml"),
                file_check("cloud/azure/environment/conda.yml"),
                file_check("cloud/azure/environment/environment.yml"),
                file_check("cloud/azure/jobs/day16_smoke.yml"),
                file_check("docs/AZURE_DATA_ENVIRONMENT.md"),
                file_check("docs/evidence/day16_smoke_summary.json"),
            ],
        },
        {
            "day": 17,
            "source": "Manual Volume 6",
            "delivery": "Reusable prepare, train and evaluate Azure ML components.",
            "checks": [
                commit_history_check(17),
                file_check("cloud/azure/components/prepare_data/component.yml"),
                file_check("cloud/azure/components/train_model/component.yml"),
                file_check("cloud/azure/components/evaluate_model/component.yml"),
                file_check("docs/AZURE_COMPONENTS.md"),
                file_check("docs/evidence/day17_evaluation_metrics.json"),
                contains_check(
                    "docs/evidence/day17_evaluation_metrics.json",
                    ['"status": "PASS"', '"quality_gate"', '"regression_gate"'],
                ),
            ],
        },
        {
            "day": 18,
            "source": "Versioned repository evidence",
            "delivery": "End-to-end Azure ML pipeline and gated model registration.",
            "checks": [
                commit_history_check(18),
                file_check("cloud/azure/jobs/day18_pipeline.yml"),
                file_check("docs/AZURE_PIPELINE.md"),
                file_check("docs/evidence/day18_evaluation_metrics.json"),
                contains_check(
                    "docs/evidence/day18_evaluation_metrics.json",
                    ['"status": "PASS"', '"quality_gate"', '"regression_gate"'],
                ),
            ],
        },
        {
            "day": 19,
            "source": "Versioned repository evidence",
            "delivery": "Managed Online Endpoint and validated cloud inference.",
            "checks": [
                commit_history_check(19),
                file_check("cloud/azure/online/deployment.yml"),
                file_check("cloud/azure/online/score.py"),
                file_check("docs/AZURE_ONLINE_ENDPOINT.md"),
                file_check("docs/evidence/day19_online_endpoint.json"),
                contains_check(
                    "docs/evidence/day19_online_endpoint.json",
                    ['"status": "PASS"', '"threshold"', '"model_version"'],
                ),
            ],
        },
        {
            "day": 20,
            "source": "Manual Volume 7 and versioned repository evidence",
            "delivery": "Operational/model monitoring, drift and retraining policy.",
            "checks": [
                commit_history_check(20),
                file_check("docs/AZURE_MONITORING.md"),
                file_check("docs/RETRAINING_STRATEGY.md"),
                file_check("docs/evidence/day20_operational_observability.json"),
                file_check("docs/evidence/day20_data_collection_validation.json"),
                file_check("docs/evidence/day20_drift_report.json"),
                file_check("docs/evidence/day20_retraining_strategy.json"),
                contains_check(
                    "docs/evidence/day20_retraining_strategy.json",
                    ['"status": "PASS"', "DO_NOT_RETRAIN"],
                ),
            ],
        },
        {
            "day": 21,
            "source": "Versioned repository evidence",
            "delivery": "Validated local recruiter-facing Streamlit demo.",
            "checks": [
                commit_history_check(21),
                file_check("demo/app.py"),
                file_check("scripts/demo/validate_day21_recruiter_demo.py"),
                file_check("docs/RECRUITER_DEMO.md"),
                file_check("docs/evidence/day21_recruiter_demo.json"),
                contains_check(
                    "docs/evidence/day21_recruiter_demo.json",
                    ['"status": "PASS"', '"risk_score"'],
                ),
            ],
        },
        {
            "day": 22,
            "source": "Versioned repository evidence",
            "delivery": "Public Streamlit Community Cloud deployment.",
            "checks": [
                commit_history_check(22),
                file_check("demo/requirements.txt"),
                file_check("scripts/demo/record_day22_public_deployment.py"),
                file_check("docs/evidence/day22_public_deployment.json"),
                contains_check(
                    "docs/evidence/day22_public_deployment.json",
                    ['"status": "PASS"', PUBLIC_URL, '"python_version": "3.12"'],
                ),
                contains_check("README.md", [PUBLIC_URL]),
            ],
        },
    ]


def main() -> int:
    branch = git_output("branch", "--show-current")
    commit = git_output("rev-parse", "HEAD")
    working_tree_entries = git_output("status", "--short").splitlines()

    days = build_day_specs()
    failed_days: list[int] = []
    failed_checks: list[dict[str, Any]] = []

    for day in days:
        day["status"] = (
            "PASS" if all(check["passed"] for check in day["checks"]) else "FAIL"
        )

        if day["status"] == "FAIL":
            failed_days.append(day["day"])

        for check in day["checks"]:
            if not check["passed"]:
                failed_checks.append(
                    {
                        "day": day["day"],
                        "delivery": day["delivery"],
                        **check,
                    }
                )

    link_failures = markdown_link_failures()
    hygiene_failures = tracked_hygiene_failures()
    public_url_checks = {
        "README.md": PUBLIC_URL
        in (ROOT / "README.md").read_text(encoding="utf-8"),
        "docs/RECRUITER_DEMO.md": PUBLIC_URL
        in (ROOT / "docs" / "RECRUITER_DEMO.md").read_text(encoding="utf-8"),
    }

    global_status = (
        "PASS"
        if not link_failures
        and not hygiene_failures
        and all(public_url_checks.values())
        else "FAIL"
    )
    overall_status = "PASS" if not failed_days and global_status == "PASS" else "FAIL"

    evidence = {
        "day": 23,
        "scope": "manual_guided_portfolio_readiness_audit",
        "status": overall_status,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "branch": branch,
            "git_commit": commit,
            "working_tree_entries_at_start": working_tree_entries,
        },
        "audit_method": {
            "days_1_to_6": "Manual principal, Parte 3, checked against repository artifacts",
            "day_7": (
                "Git history, including the formal pytest and Ruff commit "
                f"{DAY7_COMMIT}"
            ),
            "days_8_to_10": "Manual Volume 3, checked against API and Docker artifacts",
            "days_11_to_12": "Manual Volume 4, checked against CI/CD artifacts",
            "days_13_to_14": "Manual Volume 5, checked against MLflow artifacts",
            "days_15_to_17": "Manual Volume 6, checked against Azure ML artifacts",
            "days_18_to_22": (
                "Versioned documentation, evidence JSONs and exact Day commits"
            ),
            "day_20_interpretation": (
                "Manual Volume 7, preserving the synthetic-traffic limitation"
            ),
        },
        "days": days,
        "global_checks": {
            "status": global_status,
            "broken_local_markdown_links": link_failures,
            "tracked_hygiene_failures": hygiene_failures,
            "public_url_checks": public_url_checks,
        },
        "failed_days": failed_days,
        "failed_checks": failed_checks,
        "limitations": [
            (
                "Days 1-17 are historical artifact audits guided by the study manuals; "
                "the audit does not rerun every historical cloud or training job."
            ),
            (
                "Azure resources are not recreated. The Managed Online Endpoint remains "
                "deleted to avoid continuous inference-compute cost."
            ),
            (
                "The Day 20 monitoring batch was synthetic and demonstrates monitoring "
                "mechanics, not real production drift."
            ),
            (
                "The public Streamlit demo is a portfolio interface and not a production "
                "AML decision system."
            ),
        ],
    }

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("AMLGuard Day 23 portfolio-readiness evidence")
    print(f"- branch: {branch}")
    print(f"- commit audited: {commit}")
    print(f"- days passed: {sum(day['status'] == 'PASS' for day in days)}/22")
    print(f"- broken local Markdown links: {len(link_failures)}")
    print(f"- tracked hygiene failures: {len(hygiene_failures)}")
    print(f"- public URL checks passed: {all(public_url_checks.values())}")
    print(f"- status: {overall_status}")
    print(f"- evidence: {EVIDENCE_PATH.relative_to(ROOT)}")

    if failed_checks:
        print("- failed checks:")
        print(json.dumps(failed_checks, indent=2, ensure_ascii=False))

    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
