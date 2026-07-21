# Release-contract regression tests for AMLGuard v1.0.1.

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.api.main import SERVICE_VERSION
from src.config import MODEL_VERSION

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_SHA256 = "b9c069b8fe40f7d905b59a1af6ba2d9c21b1c848daf5c78fbbd50ad94cfe1d2e"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(
        r'(?ms)^\[project\].*?^version\s*=\s*"([^"]+)"',
        text,
    )
    assert match is not None
    return match.group(1)


def test_release_version_domains_are_explicitly_separate() -> None:
    assert _project_version() == "1.0.1"
    assert SERVICE_VERSION == "1.0.1"
    assert MODEL_VERSION == "0.1.0"
    assert SERVICE_VERSION != MODEL_VERSION


def test_frozen_model_artifact_sha256_is_unchanged() -> None:
    model_path = ROOT / "artifacts" / "model.joblib"
    assert model_path.is_file()
    assert _sha256(model_path) == EXPECTED_MODEL_SHA256


def test_readme_headline_average_precision_matches_frozen_baseline() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    baseline = json.loads(
        (ROOT / "artifacts" / "baseline_metrics.json").read_text(encoding="utf-8")
    )
    expected = f"{baseline['final_model']['metrics']['average_precision']:.4f}"

    assert expected == "0.0368"
    assert f"| **Average Precision** | **{expected}** |" in readme
    assert "| **Average Precision** | **0.0369** |" not in readme


def test_portuguese_study_manuals_are_disclosed_in_english() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    manual_index = (ROOT / "docs" / "manuals" / "README.md").read_text(
        encoding="utf-8"
    )

    disclosure = "Portuguese-language learning records"
    assert disclosure in readme
    assert disclosure in manual_index
