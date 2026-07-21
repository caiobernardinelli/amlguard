# AMLGuard v1.0.1

AMLGuard v1.0.1 is a maintenance release that corrects public-facing
version and metric consistency without retraining or replacing the frozen model.

## Fixed

- README Average Precision headline: `0.0369` → `0.0368`.
  The frozen value remains `0.036833`.
- FastAPI `/health` and OpenAPI service version: `0.1.0` → `1.0.1`.

## Added

- Exact tests for project, service and model version-domain separation.
- Permanent SHA-256 protection for the frozen `artifacts/model.joblib`.
- Automated verification that the README headline metric matches the frozen baseline.
- English disclosure for the Portuguese-language study manuals.

## Version domains

| Domain | Version |
| --- | --- |
| Project release | `v1.0.1` |
| Python package | `1.0.1` |
| FastAPI service | `1.0.1` |
| Frozen model contract | `0.1.0` |
| Azure ML model asset | `AMLGuard:1` |

The model was not retrained. Its decision threshold, baseline metrics and artifact
hash are unchanged.

## Frozen model integrity

```text
SHA-256: b9c069b8fe40f7d905b59a1af6ba2d9c21b1c848daf5c78fbbd50ad94cfe1d2e
```

## Verification

```bash
python -m ruff check src tests demo scripts
python -m pytest -o addopts="" -q
python scripts/audit/validate_v1_0_1_hotfix.py
```

Expected result: `60 passed` and hotfix validation status `PASS`.

## Public demo

The Streamlit application remains the recruiter-facing interface:

https://amlguard-demo.streamlit.app

Streamlit Community Cloud availability is not a production SLA. Check the app in
a logged-out browser session and wake it before application campaigns when needed.
