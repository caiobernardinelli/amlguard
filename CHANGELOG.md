# Changelog

All notable changes to AMLGuard are documented in this file.

The format follows Keep a Changelog principles, and this project uses Semantic
Versioning for portfolio releases.

## [1.0.1] - 2026-07-22

### Fixed

- Corrected the README headline Average Precision from `0.0369` to `0.0368`,
  matching the frozen `0.036833` baseline rounded to four decimal places.
- Aligned the FastAPI service and OpenAPI version with the maintenance release:
  `service_version = 1.0.1`.

### Added

- Exact release-contract tests separating project/service version `1.0.1`
  from the frozen model contract `0.1.0`.
- A permanent SHA-256 integrity test for `artifacts/model.joblib`.
- English disclosure explaining that the study-manual PDFs are
  Portuguese-language learning records.

### Unchanged

- Frozen model contract: `0.1.0`.
- Frozen model SHA-256: `b9c069b8fe40f7d905b59a1af6ba2d9c21b1c848daf5c78fbbd50ad94cfe1d2e`.
- Decision threshold and validated model metrics.
- Azure ML registered model asset: `AMLGuard:1`.

## [1.0.0] - 2026-07-21

### Added

- Leakage-safe feature engineering and reproducible Weighted XGBoost training.
- Frozen baseline metrics with absolute quality and relative regression gates.
- FastAPI and Pydantic serving contracts for single and batch inference.
- Multi-stage Docker image, Docker Compose and GHCR publication.
- GitHub Actions checks with Ruff, pytest and Hadolint.
- MLflow Tracking, PyFunc packaging and candidate/champion model promotion.
- Azure ML workspace, versioned data/environment assets and reusable components.
- End-to-end Azure ML pipeline with gated model registration.
- Validated Managed Online Endpoint deployment and cloud inference evidence.
- Azure Monitor and Model Data Collector observability evidence.
- PSI-based data and prediction drift analysis with retraining governance.
- Public Streamlit recruiter demo using the frozen model artifact.
- Manual-guided portfolio-readiness audit covering Days 1-22.
- Final release documentation, evidence and study manual.

### Changed

- Project package version promoted from `0.1.0` to `1.0.0`.
- README updated to describe the complete 24-day delivery and final release.
- Reproduction instructions now lint all production, demo and audit scripts.

### Security and cost controls

- Raw transaction data, credentials and raw monitoring collections remain
  excluded from version control.
- The Azure ML Managed Online Endpoint remains deleted after validation to avoid
  continuous inference-compute cost.
- The public demo accepts synthetic or non-sensitive transaction profiles only.

### Important version distinction

- Portfolio/project release: `v1.0.0`.
- Frozen local model contract: `model_version = 0.1.0`.
- Azure ML registered model asset: `AMLGuard:1`.

The three identifiers belong to different versioning domains and are not
interchangeable.
