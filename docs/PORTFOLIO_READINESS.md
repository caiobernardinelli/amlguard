# AMLGuard Portfolio Readiness

## Purpose

Day 23 turns the completed AMLGuard implementation into an auditable portfolio
asset. The goal is not to add another model or cloud resource. The goal is to
prove that the repository tells a coherent, evidence-backed story from the
original academic notebook through production-style serving, monitoring and a
public recruiter demo.

The permanent validator is:

```text
scripts/audit/validate_day23_portfolio_readiness.py
```

Run it from the repository root:

```bash
python scripts/audit/validate_day23_portfolio_readiness.py
```

Generated evidence:

```text
docs/evidence/day23_portfolio_readiness.json
```

## Audit result

The Day 23 audit validates the delivery history for Days 1-22.

| Delivery range | Primary historical source | Repository proof |
| --- | --- | --- |
| Days 1-6 | Main AMLGuard study manual, MLOps refactor section | package structure, baseline, schema validation, feature parity, training, evaluation and prediction artifacts |
| Day 7 | Exact Git history | formal pytest suite and Ruff hygiene commit |
| Days 8-10 | Study Manual Volume 3 | FastAPI, Pydantic, prediction endpoints and Docker |
| Days 11-12 | Study Manual Volume 4 | GitHub Actions CI/CD and GHCR publication |
| Days 13-14 | Study Manual Volume 5 | MLflow Tracking, PyFunc packaging and Model Registry promotion |
| Days 15-17 | Study Manual Volume 6 | Azure ML workspace, data/environment assets and reusable components |
| Days 18-22 | Versioned docs, JSON evidence and exact Git commits | cloud pipeline, endpoint, monitoring, local demo and public demo |
| Day 20 interpretation | Study Manual Volume 7 | observability, drift limitations and retraining decision |

Day 23 acceptance result: `PASS` when all 22 prior delivery days, repository
hygiene checks, local Markdown links and public-demo references pass.

## Recruiter walkthrough

### 60-second route

1. Open the [live Streamlit demo](https://amlguard-demo.streamlit.app).
2. Read the README `TL;DR` and the operating-point metrics.
3. Review the production-architecture table.
4. Open this document and the Day 23 evidence.

This route demonstrates that the project is not only a notebook. It includes a
tested Python package, API serving, containerisation, CI/CD, experiment tracking,
model registries, Azure ML pipelines, online inference, monitoring, drift
analysis, retraining governance and a public interface.

### Five-minute technical route

1. **Problem framing:** extreme class imbalance and analyst alert capacity.
2. **Feature engineering:** leakage-safe historical transaction count and
   deliberately excluded high-risk identifiers.
3. **Model choice:** Weighted XGBoost selected by ranking quality rather than
   default-threshold accuracy.
4. **Operating point:** threshold chosen against minimum precision and recall
   requirements.
5. **Software engineering:** package structure, schema validation, tests,
   deterministic artifacts and regression gates.
6. **Serving and delivery:** FastAPI, Pydantic, Docker, GitHub Actions and GHCR.
7. **MLOps:** MLflow Tracking, PyFunc packaging, candidate/champion aliases and
   Azure ML model promotion.
8. **Cloud proof:** Azure ML components, end-to-end pipeline and Managed Online
   Endpoint evidence.
9. **Post-deployment governance:** Azure Monitor, Model Data Collector, data and
   prediction drift analysis, and a controlled retraining policy.
10. **Portfolio accessibility:** public Streamlit application backed by the
    frozen model artifact.

## Evidence hierarchy

The audit distinguishes different kinds of proof:

- **Current implementation evidence:** files, source code, tests and
  documentation that exist in the repository now.
- **Historical delivery evidence:** exact Git commits associated with each day.
- **Execution evidence:** structured JSON artifacts captured after validated
  local or cloud runs.
- **Study-manual context:** the conceptual explanation of why each delivery was
  made and what was learned.

A historical manual alone is not treated as proof that a file currently exists.
The validator cross-checks the manual-guided expectation against repository
artifacts.

## Important limitations

- The IBM AML dataset used by this project is synthetic.
- The model score is a ranking score, not a calibrated probability of money
  laundering.
- Predictions require human review and are not autonomous compliance decisions.
- The Day 20 traffic batch was synthetic and demonstrates monitoring mechanics;
  it does not prove real production drift.
- The Azure ML Managed Online Endpoint remains deleted to avoid continuous
  inference-compute cost.
- The public Streamlit demo depends on Community Cloud availability and is not
  covered by a production SLA.
- Days 1-17 are audited from current artifacts, manuals and Git history. Day 23
  does not recreate every historical training or cloud execution.

## Day 23 acceptance checklist

- [x] Days 1-22 mapped to their intended delivery.
- [x] Days 1-17 audited using the study manuals and current repository artifacts.
- [x] Day 7 gap resolved using exact Git history.
- [x] Days 18-22 checked against versioned evidence.
- [x] Day 20 decision validated as `DO_NOT_RETRAIN`.
- [x] Local Markdown links checked.
- [x] Tracked-file hygiene checked.
- [x] Public demo references checked.
- [x] Permanent validator added.
- [x] Structured Day 23 evidence generated.
