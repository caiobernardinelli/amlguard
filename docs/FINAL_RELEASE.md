# AMLGuard Final Release

## Release identity

| Item | Value |
| --- | --- |
| Portfolio release | `v1.0.0` |
| Python package version | `1.0.0` |
| Frozen model contract | `0.1.0` |
| Azure ML model asset | `AMLGuard:1` |
| Public demo | https://amlguard-demo.streamlit.app |
| Final delivery day | Day 24 |

The portfolio release, model contract and Azure model asset use separate
versioning domains. Publishing `v1.0.0` does not silently replace or retrain the
frozen `0.1.0` model artifact.

## What the release proves

AMLGuard is an end-to-end machine-learning and MLOps portfolio project for
anti-money-laundering detection under extreme class imbalance. The release
connects the following layers into one auditable story:

1. schema-validated data loading;
2. leakage-safe feature engineering;
3. reproducible model training;
4. business-oriented precision/recall operating-point selection;
5. absolute quality and relative regression gates;
6. tested prediction contracts;
7. FastAPI and Pydantic serving;
8. Docker, Docker Compose and GHCR;
9. GitHub Actions CI/CD;
10. MLflow experiment tracking and model registry;
11. Azure ML data, environment, components and pipeline;
12. controlled cloud model registration;
13. validated real-time Azure inference;
14. observability, data collection, drift and retraining governance;
15. public Streamlit demonstration;
16. manual-guided portfolio and release audits.

## Final release validator

The permanent Day 24 validator will be stored at:

```text
scripts/audit/validate_day24_release_candidate.py
```

It will write:

```text
docs/evidence/day24_release_candidate.json
```

## Release procedure

```text
apply final files
→ add final study manual
→ run the Day 24 validator
→ review Git diff
→ commit
→ push main
→ create annotated tag v1.0.0
→ push the tag
→ create the GitHub Release from the versioned release notes
```

The tag must point to the final validated commit. It must not be created before
the final commit exists.

## Study manuals

The complete seven-volume study collection is indexed at
[`docs/manuals/README.md`](./manuals/README.md). The final volume is
[`AMLGuard_Manual_Estudo_Vol7_Demo_Portfolio_Release.pdf`](./manuals/AMLGuard_Manual_Estudo_Vol7_Demo_Portfolio_Release.pdf)
and covers Days 21-24.

## Honest limitations

- The IBM AML dataset is synthetic.
- The labelled window is short and the split is not a true future-period test.
- The model score is not a calibrated probability of money laundering.
- Predictions require human review.
- Day 20 synthetic monitoring traffic demonstrates mechanics, not real
  production drift.
- The deleted Azure endpoint is preserved through code and evidence, not as an
  always-on paid service.
- Streamlit Community Cloud availability is not a production SLA.
