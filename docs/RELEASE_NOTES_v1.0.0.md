# AMLGuard v1.0.0

AMLGuard v1.0.0 is the first complete portfolio release of the project.

## Highlights

- End-to-end AML model development under approximately 0.1% positive prevalence.
- Leakage-safe feature engineering and reproducible Weighted XGBoost training.
- Frozen quality and regression gates tied to a business operating point.
- FastAPI, Pydantic, Docker, GHCR and GitHub Actions.
- MLflow Tracking, PyFunc packaging and controlled model promotion.
- Azure ML pipeline, model asset, validated online endpoint and monitoring.
- PSI-based drift analysis and documented retraining governance.
- Public Streamlit recruiter demo.
- Structured evidence and manual-guided audit across the full delivery history.

## Validated operating point

| Metric | Value |
| --- | ---: |
| Average Precision | `0.036833` |
| Minimum precision target | `2%` |
| Recall | `0.670316` |
| Frozen threshold | `0.892163` |
| Alerts | `52,044` |
| True positives | `1,041` |
| False positives | `51,003` |
| False negatives | `512` |

## Public demonstration

https://amlguard-demo.streamlit.app

The demo uses the frozen local model artifact and displays the score, alert
decision, threshold and model version. It is a portfolio interface, not a
production AML decision system.

## Verification

```bash
pip install -e ".[serve,dev,demo]"
python -m ruff check src tests demo scripts
python -m pytest -o addopts="" -q
```

## Versioning

- Project release: `v1.0.0`
- Python package: `1.0.0`
- Frozen model contract: `0.1.0`
- Azure ML model asset: `AMLGuard:1`

## Cost status

The Azure ML Managed Online Endpoint used for Days 19-20 was deleted after
validation. The public recruiter experience is provided by Streamlit Community
Cloud and does not require recreating the continuously provisioned Azure
endpoint.

## Limitations

The dataset is synthetic, the model score is not a calibrated probability,
predictions require human review, and the recorded drift workflow is a
demonstration using synthetic monitoring traffic.
