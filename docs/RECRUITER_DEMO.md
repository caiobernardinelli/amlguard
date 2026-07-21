# AMLGuard Recruiter Demo

## Purpose

The Recruiter Demo is a lightweight Streamlit interface that lets a reviewer
enter a synthetic transaction profile and obtain a score from the same
persisted AMLGuard model used by the project's local prediction contract.

Day 21 delivered the **validated local demo**. Day 22 deployed the same
interface publicly on Streamlit Community Cloud and validated a live
browser submission against the frozen prediction contract.

## Entry point

```text
demo/app.py
```

Local command:

```bash
python -m streamlit run demo/app.py
```

## Public deployment

Live application:

```text
https://amlguard-demo.streamlit.app
```

Deployment contract:

| Setting | Validated value |
| --- | --- |
| Platform | Streamlit Community Cloud |
| Repository | `caiobernardinelli/amlguard` |
| Branch | `main` |
| Entrypoint | `demo/app.py` |
| Dependency file | `demo/requirements.txt` |
| Python runtime | `3.12` |
| Secrets required | No |

The first deployment used the platform default Python `3.14.6` runtime and failed
while importing the persisted-model dependency chain. The app was deleted and
recreated with Python `3.12`, matching the validated local environment. The
recreated deployment loaded successfully and produced the expected live result.

The application calls:

```python
src.models.predict.predict_transaction
```

and preserves the project response contract:

- `risk_score`;
- `is_alert`;
- `threshold`;
- `model_version`.

## Model artifact

The recruiter demo uses the frozen local model artifact:

```text
artifacts/model.joblib
```

The file is intentionally whitelisted in `.gitignore` for the recruiter demo.
Other generated artifacts remain ignored.

Validated Day 21 artifact:

```text
SHA-256: b9c069b8fe40f7d905b59a1af6ba2d9c21b1c848daf5c78fbbd50ad94cfe1d2e
```

The artifact is small enough to ship with the portfolio repository and avoids
requiring a continuously running Azure ML Managed Online Endpoint solely for a
public demonstration.

## Input contract

The interface exposes the five validated AMLGuard model features:

| Recruiter-facing input | Model feature |
| --- | --- |
| Payment format | `Payment Format` |
| Amount paid | `Amount Paid` |
| Sender previous transaction count | `sender_previous_tx_count` |
| Business-hours checkbox | `is_business_hours` |
| Same-account checkbox | `same_account` |

The interface accepts synthetic or non-sensitive values only. It must not be
used to submit real account identifiers, personal data, credentials, or
confidential banking information.

## Validated local example

Day 21 validated the following synthetic input:

```json
{
  "Payment Format": "ACH",
  "Amount Paid": 13701.30,
  "sender_previous_tx_count": 238,
  "is_business_hours": 0,
  "same_account": 0
}
```

Result:

```json
{
  "risk_score": 0.985365629196167,
  "is_alert": true,
  "threshold": 0.892163,
  "model_version": "0.1.0"
}
```

The Streamlit interface displays the score rounded to four decimal places
(`0.9854`) while the validation evidence stores the full model output.

The score is presented as a model risk score in `[0, 1]`, not as a calibrated
probability.

## Validation

Permanent validator:

```text
scripts/demo/validate_day21_recruiter_demo.py
```

Run:

```bash
python scripts/demo/validate_day21_recruiter_demo.py
```

Evidence:

```text
docs/evidence/day21_recruiter_demo.json
```

The validator checks:

- demo entry point availability;
- exact model artifact SHA-256;
- recruiter-form to model-feature mapping;
- exact validated local inference score;
- alert decision;
- frozen operating threshold;
- model version.

Day 21 acceptance result: `PASS`.

## Day 22 public validation

Permanent evidence recorder:

```text
scripts/demo/record_day22_public_deployment.py
```

Run:

```bash
python scripts/demo/record_day22_public_deployment.py
```

Evidence:

```text
docs/evidence/day22_public_deployment.json
```

The Day 22 acceptance check combined repository-side contract validation with a
manual public-browser test. Using the default synthetic form values, the live
application displayed:

```json
{
  "risk_score_display": 0.9854,
  "is_alert": true,
  "threshold": 0.892163,
  "model_version": "0.1.0"
}
```

The public URL rendered successfully and the form submission completed. Direct
`urllib` checks were not used as an availability gate because the Community
Cloud routing layer returned repeated HTTP `303` redirects to that client.

Day 22 acceptance result: `PASS`.

## CI coverage

The GitHub Actions Python job now installs the `demo` optional dependency group
and includes both `demo/` and `scripts/demo/` in Ruff checks.

The existing pytest suite remains unchanged at 56 passing tests for Day 21.
The permanent Day 21 validator provides explicit demo/model parity evidence.

## Cost-aware deployment strategy

The Azure ML Managed Online Endpoint was used to prove real cloud deployment,
online inference, Azure Monitor telemetry, and Model Data Collector behaviour.
After the required Day 19-20 evidence was captured, the endpoint was deleted to
avoid paying for continuously provisioned inference compute.

The recruiter demo is deliberately separated from that cloud proof:

```text
Azure ML deployment evidence
    -> proves production-style cloud serving and monitoring

Recruiter Demo
    -> provides a low-cost public portfolio experience
```

Day 22 deployed the portfolio interface on Streamlit Community Cloud. The
continuously provisioned Azure ML endpoint remains deleted and is not required
for the public recruiter experience.

## Limitations

- The underlying AML dataset is synthetic.
- The demo is a portfolio interface, not a production AML decision system.
- Model outputs require human review and must not be treated as autonomous
  compliance decisions.
- Public availability depends on Streamlit Community Cloud and may include
  platform cold starts or temporary service interruptions.
- Day 22 validates browser rendering and one successful live inference; it
  does not constitute a production availability SLA.
