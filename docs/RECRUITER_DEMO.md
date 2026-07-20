# AMLGuard Recruiter Demo

## Purpose

The Recruiter Demo is a lightweight Streamlit interface that lets a reviewer
enter a synthetic transaction profile and obtain a score from the same
persisted AMLGuard model used by the project's local prediction contract.

The Day 21 deliverable is a **validated local demo**. Public hosting is planned
for the next delivery step and is not claimed as delivered here.

## Entry point

```text
demo/app.py
```

Local command:

```bash
python -m streamlit run demo/app.py
```

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

The public deployment step should prefer a free-tier or scale-to-zero hosting
option. A continuously provisioned Azure ML endpoint is not required merely to
keep the portfolio demo accessible.

## Limitations

- The underlying AML dataset is synthetic.
- The demo is a portfolio interface, not a production AML decision system.
- Model outputs require human review and must not be treated as autonomous
  compliance decisions.
- The Day 21 demo is local; no public URL is claimed yet.
- Public-hosting availability and cold-start behaviour will be documented only
  after deployment is actually validated.
