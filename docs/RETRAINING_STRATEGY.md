# AMLGuard Retraining Strategy

## Purpose

This document defines the initial production-style retraining policy for
AMLGuard. The policy separates three different concerns:

1. **Monitoring signals** indicate that the current population may have changed.
2. **Model-quality evidence** determines whether the existing model is degrading.
3. **Promotion gates** determine whether a newly trained candidate is safe to
   register and deploy.

Drift alone does **not** automatically trigger retraining or deployment.

> This is a portfolio-grade operational policy for the AMLGuard project.
> It is not a financial-regulatory standard or compliance policy.

## Monitoring inputs

AMLGuard monitors two classes of telemetry.

### Operational observability

- request volume;
- successful and failed responses;
- request latency;
- P50, P95, and P99 latency.

### Model observability

- input feature distributions;
- prediction score distribution;
- alert rate;
- data drift;
- prediction drift.

Production inference data is collected through Azure ML Model Data Collector
using the `model_inputs` and `model_outputs` collections.

## Drift heuristic

AMLGuard currently uses Population Stability Index (PSI) as a project-level
monitoring heuristic.

| PSI | Signal |
| --- | --- |
| `< 0.10` | Low |
| `0.10 <= PSI < 0.25` | Moderate |
| `>= 0.25` | Significant |

These thresholds are monitoring heuristics, not universal statistical laws.

A single significant PSI result is treated as an **investigation trigger**,
not as proof that the model must be retrained.

## Investigation trigger

Open a model-health investigation when one or more of the following conditions
persist across **three consecutive monitoring windows**:

- prediction PSI is `>= 0.25`;
- at least one business-relevant model input has PSI `>= 0.25`;
- multiple input features show moderate drift simultaneously;
- alert rate changes materially from its validated reference behavior;
- operational errors or latency indicate that scoring quality may be affected.

The monitoring window must contain representative traffic. Synthetic test
traffic, smoke tests, and controlled monitoring batches are excluded from
production retraining decisions.

## Retraining eligibility

A drift investigation may create a retraining candidate when:

1. the drift signal is sustained on representative traffic;
2. enough ground-truth labels have become available for a reliable evaluation;
3. model-quality evaluation shows degradation or a material population change
   justifies rebuilding the candidate model.

AMLGuard does not automatically retrain solely because PSI crosses a threshold.

## Frozen model-quality gates

Any retrained candidate must be evaluated against the existing AMLGuard
non-regression contract.

- Minimum Average Precision: `0.035`
- Precision operating target: `0.02`
- Minimum Recall at the operating point: `0.65`

The existing frozen baseline must not be silently changed.

A candidate that fails either quality gate is rejected.

## Candidate promotion flow

A retrained candidate follows this sequence:

1. reproduce the deterministic train/test pipeline;
2. train the candidate;
3. evaluate against the frozen quality gates;
4. compare with the current validated baseline;
5. record experiment metadata and artifacts;
6. register the candidate model only after gates pass;
7. validate serving compatibility;
8. deploy through a controlled deployment workflow;
9. run post-deployment smoke and observability checks;
10. promote traffic only after validation.

No drift signal bypasses these gates.

## Rollback principle

The currently validated model remains the fallback until the candidate:

- passes quality gates;
- serves correctly;
- preserves the inference contract;
- passes post-deployment health checks.

A failed candidate deployment must not replace the last known-good model.

## Day 20 interpretation

The Day 20 drift report used:

- the deterministic frozen AMLGuard test split as the reference population;
- 71 Azure ML Data Collector records as the current population;
- synthetic monitoring traffic for the current population.

The report produced significant drift signals. This is expected because the
current batch was deliberately synthetic and is not representative production
traffic.

**Decision: do not retrain from the Day 20 synthetic batch.**

The result validates the monitoring and drift-detection workflow only.

## Future production automation

A later production iteration may automate:

- scheduled drift computation;
- monitoring-window persistence checks;
- model-health alerts;
- retraining pipeline submission;
- candidate-versus-champion evaluation;
- approval-based promotion.

Automatic model deployment should remain gated by explicit quality and
deployment validation.
