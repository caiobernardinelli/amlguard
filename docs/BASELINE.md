# Baseline reference

This document freezes the **validated results** produced by the academic
notebook (`notebooks/01_aml_pipeline.ipynb`). It is the regression guard for the
end-to-end refactor: every script written during Phase 1 onward must reproduce
these numbers. The machine-readable version lives in
[`artifacts/baseline_metrics.json`](../artifacts/baseline_metrics.json) and is
loaded by the evaluation code.

## Why freeze a baseline

The notebook is the source of truth for *what the model does*. Moving the logic
into scripts, an API, containers and a cloud pipeline is a large refactor, and
refactors silently break numbers. Freezing the baseline means any regression is
caught by comparison rather than noticed months later.

**Rule:** no refactor is "done" until its output matches this baseline within
the tolerances below.

## Dataset and split

| Item | Value |
|---|---|
| Dataset | IBM AML — HI-Small |
| Transactions | 5,078,345 |
| Laundering cases | 5,177 (0.1019%) |
| Split | stratified 70 / 30, `random_state = 42` |
| Test set | 1,523,504 (1,553 laundering) |
| Model features | Payment Format, Amount Paid, sender_previous_tx_count, is_business_hours, same_account |
| Primary metric | Average Precision |

## Selected model — XGBoost with `scale_pos_weight`

Operating point = recall-maximising model at **minimum precision ≥ 2%**,
threshold ≈ **0.892163**.

| Metric | Value |
|---|---:|
| Average Precision | 0.036833 |
| Accuracy | 0.966187 |
| Precision | 0.020002 |
| Recall | 0.670316 |
| Alerts | 52,044 (3.42% of test) |
| True positives | 1,041 |
| False positives | 51,003 |
| False negatives | 512 |
| True negatives | 1,470,948 |

## Non-regression gate

A refactored pipeline is accepted only if:

- Average Precision ≥ **0.035**
- Recall at 2% precision ≥ **0.65**
- A reloaded model reproduces in-memory scores within **1e-6**

## Model comparison (threshold 0.50)

| Model | AP | Recall | Precision | Alert rate |
|---|---:|---:|---:|---:|
| XGBoost — Scale Pos Weight | 0.0368 | 89.6% | 0.82% | 11.2% |
| XGBoost — Random Undersampling 1:10 | 0.0333 | 60.3% | 2.19% | 2.81% |
| Random Forest — Balanced Subsample | 0.0325 | 81.8% | 1.47% | 5.67% |
| Logistic Regression — Balanced | 0.0084 | 90.3% | 0.73% | 12.6% |
| Logistic Regression — Unweighted | 0.0092 | 0.0% | — | 0.0% |
| Dummy (most-frequent) | 0.0010 | 0.0% | — | 0.0% |

_Frozen on Day 1 (11/07/2026) of the end-to-end MLOps schedule._
