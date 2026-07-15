# AMLGuard — Anti-Money Laundering Detection under Extreme Class Imbalance

**End-to-end machine-learning pipeline for financial-crime / fraud detection on a highly imbalanced transaction dataset (~0.1% illicit).** Built with Python, scikit-learn, XGBoost, imbalanced-learn and SHAP, and evaluated the way a production compliance team actually operates — by *alert budget* and *recall at a fixed precision*, not by accuracy.

![CI](https://github.com/caiobernardinelli/amlguard/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-brightgreen)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Skills demonstrated:** fraud detection · anomaly detection · imbalanced classification · precision–recall / PR-AUC (Average Precision) · feature engineering · data-leakage prevention · threshold tuning · cost-sensitive learning · XGBoost · Random Forest · Logistic Regression · SHAP explainability · model evaluation · FastAPI scoring skeleton · MLOps roadmap (Docker · MLflow · DVC).

---

## TL;DR

On **5,078,345** real-schema transactions where only **0.1019%** are laundering, a naïve "everything is legitimate" classifier scores **99.9% accuracy** and catches **zero** criminals — so accuracy is discarded from the start. AMLGuard instead ranks transactions by risk and selects an operating point against a realistic analyst workload.

**Selected model — XGBoost with `scale_pos_weight`, at a 2% minimum-precision operating point:**

| Metric | Value | What it means for a compliance team |
|---|---:|---|
| **Average Precision** | **0.0369** | Best ranking of all evaluated models (~36× the 0.001 base rate) |
| **Recall** | **67.0%** | Detects two-thirds of the labelled laundering transactions |
| **Precision** | **2.0%** | ~1 true case per 50 reviewed alerts |
| **Alert rate** | **3.42%** | Only 3.42% of transactions sent for human review |
| True / False positives | 1,041 / 51,003 | 1,041 laundering cases caught |
| False negatives | 512 | Undetected laundering — the residual regulatory risk |

The point of the project is **not** a claim that laundering is "solved." It is a documented, leakage-aware, operationally-evaluated pipeline that makes the precision–recall–workload trade-off explicit.

---

## Why this problem matters (business context)

The EU's **Anti-Money Laundering Authority (AMLA)**, based in Frankfurt, took over the EU-level AML/CFT mandates from the European Banking Authority on **1 January 2026** and will directly supervise up to ~40 high-risk cross-border institutions from 2028. European banks are under rising regulatory pressure while drowning in **false positives** — legacy monitoring systems raise far more alerts than analysts can review. Because illicit activity is **well under 1%** of all transactions, this is a textbook **extreme class-imbalance** problem, which is exactly what AMLGuard is built to handle.

---

## Dataset

- **Source:** [IBM Transactions for Anti-Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) — version **HI-Small** (high illicit ratio).
- **Size:** 5,078,345 transactions · **5,177** labelled laundering cases · **0.1019%** positive rate · ~18-day window.
- **License:** Community Data License Agreement — Sharing 1.0.
- **Note on validity:** the data is **synthetic**. Strong signals may reflect simulator rules rather than generalisable laundering behaviour, so every high-lift feature is interpreted critically throughout the notebook.

---

## Results at a glance

**Recall grows with the alert budget** — the core operational trade-off. More analyst capacity → more laundering caught, at diminishing returns:

![Recall by alert budget](./assets/recall_by_alert_budget.png)

**Precision–recall curves** (the correct lens under extreme imbalance — ROC would be misleadingly optimistic here):

![Precision–recall curves](./assets/precision_recall_curves.png)

**Selected operating point — confusion matrix** (XGBoost, threshold ≈ 0.892, precision fixed at 2%):

![Final confusion matrix](./assets/final_confusion_matrix.png)

### Model comparison (default threshold 0.50)

| Model | Average Precision | Recall | Precision | Alert rate |
|---|---:|---:|---:|---:|
| **XGBoost — Scale Pos Weight** | **0.0368** | 89.6% | 0.82% | 11.2% |
| XGBoost — Random Undersampling 1:10 | 0.0333 | 60.3% | 2.19% | 2.81% |
| Random Forest — Balanced Subsample | 0.0325 | 81.8% | 1.47% | 5.67% |
| Logistic Regression — Balanced | 0.0084 | 90.3% | 0.73% | 12.6% |
| Logistic Regression — Unweighted | 0.0092 | 0.0% | — | 0.0% |
| Dummy (most-frequent) | 0.0010 | 0.0% | — | 0.0% |

**Methodological finding:** class weighting did **not** improve Average Precision over the unweighted linear model — it only shifted where the 0.50 threshold falls. Re-weighting is therefore treated as an implicit *threshold* intervention, and the real lever is **threshold tuning on the best-ranking model**, not reflexive rebalancing.

---

## Methodology

1. **Descriptive analysis & EDA** — quantify the imbalance; test hypotheses (payment format, cross-currency, same-bank) using **illicit rate within group**, never raw counts.
2. **Feature engineering (leakage-safe):** `Payment Format`, `Amount Paid`, `sender_previous_tx_count` (chronological `cumcount` — no future information), `is_business_hours`, `same_account`. Cross-currency, account/bank IDs and redundant amount columns are **excluded** with documented justification.
3. **Data preparation:** stratified 70/30 split; all encoding/scaling fitted **inside the pipeline on the training fold only** to prevent data leakage.
4. **Modelling:** Dummy → Logistic Regression (weighted + unweighted control) → Random Forest → **XGBoost** → random undersampling. Imbalance handled via `class_weight` / `scale_pos_weight` / undersampling; SMOTE scope explicitly reasoned about rather than blindly applied.
5. **Evaluation for the real world:** precision–recall curves, **recall at fixed precision** (0.5% / 1% / 2% / 5%), and **recall under fixed alert budgets** — translating the threshold into "how much can the compliance team review per day."
6. **Explainability:** **SHAP** global (beeswarm) + local explanations for the highest-risk true-positive and false-positive alerts, with sparse-matrix consistency assertions and a model-risk note on potentially generator-specific features.
7. **Minimal serving layer:** in-notebook scoring function + optional **FastAPI** `/score` + `/health` skeleton.

---

## Explainability (SHAP)

Global feature contributions for the selected XGBoost model. Explanations improve **auditability** but do not prove causality, and features such as `same_account` are flagged as potentially encoding synthetic-generator rules:

![SHAP beeswarm](./assets/shap_beeswarm.png)

---

## Repository structure

```
amlguard/
├── notebooks/
│   └── 01_aml_pipeline.ipynb     # full pipeline — Sections 1–9 (EDA → modelling → SHAP → conclusion)
├── src/
│   ├── features.py               # leakage-safe feature functions (reusable by the API)
│   └── api.py                    # FastAPI scoring skeleton (Phase 2)
├── assets/                       # figures used in this README
├── data/
│   └── raw/                      # HI-Small_Trans.csv is git-ignored (476 MB; fetched at runtime)
├── docker-compose.yml            # Phase 2 serving stack (documented skeleton)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to reproduce

```bash
# 1. Clone
git clone https://github.com/caiobernardinelli/amlguard.git
cd amlguard

# 2. Environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the notebook
jupyter lab notebooks/01_aml_pipeline.ipynb
```

The notebook downloads `HI-Small_Trans.csv` automatically on first run (via `gdown`); the raw CSV is intentionally **not** committed. For a clean top-to-bottom run, use **Kernel → Restart & Run All** (the notebook is ordered to reproduce every result from scratch).

---

## Limitations (stated honestly)

- **Synthetic data** and an **~18-day window** limit external validity.
- The **random stratified split** does not reproduce a true future-deployment period; the same entities may appear in train and test.
- No KYC, sanctions, geography or investigation-history context is available.
- `same_account` and some payment-format signals may reflect **simulator rules**, so results should not be read as real-world laundering laws.

---

## Roadmap (Phase 2)

| Priority | Improvement | Value |
|---|---|---|
| 1 | Temporal + entity-aware validation | Generalisation to future periods / unseen accounts |
| 2 | Independent threshold calibration | Separate model comparison, operating-point choice, final eval |
| 3 | Rolling behavioural features | Velocity over multiple windows, counterparty diversity |
| 4 | **Graph features (NetworkX)** | Fan-in/fan-out, cycles, hubs — laundering as a network |
| 5 | Sensitivity analysis without `same_account` | Quantify dependence on a possibly confounded signal |
| 6 | SMOTENC / cost-sensitive comparison | Category-aware resampling |
| 7 | Probability calibration | Stable, interpretable risk scores |
| 8 | Serving stack: **FastAPI + Docker + MLflow + DVC** + Streamlit triage UI | Deployable, tracked, reproducible pipeline |

---

## Author

**Caio Bernardinelli** — Data & AI professional transitioning from 15 years across engineering, Business Intelligence and data. Building end-to-end ML pipelines for the EU and Brazilian markets.
Holds **Portuguese (EU) citizenship** — eligible to work in the European Union without a visa.

- GitHub: [@caiobernardinelli](https://github.com/caiobernardinelli)
- LinkedIn: https://www.linkedin.com/in/caio-fl%C3%A1vio-bernardinelli/


*Developed as the capstone (Projeto Integrador I) of the Técnico em Inteligência Artificial programme at IFNMG, and maintained as a portfolio project.*

## License

Code released under the **MIT License**. The dataset is governed by its own Community Data License Agreement (Sharing 1.0) and is not redistributed here.
