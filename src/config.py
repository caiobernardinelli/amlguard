"""Central configuration for AMLGuard.

Single source of truth for paths, feature definitions, the random seed and the
selected-model contract. Every script (data loading, feature engineering,
training, evaluation, serving) imports from here so that no value is duplicated
and no personal/absolute path is hard-coded anywhere in the codebase.

Paths are resolved relative to the repository root, so the project runs
identically on any machine or in CI.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (all relative to the repository root — never machine-specific)
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"

RAW_CSV_PATH: Path = RAW_DATA_DIR / "HI-Small_Trans.csv"
MODEL_PATH: Path = ARTIFACTS_DIR / "model.joblib"
MODEL_METADATA_PATH: Path = ARTIFACTS_DIR / "model_metadata.json"
BASELINE_METRICS_PATH: Path = ARTIFACTS_DIR / "baseline_metrics.json"

# === AMLGUARD DAY 13 START ===
MLFLOW_DB_PATH: Path = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS_DIR: Path = PROJECT_ROOT / "mlartifacts"
MLFLOW_EXPERIMENT_NAME: str = "AMLGuard"
# === AMLGUARD DAY 13 END ===

# Runtime download fallback for the raw CSV (~476 MB, not committed).
# Mirrors the notebook's Section 2 loader.
RAW_CSV_GDRIVE_ID: str = "1359N_tsRuUtCFMWCV6BtHjDdF280rb8e"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.30

# --------------------------------------------------------------------------- #
# Schema and features
# --------------------------------------------------------------------------- #
TARGET: str = "Is Laundering"

RAW_REQUIRED_COLUMNS: tuple[str, ...] = (
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
)

# Model inputs selected in notebook Section 5.4.
CATEGORICAL_FEATURES: list[str] = ["Payment Format"]
NUMERIC_FEATURES: list[str] = ["Amount Paid", "sender_previous_tx_count"]
BINARY_FEATURES: list[str] = ["is_business_hours", "same_account"]

MODEL_FEATURES: list[str] = (
    CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES
)

# --------------------------------------------------------------------------- #
# XGBoost hyperparameters — frozen from notebook Section 7
# (cell 55 of notebooks/01_aml_pipeline.ipynb: "XGBoost — Scale Pos Weight").
# Any change here requires re-running scripts/verify_train_determinism.py and
# the full-CSV training, then comparing metrics against baseline_metrics.json.
# --------------------------------------------------------------------------- #
XGBOOST_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.08,
    "min_child_weight": 10,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# --------------------------------------------------------------------------- #
# Selected-model contract (frozen from the validated baseline)
# See artifacts/baseline_metrics.json and docs/BASELINE.md.
# --------------------------------------------------------------------------- #
FINAL_MODEL_NAME: str = "XGBoost — Scale Pos Weight"
FINAL_PRECISION_TARGET: float = 0.02
FINAL_THRESHOLD: float = 0.892163
MODEL_VERSION: str = "0.1.0"

# Non-regression gate: a refactored pipeline must clear these to be accepted.
MIN_AVERAGE_PRECISION: float = 0.035
MIN_RECALL_AT_2PCT_PRECISION: float = 0.65
SCORE_REPRODUCTION_TOLERANCE: float = 1e-6
