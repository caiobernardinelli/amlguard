"""Azure ML managed online endpoint scoring entry point for AMLGuard."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

MODEL: Any = None
MODEL_FEATURES: list[str] = []
THRESHOLD: float = 0.0
MODEL_VERSION: str = ""


def init() -> None:
    """Load the registered AMLGuard model once when the container starts."""
    global MODEL, MODEL_FEATURES, THRESHOLD, MODEL_VERSION

    model_root = Path(os.environ["AZUREML_MODEL_DIR"])

    model_candidates = list(model_root.rglob("model.joblib"))
    if len(model_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one model.joblib under {model_root}, "
            f"found {len(model_candidates)}."
        )

    model_path = model_candidates[0]
    MODEL = joblib.load(model_path)

    metadata_path = model_path.parent / "training_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Training metadata not found at {metadata_path}."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    MODEL_FEATURES = list(metadata["features"])

    THRESHOLD = float(os.environ["AMLGUARD_THRESHOLD"])
    MODEL_VERSION = os.environ["AMLGUARD_MODEL_VERSION"]


def run(raw_data: str | dict[str, Any]) -> dict[str, Any]:
    """Score one AML transaction and return the cloud inference contract."""
    if MODEL is None:
        raise RuntimeError("AMLGuard model is not initialized.")

    payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data

    if not isinstance(payload, dict):
        raise ValueError("Request payload must be a JSON object.")

    record = payload.get("data")
    if not isinstance(record, dict):
        raise ValueError(
            "Request payload must contain a 'data' object with model features."
        )

    missing_features = [
        feature for feature in MODEL_FEATURES if feature not in record
    ]
    if missing_features:
        raise ValueError(
            f"Missing required feature(s): {missing_features}."
        )

    frame = pd.DataFrame(
        [{feature: record[feature] for feature in MODEL_FEATURES}]
    )

    score = float(MODEL.predict_proba(frame)[0, 1])

    return {
        "score": score,
        "alert": bool(score >= THRESHOLD),
        "threshold": THRESHOLD,
        "model_version": MODEL_VERSION,
    }
