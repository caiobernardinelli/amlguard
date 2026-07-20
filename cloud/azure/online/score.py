"""Azure ML managed online endpoint scoring entry point for AMLGuard."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from azureml.ai.monitoring import Collector

MODEL: Any = None
MODEL_FEATURES: list[str] = []
THRESHOLD: float = 0.0
MODEL_VERSION: str = ""
INPUTS_COLLECTOR: Collector | None = None
OUTPUTS_COLLECTOR: Collector | None = None

LOGGER = logging.getLogger("amlguard.monitoring")


def _log_collection_error(error: Exception) -> None:
    """Log monitoring failures without interrupting real-time inference."""
    LOGGER.warning("Azure ML model data collection failed: %s", error)


def init() -> None:
    """Load the model and initialize production inference data collectors."""
    global MODEL, MODEL_FEATURES, THRESHOLD, MODEL_VERSION
    global INPUTS_COLLECTOR, OUTPUTS_COLLECTOR

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

    INPUTS_COLLECTOR = Collector(
        name="model_inputs",
        on_error=_log_collection_error,
    )
    OUTPUTS_COLLECTOR = Collector(
        name="model_outputs",
        on_error=_log_collection_error,
    )


def run(raw_data: str | dict[str, Any]) -> dict[str, Any]:
    """Score one AML transaction and collect tabular monitoring telemetry."""
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

    collection_context = None
    if INPUTS_COLLECTOR is not None:
        collection_context = INPUTS_COLLECTOR.collect(frame)

    score = float(MODEL.predict_proba(frame)[0, 1])
    alert = bool(score >= THRESHOLD)

    output_frame = pd.DataFrame(
        [
            {
                "score": score,
                "alert": alert,
                "threshold": THRESHOLD,
                "model_version": MODEL_VERSION,
            }
        ]
    )

    if OUTPUTS_COLLECTOR is not None:
        OUTPUTS_COLLECTOR.collect(output_frame, collection_context)

    return {
        "score": score,
        "alert": alert,
        "threshold": THRESHOLD,
        "model_version": MODEL_VERSION,
    }
