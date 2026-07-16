"""Package the persisted AMLGuard pipeline as a versioned MLflow model.

The MLflow model preserves the public prediction contract: probability score,
frozen alert threshold, decision, and application model version.

CLI
---
Package and register the current persisted model as the candidate version::

    python -m src.models.mlflow_model

Promote the validated candidate alias to champion after human review::

    python -m src.models.mlflow_model --promote-candidate
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FINAL_THRESHOLD,
    MLFLOW_CANDIDATE_ALIAS,
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_ARTIFACT_NAME,
    MLFLOW_MODEL_INFO_PATH,
    MLFLOW_REGISTERED_MODEL_NAME,
    MODEL_FEATURES,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    PROJECT_ROOT,
)
from src.models.tracking import resolve_tracking_uri

try:
    import mlflow
except ImportError:
    mlflow = None

_PYTHON_MODEL_BASE = mlflow.pyfunc.PythonModel if mlflow is not None else object


class AMLGuardPythonModel(_PYTHON_MODEL_BASE):
    """Portable MLflow wrapper around the persisted sklearn pipeline."""

    def __init__(
        self,
        threshold: float,
        model_version: str,
        required_features: list[str],
    ) -> None:
        self.threshold = float(threshold)
        self.model_version = str(model_version)
        self.required_features = list(required_features)
        self._pipeline: Any | None = None

    def load_context(self, context: Any) -> None:
        """Load the packaged pipeline artifact once per model process."""

        self._pipeline = joblib.load(context.artifacts["pipeline"])

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Return scores and alert decisions using the frozen threshold."""

        del params
        if self._pipeline is None:
            if context is None:
                raise RuntimeError("MLflow model context is required.")
            self.load_context(context)

        frame = _validate_model_input(
            model_input,
            required_features=self.required_features,
        )
        scores = self._pipeline.predict_proba(frame)[:, 1]

        return pd.DataFrame(
            {
                "risk_score": scores.astype(float),
                "is_alert": scores >= self.threshold,
                "threshold": np.full(len(scores), self.threshold),
                "model_version": np.full(
                    len(scores),
                    self.model_version,
                    dtype=object,
                ),
            }
        )


def _require_mlflow() -> Any:
    """Return MLflow or raise a direct installation instruction."""

    if mlflow is None:
        raise RuntimeError(
            "MLflow is not installed. Run "
            '`python -m pip install -e ".[mlops]"` first.'
        )
    return mlflow


def _validate_model_input(
    model_input: Any,
    *,
    required_features: list[str] | None = None,
) -> pd.DataFrame:
    """Return an ordered DataFrame or name every missing feature."""

    features = list(required_features or MODEL_FEATURES)
    frame = (
        model_input.copy()
        if isinstance(model_input, pd.DataFrame)
        else pd.DataFrame(model_input)
    )
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError(
            f"Input is missing required feature(s): {missing}. "
            f"Expected features: {features}."
        )
    return frame.loc[:, features]


def build_input_example() -> pd.DataFrame:
    """Create a minimal representative example with stable pandas dtypes."""

    return pd.DataFrame(
        {
            "Payment Format": pd.Series(
                ["Wire", "ACH"],
                dtype="object",
            ),
            "Amount Paid": pd.Series(
                [75_000.0, 1_250.0],
                dtype="float64",
            ),
            "sender_previous_tx_count": pd.Series(
                [3, 1],
                dtype="int64",
            ),
            "is_business_hours": pd.Series(
                [0, 1],
                dtype="int64",
            ),
            "same_account": pd.Series(
                [0, 1],
                dtype="int64",
            ),
        }
    ).loc[:, MODEL_FEATURES]


def _predict_with_pipeline(
    pipeline: Any,
    model_input: pd.DataFrame,
    *,
    threshold: float = FINAL_THRESHOLD,
    model_version: str = MODEL_VERSION,
) -> pd.DataFrame:
    """Produce the same output contract without invoking MLflow."""

    frame = _validate_model_input(model_input)
    scores = pipeline.predict_proba(frame)[:, 1]
    return pd.DataFrame(
        {
            "risk_score": scores.astype(float),
            "is_alert": scores >= float(threshold),
            "threshold": np.full(len(scores), float(threshold)),
            "model_version": np.full(
                len(scores),
                str(model_version),
                dtype=object,
            ),
        }
    )


def _pip_requirements(metadata: dict[str, Any]) -> list[str]:
    """Pin the inference environment stored with the MLflow model."""

    library_versions = metadata["library_versions"]
    return [
        f"mlflow=={version('mlflow')}",
        f"numpy=={library_versions['numpy']}",
        f"pandas=={library_versions['pandas']}",
        f"scikit-learn=={library_versions['scikit-learn']}",
        f"xgboost=={library_versions['xgboost']}",
        f"joblib=={library_versions['joblib']}",
    ]


def _find_registered_version(
    client: Any,
    registered_model_name: str,
    run_id: str,
) -> str:
    """Find the model version created by the current packaging run."""

    versions = list(
        client.search_model_versions(
            f"name = '{registered_model_name}'"
        )
    )
    matching = [
        item
        for item in versions
        if getattr(item, "run_id", None) == run_id
    ]
    candidates = matching or versions
    if not candidates:
        raise RuntimeError(
            f"No registered version found for {registered_model_name}."
        )
    return str(max(candidates, key=lambda item: int(item.version)).version)


def _assert_prediction_parity(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
) -> None:
    """Fail when a loaded MLflow model changes the serving contract."""

    expected_columns = [
        "risk_score",
        "is_alert",
        "threshold",
        "model_version",
    ]
    if list(observed.columns) != expected_columns:
        raise RuntimeError(
            "Loaded MLflow model returned an unexpected output schema: "
            f"{list(observed.columns)}."
        )

    if not np.allclose(
        expected["risk_score"].to_numpy(),
        observed["risk_score"].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Loaded MLflow model scores differ from the local pipeline."
        )

    for column in ("is_alert", "threshold", "model_version"):
        if expected[column].tolist() != observed[column].tolist():
            raise RuntimeError(
                f"Loaded MLflow model changed output column {column!r}."
            )


def package_existing_model(
    *,
    tracking_uri: str | None = None,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    registered_model_name: str = MLFLOW_REGISTERED_MODEL_NAME,
) -> dict[str, Any]:
    """Package, register, alias, reload, and smoke-test the persisted model."""

    required_paths = (MODEL_PATH, MODEL_METADATA_PATH)
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Required model artifacts are missing: {missing_text}"
        )

    mlflow_module = _require_mlflow()
    resolved_uri = resolve_tracking_uri(tracking_uri)
    mlflow_module.set_tracking_uri(resolved_uri)
    mlflow_module.set_registry_uri(resolved_uri)
    mlflow_module.set_experiment(experiment_name)

    metadata = json.loads(
        MODEL_METADATA_PATH.read_text(encoding="utf-8")
    )
    if metadata["baseline_gate"]["status"] != "PASS":
        raise RuntimeError(
            "Only a model with baseline_gate=PASS can be registered."
        )

    pipeline = joblib.load(MODEL_PATH)
    input_example = build_input_example()
    expected_output = _predict_with_pipeline(
        pipeline,
        input_example,
        threshold=FINAL_THRESHOLD,
        model_version=MODEL_VERSION,
    )
    signature = mlflow_module.models.infer_signature(
        input_example,
        expected_output,
    )

    commit = str(metadata.get("git_commit", "unknown"))
    short_commit = commit[:7] if commit != "unknown" else "unknown"
    run_name = f"package-{MODEL_VERSION}-{short_commit}"

    with mlflow_module.start_run(run_name=run_name) as run:
        run_id = str(run.info.run_id)
        mlflow_module.log_params(
            {
                "application_model_version": MODEL_VERSION,
                "decision_threshold": FINAL_THRESHOLD,
                "registered_model_name": registered_model_name,
                "source_training_run_id": metadata.get(
                    "mlflow_run_id",
                    "unknown",
                ),
            }
        )
        mlflow_module.set_tags(
            {
                "project": "AMLGuard",
                "stage": "model_packaging",
                "baseline_gate": "PASS",
                "validation_status": "passed",
                "git_commit": commit,
                "threshold_source": "src.config.FINAL_THRESHOLD",
            }
        )

        model_info = mlflow_module.pyfunc.log_model(
            name=MLFLOW_MODEL_ARTIFACT_NAME,
            python_model=AMLGuardPythonModel(
                threshold=FINAL_THRESHOLD,
                model_version=MODEL_VERSION,
                required_features=MODEL_FEATURES,
            ),
            artifacts={"pipeline": str(MODEL_PATH)},
            code_paths=[str(PROJECT_ROOT / "src")],
            signature=signature,
            input_example=input_example,
            pip_requirements=_pip_requirements(metadata),
            registered_model_name=registered_model_name,
            metadata={
                "application_model_version": MODEL_VERSION,
                "decision_threshold": FINAL_THRESHOLD,
                "source_training_run_id": metadata.get(
                    "mlflow_run_id",
                    "unknown",
                ),
                "baseline_gate": "PASS",
            },
        )

    client = mlflow_module.MlflowClient()
    registered_version = _find_registered_version(
        client,
        registered_model_name,
        run_id,
    )
    client.set_model_version_tag(
        registered_model_name,
        registered_version,
        "validation_status",
        "passed",
    )
    client.set_model_version_tag(
        registered_model_name,
        registered_version,
        "application_model_version",
        MODEL_VERSION,
    )
    client.set_registered_model_alias(
        registered_model_name,
        MLFLOW_CANDIDATE_ALIAS,
        int(registered_version),
    )

    registered_uri = (
        f"models:/{registered_model_name}@{MLFLOW_CANDIDATE_ALIAS}"
    )
    loaded_model = mlflow_module.pyfunc.load_model(registered_uri)
    observed_output = loaded_model.predict(input_example)
    _assert_prediction_parity(expected_output, observed_output)

    result = {
        "packaging_run_id": run_id,
        "logged_model_uri": str(model_info.model_uri),
        "registered_model_name": registered_model_name,
        "registered_model_version": registered_version,
        "alias": MLFLOW_CANDIDATE_ALIAS,
        "registered_model_uri": registered_uri,
        "application_model_version": MODEL_VERSION,
        "decision_threshold": FINAL_THRESHOLD,
        "signature_inputs": list(input_example.columns),
        "signature_outputs": list(expected_output.columns),
        "smoke_test": "PASS",
    }
    MLFLOW_MODEL_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    MLFLOW_MODEL_INFO_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def promote_candidate_to_champion(
    *,
    tracking_uri: str | None = None,
    registered_model_name: str = MLFLOW_REGISTERED_MODEL_NAME,
) -> str:
    """Promote the validated candidate version to the champion alias."""

    mlflow_module = _require_mlflow()
    resolved_uri = resolve_tracking_uri(tracking_uri)
    mlflow_module.set_tracking_uri(resolved_uri)
    mlflow_module.set_registry_uri(resolved_uri)

    client = mlflow_module.MlflowClient()
    candidate = client.get_model_version_by_alias(
        registered_model_name,
        MLFLOW_CANDIDATE_ALIAS,
    )
    validation_status = candidate.tags.get("validation_status")
    if validation_status != "passed":
        raise RuntimeError(
            "Candidate promotion requires validation_status=passed."
        )

    candidate_uri = (
        f"models:/{registered_model_name}@{MLFLOW_CANDIDATE_ALIAS}"
    )
    loaded_model = mlflow_module.pyfunc.load_model(candidate_uri)
    smoke_output = loaded_model.predict(build_input_example())
    if smoke_output.empty:
        raise RuntimeError("Candidate model smoke test returned no rows.")

    client.set_registered_model_alias(
        registered_model_name,
        MLFLOW_CHAMPION_ALIAS,
        int(candidate.version),
    )
    return str(candidate.version)


def main() -> int:
    """Package the model or promote the current candidate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI; defaults to the local SQLite backend.",
    )
    parser.add_argument(
        "--registered-model-name",
        default=MLFLOW_REGISTERED_MODEL_NAME,
        help="Registered model name.",
    )
    parser.add_argument(
        "--promote-candidate",
        action="store_true",
        help="Assign the champion alias to the validated candidate.",
    )
    args = parser.parse_args()

    try:
        if args.promote_candidate:
            version_number = promote_candidate_to_champion(
                tracking_uri=args.tracking_uri,
                registered_model_name=args.registered_model_name,
            )
            print("Candidate promoted successfully.")
            print(
                f"Champion URI: models:/"
                f"{args.registered_model_name}@"
                f"{MLFLOW_CHAMPION_ALIAS}"
            )
            print(f"Registered version: {version_number}")
        else:
            result = package_existing_model(
                tracking_uri=args.tracking_uri,
                registered_model_name=args.registered_model_name,
            )
            print("MLflow model packaged successfully.")
            print(
                "Registered model: "
                f"{result['registered_model_name']}"
            )
            print(
                "Registered version: "
                f"{result['registered_model_version']}"
            )
            print(f"Candidate URI: {result['registered_model_uri']}")
            print(f"Signature: {result['signature_inputs']}")
            print(f"Outputs: {result['signature_outputs']}")
            print(f"Smoke test: {result['smoke_test']}")
            print(
                "Review the candidate before running "
                "`python -m src.models.mlflow_model "
                "--promote-candidate`."
            )
    except (
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
