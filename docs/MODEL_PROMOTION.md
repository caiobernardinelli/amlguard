# AMLGuard model versioning and promotion

## Purpose

Day 14 packages the persisted sklearn pipeline as an MLflow PyFunc model with:

- the preprocessing and XGBoost pipeline;
- the frozen serving threshold from `src.config.FINAL_THRESHOLD`;
- an input and output signature;
- a representative input example;
- pinned inference dependencies;
- a registered model version;
- a candidate alias;
- a reload-and-predict smoke test.

Packaging does not retrain the model and does not change the frozen baseline.

## Package a candidate

From the repository root:

```cmd
python -m src.models.mlflow_model
```

The command creates a new registered version under `AMLGuard`, assigns the
`candidate` alias, reloads `models:/AMLGuard@candidate`, and confirms prediction
parity with the local persisted pipeline.

The local evidence is written to:

```text
artifacts/mlflow_model_info.json
```

This file is operational evidence and remains outside Git through the existing
artifact ignore rules.

## Review before promotion

A candidate must satisfy all of the following:

1. `baseline_gate` is `PASS`;
2. `validation_status` is `passed`;
3. the registered model has an input and output signature;
4. the input example is visible in MLflow;
5. loading by candidate alias succeeds;
6. the smoke-test scores match the local pipeline;
7. the threshold matches `src.config.FINAL_THRESHOLD`.

The threshold stored in historical training metadata can differ slightly from
the frozen public serving threshold. Packaging uses the public serving contract
so the MLflow model and FastAPI produce the same alert decision.

## Promote candidate to champion

After human review:

```cmd
python -m src.models.mlflow_model --promote-candidate
```

Promotion moves the `champion` alias to the current validated candidate. It does
not copy or retrain the model.

Load the promoted version with:

```python
import mlflow
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.pyfunc.load_model("models:/AMLGuard@champion")
predictions = model.predict(
    pd.DataFrame(
        [
            {
                "Payment Format": "Wire",
                "Amount Paid": 75000.0,
                "sender_previous_tx_count": 3,
                "is_business_hours": 0,
                "same_account": 0,
            }
        ]
    )
)
print(predictions)
```

## Rollback

Aliases are movable references. To roll back, assign `champion` to a previously
validated version in the MLflow UI or with `MlflowClient`.

Never delete historical versions merely to roll back. Version history is part of
the audit trail.
