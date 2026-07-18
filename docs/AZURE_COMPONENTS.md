# AMLGuard Azure ML Components

## Day 17 objective

Create three reusable Azure Machine Learning command components with explicit
inputs and outputs. Each component is registered and tested independently before
Day 18 connects the same versions into one end-to-end pipeline.

```text
ibm-aml-hi-small:1
        |
        v
prepare_data
        |
        v
prepared_split.joblib
        |
        v
train_model
        |
        v
model.joblib
        |
        v
evaluate_model
        |
        v
metrics.json
```

## Component contracts

### `amlguard_prepare_data:1`

Input: `raw_data` (`uri_file`)

Output: `prepared_data` (`uri_folder`)

Responsibilities: schema validation, leakage-safe feature engineering, and the
deterministic stratified 70/30 split.

### `amlguard_train_model:1`

Input: `prepared_data` (`uri_folder`)

Output: `model_output` (`uri_folder`)

Responsibilities: derive `scale_pos_weight`, train the frozen weighted XGBoost
pipeline, and persist `model.joblib` plus training metadata.

### `amlguard_evaluate_model:1`

Inputs: `prepared_data` and `model_input` (`uri_folder`)

Output: `evaluation_output` (`uri_folder`)

Responsibilities: evaluate on the frozen test split, enforce the absolute
quality gate, compare tracked metrics against baseline tolerances, and write
`metrics.json`.

## Register components

```cmd
az ml component create --file cloud/azure/components/prepare_data/component.yml
az ml component create --file cloud/azure/components/train_model/component.yml
az ml component create --file cloud/azure/components/evaluate_model/component.yml
```

Verify:

```cmd
az ml component list --query "[?starts_with(name, 'amlguard_')].{Name:name,LatestVersion:latest_version}" --output table
```

## Isolated executions

Run sequentially:

```cmd
az ml job create --file cloud/azure/jobs/day17_prepare.yml --stream
az ml job create --file cloud/azure/jobs/day17_train.yml --stream
az ml job create --file cloud/azure/jobs/day17_evaluate.yml --stream
```

Do not start the next job until the previous one is `Completed`.

## Successful isolated cloud runs

| Component | Registered version | Run ID | Status |
|---|---:|---|---|
| `amlguard_prepare_data` | `1` | `lucid_bag_rblj448q2k` | `Completed` |
| `amlguard_train_model` | `1` | `nice_map_jw7r2gwc3k` | `Completed` |
| `amlguard_evaluate_model` | `1` | `helpful_snake_mk7qwbycm5` | `Completed` |

The evaluation component completed with both the absolute quality gate and the
baseline regression gate passing. The downloaded cloud evidence is versioned at
`docs/evidence/day17_evaluation_metrics.json`.

## Cost control

All isolated component runs use Azure ML serverless compute through the pipeline
setting `default_compute: azureml:serverless`. No persistent compute cluster or
online endpoint is created on Day 17.

## Day 17 acceptance checklist

- [x] `amlguard_prepare_data:1` registered
- [x] `amlguard_train_model:1` registered
- [x] `amlguard_evaluate_model:1` registered
- [x] prepare_data isolated run completed
- [x] train_model isolated run completed
- [x] evaluate_model isolated run completed
- [x] quality gate passed
- [x] regression gate passed
- [x] explicit component inputs and outputs documented
- [x] no persistent compute or endpoint created
