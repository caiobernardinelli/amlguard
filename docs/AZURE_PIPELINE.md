# AMLGuard Azure ML End-to-End Pipeline

## Day 18 execution evidence

Pipeline run:

```text
Run ID: wheat_cassava_65j58pbzkl
Display name: amlguard-day18-end-to-end
Status: Completed

prepare_data: Completed
train_model: Completed
evaluate_model: Completed

docs/evidence/day18_evaluation_metrics.json

status: PASS
quality_gate.status: PASS
regression_gate.status: PASS

Name: AMLGuard
Version: 1
Type: custom_model
Stage: Development
Source job: wheat_cassava_65j58pbzkl



## Day 18 objective

Connect the three Azure Machine Learning components validated independently on
Day 17 into one end-to-end pipeline:

```text
ibm-aml-hi-small:1
        |
        v
amlguard_prepare_data:1
        |
        v
amlguard_train_model:1
        |
        v
amlguard_evaluate_model:1
        |
        +----> evaluation_output / metrics.json
        |
        +----> model_output / model.joblib
```

The model is registered as an Azure ML model asset only after:

1. the parent pipeline status is `Completed`;
2. `metrics.json` reports `status = PASS`;
3. `quality_gate.status = PASS`;
4. `regression_gate.status = PASS`.

This keeps model promotion explicitly gated by evaluation results.

## Pipeline specification

File:

```text
cloud/azure/jobs/day18_pipeline.yml
```

The pipeline reuses:

```text
amlguard_prepare_data:1
amlguard_train_model:1
amlguard_evaluate_model:1
```

Input:

```text
azureml:ibm-aml-hi-small:1
```

Compute: Azure ML serverless.

## Run the pipeline

```cmd
az ml job create --file cloud/azure/jobs/day18_pipeline.yml --stream
```

Verify:

```cmd
az ml job show --name <RUN_ID> --query "{Name:name,Status:status,DisplayName:display_name}" --output table
```

The pipeline must be `Completed` before model promotion.

## Download evaluation evidence

```cmd
rmdir /s /q artifacts\azure_day18 2>nul
az ml job download --name <RUN_ID> --output-name evaluation_output --download-path artifacts\azure_day18
dir /s /b artifacts\azure_day18\metrics.json
```

Confirm:

```json
{
  "status": "PASS",
  "quality_gate": {
    "status": "PASS"
  },
  "regression_gate": {
    "status": "PASS"
  }
}
```

## Register the gated model

Only after the checks above pass:

```cmd
az ml model create --name AMLGuard --version 1 --type custom_model --path azureml://jobs/<RUN_ID>/outputs/model_output --tags project=AMLGuard phase=day18 gate_status=PASS
```

Verify:

```cmd
az ml model show --name AMLGuard --version 1 --query "{Name:name,Version:version,Type:type}" --output table
```

## Why registration happens after the pipeline run

The evaluation component exits non-zero when either the absolute quality gate
or baseline regression gate fails. The separate registration command is a
deliberate promotion step executed only after the successful pipeline and its
evaluation evidence have been verified.

This avoids adding Azure management credentials or Azure SDK dependencies inside
the training environment.

## Cost control

The pipeline uses Azure ML serverless compute. It creates no persistent compute
cluster and no online endpoint.

## Day 18 acceptance checklist

- [x] Day 17 component versions reused without modification
- [x] End-to-end pipeline submitted
- [x] prepare_data step completed
- [x] train_model step completed
- [x] evaluate_model step completed
- [x] parent pipeline completed
- [x] quality gate passed
- [x] regression gate passed
- [x] evaluation metrics downloaded and verified
- [x] Azure ML model `AMLGuard:1` registered only after gates passed
- [x] registered model version verified
- [x] no persistent compute or endpoint created
