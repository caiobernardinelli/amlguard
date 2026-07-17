# AMLGuard Azure Data Asset and Environment

## Day 16 objective

Register the IBM AML HI-Small CSV as a versioned Azure Machine Learning Data
Asset, register a reproducible training environment, and execute a serverless
smoke-test job that can access the data and import the AMLGuard project.

## Dataset identity

| Property | Value |
|---|---|
| Data Asset | `ibm-aml-hi-small` |
| Version | `1` |
| Type | `uri_file` |
| Local file | `data/raw/HI-Small_Trans.csv` |
| Size | `475,664,283` bytes |
| Size in MiB | approximately `453.63` |
| SHA-256 | `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040` |
| Nature | Synthetic IBM AML dataset |

The raw CSV remains excluded from Git. During Data Asset registration, Azure ML
uploads the local file to the workspace's default Blob datastore and records a
versioned reference.

## Environment identity

| Property | Value |
|---|---|
| Environment | `amlguard-training` |
| Version | `1` |
| Python | `3.12` |
| Base image | `mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest` |
| Purpose | Data validation and CPU model training |

The environment captures the core AMLGuard training libraries. MLflow/Azure
tracking integration can be introduced as a new environment version when the
cloud training component requires it; version 1 remains immutable evidence.

## Register the Data Asset

From the repository root:

```cmd
az ml data create --file cloud/azure/data/ibm_aml_hi_small.yml
```

The first upload is approximately 454 MiB and can take several minutes.

Verify:

```cmd
az ml data show --name ibm-aml-hi-small --version 1 --query "{Name:name,Version:version,Type:type,Path:path}" --output table
```

## Register the environment

```cmd
az ml environment create --file cloud/azure/environment/environment.yml
```

Verify:

```cmd
az ml environment show --name amlguard-training --version 1 --query "{Name:name,Version:version,Image:image,Build:build}" --output yaml
```

Registering the specification is separate from building the final image. The
first job that uses the custom environment can take longer while Azure builds
and caches it.

## Submit the serverless smoke test

```cmd
az ml job create --file cloud/azure/jobs/day16_smoke.yml --stream
```

The job deliberately omits a named compute target, so Azure ML selects
serverless CPU compute. It performs these checks:

1. downloads `azureml:ibm-aml-hi-small:1`;
2. imports `src.config` and `src.data.load_data`;
3. verifies exact file size and SHA-256;
4. reads a 1,000-row sample;
5. validates the expected schema;
6. writes `day16_smoke_summary.json` as a tracked job output.

Expected evidence:

```json
{
  "status": "PASS",
  "data_asset": "azureml:ibm-aml-hi-small:1",
  "environment": "azureml:amlguard-training:1",
  "size_bytes": 475664283,
  "sha256": "b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040",
  "project_import": "PASS",
  "schema_validation": "PASS"
}
```

## Cloud import-path lesson

The first two cloud submissions failed with:

```text
ModuleNotFoundError: No module named 'src'
```

The repository had been uploaded correctly. The failure was caused by launching
the nested file directly with `python cloud/azure/jobs/smoke_test.py`. Direct
script execution makes the script's own directory the first import location,
while the Azure environment had not installed the AMLGuard repository as a
package.

The job now uses module execution:

```text
python -m cloud.azure.jobs.smoke_test
```

This starts Python from the uploaded code root, allowing imports such as
`src.config` and `src.data.load_data` to resolve consistently. The failed runs
remain useful operational evidence and were not caused by the Data Asset or the
registered environment.

## Successful cloud run

| Property | Value |
|---|---|
| Run ID | `happy_bone_n2gm2r19qz` |
| Azure ML status | `Completed` |
| Smoke-test status | `PASS` |
| Data Asset | `azureml:ibm-aml-hi-small:1` |
| Environment | `azureml:amlguard-training:1` |
| Project import | `PASS` |
| Schema validation | `PASS` |
| Sample rows validated | `1,000` |
| Versioned evidence | `docs/evidence/day16_smoke_summary.json` |

The first two submissions failed because the nested script was executed
directly and Python could not resolve the repository-root `src` package. After
switching the job command to module execution, the third submission completed
successfully. The failed runs remain part of the engineering history and show
that the root cause was diagnosed rather than hidden.

## Inspect the latest job

```cmd
az ml job list --max-results 5 --query "[].{Name:name,DisplayName:display_name,Status:status}" --output table
```

Use the exact job name returned by the creation command:

```cmd
az ml job show --name <JOB_NAME> --query "{Name:name,Status:status,StudioUrl:services.Studio.endpoint}" --output yaml
```

## Cost control

Serverless compute is created only for the duration of the job. The job and
environment build can create a small usage charge. No persistent compute cluster
or online endpoint is created on Day 16.

## Day 16 acceptance checklist

- [x] Data Asset `ibm-aml-hi-small:1` registered
- [x] Dataset size and SHA-256 documented
- [x] Environment `amlguard-training:1` registered
- [x] Serverless smoke-test job completed
- [x] Job accessed the registered Data Asset
- [x] Job imported the AMLGuard project
- [x] Schema validation passed
- [x] Evidence JSON produced
- [x] No persistent compute or endpoint created
