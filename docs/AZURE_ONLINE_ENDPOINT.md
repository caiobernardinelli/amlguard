# AMLGuard Azure ML Managed Online Endpoint

## Day 19 objective

Deploy the gated Azure ML model asset `AMLGuard:1` behind a managed real-time
inference endpoint and validate a live cloud request.

The deployment was performed only after the Day 18 end-to-end Azure ML pipeline
completed successfully and both the quality and regression gates passed.

## Architecture

```text
Client JSON request
        |
        v
Azure ML Managed Online Endpoint
amlguard-realtime
        |
        v
blue deployment (100% traffic)
        |
        +--> AMLGuard:1 model asset
        |
        +--> amlguard-inference:1 environment
        |
        +--> score.py
        |
        v
JSON response
score + alert + threshold + model_version
```

## Azure ML resources

```text
Endpoint: amlguard-realtime
Authentication: key
Deployment: blue
Traffic: blue = 100%
Model asset: AMLGuard:1
Environment: amlguard-inference:1
Instance type: Standard_DS2_v2
```

The endpoint uses a dedicated inference environment rather than the training
environment. The deployment loads the approved `model.joblib` from the Azure ML
model asset and preserves the frozen public decision threshold.

## Files

```text
cloud/azure/online/endpoint.yml
cloud/azure/online/deployment.yml
cloud/azure/online/environment.yml
cloud/azure/online/conda.yml
cloud/azure/online/score.py
cloud/azure/online/sample_request.json
docs/evidence/day19_online_endpoint.json
```

## Register the inference environment

```cmd
az ml environment create --file cloud\azure\online\environment.yml
```

Verify:

```cmd
az ml environment show --name amlguard-inference --version 1 --query "{Name:name,Version:version,Image:image}" --output table
```

## Create the managed online endpoint

```cmd
az ml online-endpoint create --file cloud\azure\online\endpoint.yml
```

Verify:

```cmd
az ml online-endpoint show --name amlguard-realtime --query "{Name:name,State:provisioning_state,AuthMode:auth_mode}" --output table
```

Expected state:

```text
amlguard-realtime  Succeeded  key
```

## Create the deployment

```cmd
az ml online-deployment create --file cloud\azure\online\deployment.yml --all-traffic
```

Verify:

```cmd
az ml online-deployment show --name blue --endpoint-name amlguard-realtime --query "{Name:name,State:provisioning_state,InstanceType:instance_type}" --output table
```

Expected state:

```text
blue  Succeeded  Standard_DS2_v2
```

## Invoke real-time inference

```cmd
az ml online-endpoint invoke --name amlguard-realtime --request-file cloud\azure\online\sample_request.json
```

Validated response:

```json
{
  "score": 0.99112468957901,
  "alert": true,
  "threshold": 0.892163,
  "model_version": "1"
}
```

Response contract:

```text
score          probability-like XGBoost risk score
alert          true when score >= threshold
threshold      frozen operating threshold used for the decision
model_version  Azure ML model asset version serving the request
```

## Deployment validation

The Day 19 evidence confirms:

- endpoint provisioning state: `Succeeded`;
- deployment provisioning state: `Succeeded`;
- 100% of endpoint traffic routed to `blue`;
- live Azure ML inference completed successfully;
- response includes `score`, `alert`, `threshold`, and `model_version`;
- frozen serving threshold `0.892163` was preserved;
- serving response identifies model version `1`.

Sanitized evidence:

```text
docs/evidence/day19_online_endpoint.json
```

The evidence file intentionally excludes subscription IDs, endpoint keys,
credentials, and other secrets.

## Resource-provider prerequisite discovered during deployment

The subscription required the Azure resource providers used by the managed
endpoint flow to be registered before endpoint provisioning could succeed.
The deployment was retried only after the missing providers were registered.

No subscription IDs, tenant IDs, keys, or secrets belong in the repository.

## Cost control

The managed online deployment uses one `Standard_DS2_v2` instance while active.
Unlike the serverless training pipeline, an online endpoint deployment can incur
continuous compute cost while provisioned.

Delete the endpoint when it is no longer required:

```cmd
az ml online-endpoint delete --name amlguard-realtime --yes
```

Do not delete it before completing any monitoring or observability work that
depends on live endpoint traffic.

## Day 19 acceptance checklist

- [x] dedicated Azure ML inference environment registered
- [x] Managed Online Endpoint created
- [x] key authentication configured
- [x] `AMLGuard:1` deployed
- [x] `blue` deployment succeeded
- [x] 100% traffic routed to `blue`
- [x] live cloud inference succeeded
- [x] response returned `score`
- [x] response returned `alert`
- [x] response returned `threshold`
- [x] response returned `model_version`
- [x] frozen serving threshold preserved
- [x] sanitized deployment evidence captured
- [x] no secrets persisted in repository

## Day 20 monitoring upgrade

Day 20 upgraded the existing `blue` deployment in place after the Day 19
serving validation.

The monitored deployment now uses:

```text
Model asset: AMLGuard:1
Environment: amlguard-inference:2
Data collection: model_inputs + model_outputs
Sampling rate: 1.0
```

The external inference contract is unchanged.

`amlguard-inference:1` remains the environment that was originally validated
for the Day 19 deployment. Version `2` adds Azure ML model-data collection for
the monitoring workflow documented in:

```text
docs/AZURE_MONITORING.md
```

The updated deployment returned to provisioning state `Succeeded` and a
post-update smoke request reproduced the same validated response contract.

The `Standard_DS2_v2` SKU produced an Azure CLI recommendation that
`Standard_DS3_v2` is the minimum recommended general-purpose SKU. The project
keeps `Standard_DS2_v2` for this portfolio/dev deployment to limit cost; this is
not presented as a production capacity recommendation.
