# AMLGuard Azure ML Monitoring and Drift

## Day 20 objective

Extend the live Azure ML Managed Online Endpoint with production-style
observability for both service health and model behaviour.

Day 20 adds:

- Azure Monitor operational metrics;
- Azure ML Model Data Collector for `model_inputs` and `model_outputs`;
- sanitized monitoring evidence;
- synthetic monitoring traffic;
- data-drift analysis;
- prediction-drift analysis;
- alert-rate monitoring;
- a documented retraining and promotion strategy.

The monitoring workflow is implemented and validated, but the Day 20 drift
batch is synthetic monitoring traffic. It must not be interpreted as evidence
of real production drift.

## Architecture

```text
Client
  |
  v
Azure ML Managed Online Endpoint
amlguard-realtime
  |
  v
blue deployment
  |
  +--> AMLGuard:1
  |
  +--> amlguard-inference:2
  |
  +--> score.py
  |      |
  |      +--> model_inputs Collector
  |      |
  |      +--> model_outputs Collector
  |
  +--> JSON scoring response
  |
  +--> Azure Monitor
  |      |
  |      +--> RequestsPerMinute
  |      +--> RequestLatency
  |      +--> RequestLatency_P50
  |      +--> RequestLatency_P95
  |      +--> RequestLatency_P99
  |
  +--> workspace blob storage
         |
         +--> modelDataCollector/.../model_inputs/
         +--> modelDataCollector/.../model_outputs/
```

## Operational observability

Azure Monitor exposed native endpoint metrics including:

- `RequestsPerMinute`;
- `RequestLatency`;
- `RequestLatency_P50`;
- `RequestLatency_P95`;
- `RequestLatency_P99`;
- data-collection event/error metrics.

A controlled traffic burst was observed on the `blue` deployment with:

- HTTP status code `200`;
- status class `2xx`;
- model status code `200`.

Observed server-side latency samples for the controlled test window were:

| Metric | Observed values |
| --- | --- |
| RequestLatency | 15.5 ms, 16.33 ms |
| P50 | 25 ms, 18 ms |
| P95 | 26.5 ms, 18 ms |
| P99 | 26.5 ms, 18 ms |

These measurements prove that endpoint telemetry is working. They are not an
SLA claim or a load-test benchmark.

Sanitized evidence:

```text
docs/evidence/day20_operational_observability.json
```

## Model Data Collector

The inference environment was upgraded from `amlguard-inference:1` to
`amlguard-inference:2` with Azure ML monitoring support.

The deployment enables:

```yaml
data_collector:
  sampling_rate: 1.0
  rolling_rate: Hour
  collections:
    model_inputs:
      enabled: "true"
    model_outputs:
      enabled: "true"
```

The scoring script collects:

### model_inputs

- `Payment Format`
- `Amount Paid`
- `sender_previous_tx_count`
- `is_business_hours`
- `same_account`

### model_outputs

- `score`
- `alert`
- `threshold`
- `model_version`

The external scoring response contract remains unchanged.

Azure ML registered the corresponding data assets:

```text
amlguard-realtime-blue-model_inputs:1
amlguard-realtime-blue-model_outputs:1
```

Collected JSONL data was validated structurally:

- 71 input records;
- 71 output records;
- 71 matched input/output pairs using `correlationid`;
- zero input-only records;
- zero output-only records;
- valid expected input and output schemas.

Raw collected JSONL files are intentionally not committed.

Sanitized evidence:

```text
docs/evidence/day20_data_collection_validation.json
```

## Synthetic monitoring traffic

A deterministic monitoring batch sent 70 varied synthetic requests to the live
endpoint across seven payment formats.

Results:

- 70 requested;
- 70 succeeded;
- 0 failed;
- isolated synthetic-batch alert rate: `2.86%` (`2/70`);
- score range: approximately `0.000032` to `0.964067`.

Including the earlier post-deployment smoke request, the Data Collector held
71 paired records. The smoke request was also an alert, which explains why the
collected 71-record population has a `4.23%` alert rate rather than the
`2.86%` rate of the isolated 70-record batch.

Sanitized evidence:

```text
docs/evidence/day20_model_monitoring_traffic.json
```

## Data and prediction drift

The reference population is the deterministic frozen AMLGuard test split.

The current population is the 71-record batch captured by Azure ML Model Data
Collector.

Population Stability Index (PSI) is used as a project-level monitoring
heuristic:

| PSI | Signal |
| --- | --- |
| `< 0.10` | Low |
| `0.10 <= PSI < 0.25` | Moderate |
| `>= 0.25` | Significant |

Day 20 results:

| Signal | PSI | Classification |
| --- | ---: | --- |
| Payment Format | 0.666304 | Significant |
| Amount Paid | 3.304550 | Significant |
| sender_previous_tx_count | 3.452996 | Significant |
| is_business_hours | 0.022415 | Low |
| same_account | 0.298780 | Significant |
| Prediction score distribution | 0.424604 | Significant |

Alert-rate comparison:

```text
Reference alert rate: 3.42%
Current collected alert rate: 4.23%
Absolute change: +0.81 percentage points
```

The prediction distribution changed substantially even though the alert rate
changed only modestly. This demonstrates why production model observability
should monitor score distributions and input distributions rather than relying
only on alert counts.

Because the current batch is deliberately synthetic, the significant drift
signal validates the monitoring workflow only. It is not evidence that the
real AMLGuard production population drifted.

Sanitized evidence:

```text
docs/evidence/day20_drift_report.json
```

## Retraining strategy

Drift is an investigation trigger, not an automatic retraining trigger.

The initial AMLGuard policy requires:

1. representative production traffic;
2. persistence across three consecutive monitoring windows;
3. enough ground-truth labels for meaningful evaluation;
4. evidence of model-quality degradation or a justified material population
   change before creating a retraining candidate.

Synthetic smoke tests and monitoring batches are excluded from production
retraining decisions.

Any candidate must still pass the frozen AMLGuard quality gates:

```text
Average Precision >= 0.035
Precision operating target = 0.02
Recall at the operating point >= 0.65
```

A drift signal cannot bypass model-quality, regression, serving, or
post-deployment validation gates.

Full policy:

```text
docs/RETRAINING_STRATEGY.md
```

Sanitized policy evidence:

```text
docs/evidence/day20_retraining_strategy.json
```

## Reproducible monitoring scripts

```text
scripts/monitoring/generate_day20_model_monitoring_traffic.py
scripts/monitoring/validate_day20_data_collection.py
scripts/monitoring/generate_day20_drift_report.py
```

The drift and validation scripts expect collected JSONL files to be materialized
locally under `.day20_collected/`. That directory is temporary working data and
must not be committed.

## Day 20 acceptance checklist

- [x] Azure Monitor endpoint metrics discovered
- [x] successful 2xx traffic observed
- [x] RequestLatency observed
- [x] P50/P95/P99 latency observed
- [x] inference environment upgraded to `amlguard-inference:2`
- [x] Model Data Collector enabled
- [x] `model_inputs` data asset created
- [x] `model_outputs` data asset created
- [x] collected JSONL data persisted
- [x] input/output schemas validated
- [x] 71/71 records paired by correlation ID
- [x] synthetic monitoring traffic completed 70/70
- [x] alert rate measured
- [x] score distribution measured
- [x] data drift calculated
- [x] prediction drift calculated
- [x] retraining policy documented
- [x] frozen model-quality gates preserved
- [x] sanitized evidence captured
- [x] no Azure credentials or subscription IDs committed
