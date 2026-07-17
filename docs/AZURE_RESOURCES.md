# AMLGuard Azure Resource Map

## Purpose

This document records the Azure foundation created for the AMLGuard MLOps
portfolio project. It contains resource names and operational settings only.
Passwords, tokens, access keys, client secrets, and connection strings must
never be committed.

Created on: 2026-07-16  
Environment: `dev`  
Region: `brazilsouth`

## Resource hierarchy

```text
Azure subscription
└── Resource Group: rg-amlguard-dev-brazilsouth
    ├── Azure Machine Learning Workspace: mlw-amlguard-dev
    ├── Storage Account: mlwamlgustorage7fec16600
    ├── Key Vault: mlwamlgukeyvaultaffe2c4c
    ├── Log Analytics Workspace: mlwamlgulogalyti9c0b0278
    └── Application Insights: mlwamlguinsights652b3e8f
```

Azure Container Registry is not provisioned yet. The provider is registered,
but a registry will only be created when a custom environment or deployment
requires it.

## Resource purposes

| Resource | Purpose |
|---|---|
| Resource Group | Lifecycle and cost boundary for the AMLGuard cloud lab |
| Azure ML Workspace | Jobs, data assets, environments, models, endpoints, and MLflow integration |
| Storage Account | Default workspace datastore and job artifacts |
| Key Vault | Workspace-managed secrets and credentials |
| Log Analytics | Central telemetry store |
| Application Insights | Application and endpoint observability |

## Workspace configuration

| Setting | Value |
|---|---|
| Display name | `AMLGuard Development` |
| Workspace name | `mlw-amlguard-dev` |
| Resource group | `rg-amlguard-dev-brazilsouth` |
| Location | `brazilsouth` |
| Identity | System-assigned managed identity |
| Public network access | Enabled |
| Managed network isolation | Disabled |
| System datastore authentication | Access key |
| Compute instances | None |
| Compute clusters | None |
| Online endpoints | None |
| Deployments | None |

Public network access is acceptable for this isolated portfolio development
workspace. It is not a production security recommendation.

## Azure CLI defaults

The local Azure CLI was configured with:

```cmd
az configure --defaults group=rg-amlguard-dev-brazilsouth workspace=mlw-amlguard-dev location=brazilsouth
```

`az ml` by itself only displays an error because it requires a subcommand.
Use a command such as:

```cmd
az ml workspace show --name mlw-amlguard-dev --resource-group rg-amlguard-dev-brazilsouth --output table
```

## Inspect current resources

```cmd
az resource list --resource-group rg-amlguard-dev-brazilsouth --query "[].{Name:name,Type:type,Location:location}" --output table
```

## Credential policy

- Local authentication uses `az login`.
- `.env` is excluded by `.gitignore`.
- `.env.example` contains placeholders and non-secret resource names only.
- No password, token, access key, connection string, or client secret belongs in Git.
- Future GitHub Actions authentication should prefer OpenID Connect instead of a stored client secret.

## Cost guardrails

- Do not create compute until the training job is ready.
- Compute clusters must use autoscaling with a minimum of zero nodes.
- Delete failed deployments after investigation.
- Do not leave managed online endpoints running unnecessarily.
- Review Azure Cost Management after every cloud session.
- Keep all AMLGuard resources inside this dedicated resource group.

The workspace dependencies can generate usage-based charges. Compute resources
and online deployments are the primary cost risks for the next phases.

## Cleanup

At the end of the cloud phase, deleting the dedicated resource group removes
the workspace and the resources recorded above:

```cmd
az group delete --name rg-amlguard-dev-brazilsouth --yes --no-wait
```

This command is destructive and must not be run while the project still needs
the Azure resources.

## Day 15 acceptance evidence

- [x] Azure subscription enabled and selected
- [x] Azure ML CLI extension installed
- [x] Required resource providers registered
- [x] Resource Group created in `brazilsouth`
- [x] Azure ML Workspace created and accessible
- [x] Storage, Key Vault, Log Analytics, and Application Insights provisioned
- [x] Azure CLI defaults configured
- [x] Real credentials kept outside Git
- [x] Resource map documented
- [x] No compute or endpoint created
