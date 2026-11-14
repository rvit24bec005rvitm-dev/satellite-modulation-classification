# Deploying and recovering SatModClass

The college project runs locally without a hosting account. If hosting is wanted later, the optional prepared target is one Render Docker web service deploying **main**, with a persistent disk and a shared operator token. Creating it requires account authorization and approval of the paid service. No cloud service was provisioned for the college submission. A local container, passing CI, or a GitHub push is not proof of a live deployment.

## Connect the account

Open the [plugin directory](https://chatgpt.com/plugins), find Render, install it, and complete the account authorization prompt. The account must have access to the intended Render workspace. Render also needs access to this GitHub repository. See [Render account integration](https://render.com/docs/mcp-server) and [Git provider access](https://render.com/docs/git-provider).

If using the dashboard directly, this [prepared main-branch Blueprint](https://render.com/deploy?repo=https://github.com/rvit24bec005rvitm-dev/satellite-modulation-classification/tree/main) opens a review of `render.yaml`. Review the workspace, region, compute, disk and secret before creating the service. This link does not create anything by itself. Do not put account API keys or workspace tokens into Git, screenshots, public issues or query parameters.

## Concrete service configuration

| Setting | Prepared value |
| --- | --- |
| Runtime | Multi-stage Docker image, bundled React frontend and real FastAPI API |
| Source branch | `main` |
| Compute | `0.5c-512mb`: 0.5 CPU, 512 MiB RAM |
| Replicas / Uvicorn workers | Exactly 1 / 1 |
| Persistent disk | 2 GB at `/data` |
| Workspace directory | `/data/workspace` |
| Readiness path | `/api/ready` |
| Access | `SATMOD_PUBLIC=1`, private `SATMOD_ACCESS_TOKEN` |
| Runtime identity | UID/GID 10001, after mount-directory initialization |

As checked on 15 November 2026, [Render pricing](https://render.com/pricing) lists the compute at **US$7/month** and persistent disks at **US$0.25/GB/month**: approximately **US$7.50/month** for this configuration, before usage, taxes or optional workspace features. Recheck the provider's final quote before provisioning. [Current compute IDs](https://render.com/docs/compute-plans) use the identifier above.

The actual all-model Linux container passed a bounded workload at 512 MiB / 0.5 CPU, with peak server RSS about 202 MiB. This supports the selected initial plan; it is not an arbitrary-file-size or concurrent-load capacity guarantee. The evidence is in [the completion record](../integration/completion/REPORT.md). The default runtime needs no TensorFlow. Preserve the separate synthetic-model labels and experimental limitations.

Render passes declared Docker build arguments from service variables. `RUNTIME_USER=root` enables only the narrow `/data` mount-directory bootstrap; the entrypoint drops supplementary groups and changes to UID/GID 10001 before readiness, model loading or serving requests. It does not recursively change ownership of restored files. Docker's default data path stays `/data` for existing local-volume compatibility; the new Render target uses its child `/data/workspace` to allow atomic offline restoration.

Set a long, randomly generated ASCII token in Render's secret settings; `sync: false` intentionally leaves it outside the repository. Invalid or missing public access configuration prevents startup. Keep the browser and API on the same HTTPS origin. All operators using this token share recordings, jobs, datasets and history. Locking clears tab credentials, private queries and active capture state. Token verification and downloads use request headers, never URLs.

## Verify the actual deployment

Use the provider-returned HTTPS origin. Export the operator token privately in your shell, then run:

```bash
python scripts/check_deployment.py https://YOUR-SERVICE.onrender.com --wait-seconds 90
python scripts/check_workspace_deployment.py https://YOUR-SERVICE.onrender.com --state-file /private/operator-checks/workspace-state.json
```

The state file is private operator evidence and must stay outside Git. Verify the unlock screen, a rejected wrong token, a real upload/classification, and all four actual model IDs. `/api/access`, `/api/health` and `/api/ready` are public and reveal no token or recordings; other API endpoints require `X-Satmod-Token` in public mode.

Restart or redeploy the service with the **same persistent disk**, then run:

```bash
python scripts/check_workspace_deployment.py https://YOUR-SERVICE.onrender.com --state-file /private/operator-checks/workspace-state.json --verify-existing
```

This checks saved capture, analysis, dataset, batch and evaluation data against the recorded state. A writable directory alone does not establish durable storage. Only paths under the attached disk survive replacement. Render disk deployments briefly stop the old instance; see [persistent disk behavior](https://render.com/docs/disks).

## Private backup and migration

Stop every process writing the source workspace before backup, or independently quiesce every writer and use `--quiesced`. The flag acknowledges this condition; the CLI does not stop a service. Unfinished jobs and unexpected files are refused. An SQLite backup alone is insufficient because raw captures and report artifacts live beside it.

```bash
python scripts/workspace_backup.py backup --source /absolute/private-workspace --output /private/operator-backups/workspace.tar --stopped
python scripts/workspace_backup.py restore --archive /private/operator-backups/workspace.tar --destination /private/restored-workspace --stopped
```

The archive is private mode 0600; restored directories are 0700 and files 0600. Archives are unencrypted and contain recordings/history, so retain them in private operator storage outside the repository and source workspace. Transfer them through an authenticated private channel. Never publish them as source assets, releases or downloadable public links.

The tool consolidates SQLite WAL through SQLite's backup API, checks metadata and artifact references, streams SHA-256 checksums, and validates a sibling staging directory before atomic publication. It rejects path traversal, links, duplicate/unknown members, oversized archives, checksum failures and overwrites. Default bounds are 1 GiB canonical data and 10,002 archive entries. Models and credentials are provisioned separately and are not inside a workspace backup. Valid historical analyses referencing deleted captures remain supported.

Restore to a **new child of the mounted disk**, such as `/data/workspace-restored`, while the service is stopped. Do not rename over `/data` itself: it is a mountpoint. Run the operator CLI **as service UID/GID 10001**, so the private files remain readable after privilege drop. The image includes `/app/scripts/workspace_backup.py`; use the provider's authenticated operator access rather than a public restore endpoint. Confirm the provider's stopped-service access method before scheduling a migration. If that method is unavailable, complete a controlled migration through a temporary maintenance entrypoint, under the same service identity, before starting the API.

After validation, set `SATMOD_DATA_DIR` to the restored child, start the service, verify readiness and compare record counts plus reports. Keep the original directory/archive until the deployed copy survives a restart. On Windows, use an unused destination; replacement of an existing empty directory is supported only on POSIX.

The existing local workspace was backed up and independently restored: all **294 analyses**, all metadata, and all 294 report hashes matched. That private archive remains outside Git. This local proof does not claim a completed transfer to Render.

## Operations and rollback

Keep one service instance and one worker; the in-process queue does not coordinate across replicas. Readiness checks storage and a real CNN forward pass without adding history, coalescing public probes briefly to bound repeated work. Live logs include request timing and safe error codes without capture content or tokens.

Before a release, take a private consistent workspace backup. Redeploy a previously verified commit on **main** if application rollback is needed; no force push or history rewrite is required. Code rollback does not restore data. The backup reader refuses a schema newer than it supports. For data recovery, stop writers, restore a validated archive to a new disk child, point the service at that directory and verify before resuming use.

The optional Intel macOS TensorFlow reproduction profile has a documented old protobuf advisory and is not the hosting profile. Use the audited default NumPy web runtime for this deployment. Retraining requires a separately provisioned compatible training environment; it does not replace production artifacts automatically.
