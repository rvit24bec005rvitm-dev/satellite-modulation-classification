# Phase 2 — verified implementation and readiness follow-up

This continuation works on `feat/phase2-readiness-followup`, preserving the
existing Phase 2 implementation and its local follow-up fixes. Fresh baseline
tests passed **195 Python / 40 frontend / 11 real client contracts**. The existing
compiled application also passed **28 browser scenarios** before the new UI
integration was validated. The original scientific report/checklist, history and
working tree were inspected before substantial changes.

## Current status

| Capability | Status | Current evidence / limit |
|---|---|---|
| Production CNN, canonical preprocessing and original data | EXISTING AND VERIFIED | Six original artifacts remain byte-identical to `ed8a78a`; all four experimental checkpoints match `414ddd7`. Hashes: `integration/phase2/followup/preserved-artifacts.json`. |
| External I/Q / explicit binary / SigMF subset | EXISTING AND VERIFIED | Existing encoding, bounds, clock, scaling and metadata tests preserved. No raw format guessing. Full SigMF conformance is not claimed. |
| Long-capture browser and genuine window predictions | EXISTING AND VERIFIED | Mapped windows, bounded overview, real inference, selected-window inspection and batch exports. |
| Background capture import | COMPLETE AND VERIFIED | New HTTP 202 job workflow, staged bytes, bounded decode blocks, persistent progress, shared capacity, cancellation and cleanup tests. Existing synchronous import remains compatible. |
| Background labeled-dataset registration | COMPLETE AND VERIFIED | Separate persistent import jobs; explicit labels, provenance/split validation, progress and cancellation. Cancellation leaves no partial registration. |
| Leakage checks | COMPLETE AND VERIFIED | Indexed source intervals preserve exact-duplicate, session and overlapping-window protections, including adjacent-window boundary tests and existing cross-format tests. |
| Restart recovery | COMPLETE AND VERIFIED | Interrupted staged imports fail clearly and clean partial files. If a capture committed before interruption, recovery preserves it and completes the job. Managed orphan UUID uploads are cleaned before new jobs are admitted. |
| SQLite history and managed storage | EXISTING AND VERIFIED | Schema remains v1. New job kinds use existing JSON payloads; no destructive migration is required. One process / shared workspace only. |
| Window quality, Welch, activity and domain indicators | EXISTING AND VERIFIED | Existing DSP preserved. Unknown sample clocks no longer produce a CFO ambiguity interval labeled in Hz. Domain indicators are not validated OOD decisions. |
| Dataset / job pagination | COMPLETE AND VERIFIED | Search/pagination for datasets; all job pages accessible; selected jobs polled directly even when outside the current list page. |
| Reproducible JSON / CSV / individual PDFs | COMPLETE AND VERIFIED | Individual PDF now includes input hash, capture position, clock, retention and quantity types. Batch CSV includes exact vectors, preprocessing, source positions, model/capture/dataset hashes. |
| Batch / evaluation PDF summaries | COMPLETE AND VERIFIED | Reports render stored results without inference. Vector confidence plot, provenance, labels/metrics when supplied and explicit limitations. At most 512 plotted windows, 200 table rows and 20 capture metadata previews; complete data remains in JSON/CSV. |
| Missing-model errors and upload deadlines | COMPLETE AND VERIFIED | Prediction APIs return safe 503 responses without private artifact paths; slow body reads return 408 and release import slots. |
| Premium UI, routes and Earth/galaxy | EXISTING AND VERIFIED | Visual design preserved. Globe change adds camera-azimuth diagnostic telemetry only. Browser drag check verifies an actual camera turn. New screenshots no longer overwrite original scientific audit images. |
| Local Python / frontend / contracts | COMPLETE AND VERIFIED | 209 Python, 43 frontend, 11 real contracts; details below. |
| Final production browser run | COMPLETE AND VERIFIED | 29 scenarios passed in 2.5 minutes against the final compiled app, including asynchronous imports, cancellation, 201-window evaluation export, batch PDF and globe interaction. |
| Container / Linux / Windows current-source CI | COMPLETE AND VERIFIED | All eight jobs passed on application source `b4f1c20`: Linux Python 3.10–3.12, Windows Python 3.12, Linux/Windows browsers, runtime security and container execution with protected named-volume recreation. |
| Windows and Linux production browsers | COMPLETE AND VERIFIED | Actual Windows Chromium and Linux Chrome runners each passed all 29 scenarios; macOS Chrome also passed 29 locally. |
| Training pipeline and model comparison | EXISTING AND VERIFIED | Existing grouped CNN/ResNet/CLDNN/RF infrastructure and artifacts retained; no training or model promotion performed. |
| Independent satellite-domain accuracy, production calibration / OOD / additional output classes | BLOCKED | No independent labeled acquisition corpus or justified production thresholds/checkpoint is available. |
| Permanent public hosting | BLOCKED | Hosting credentials/account connection and durable storage provisioning are required. No permanent deployment is claimed. |
| Full SigMF implementation, full-capture background DSP, persistent attribution PDFs | PARTIALLY IMPLEMENTED | Documented SigMF subset and bounded synchronous overview/DSP remain. Existing genuine IG/saliency and explanation JSON export are preserved. Attribution PDF persistence was not added. |
| Development-tool dependency remediation | PARTIALLY IMPLEMENTED | Previous runtime audits passed; 14 recorded development-tool findings remain. No application runtime dependency changed in this follow-up. |

## New architecture and API behavior

`captures.stage_upload` hashes and stages uploaded bytes once. Both synchronous
and background paths use the same decoder, with at most 65,536 samples per decode
block. The shared executor still has **two workers and four admitted jobs**, now
including capture import, dataset import, batch inference and external evaluation.
Two request-body upload slots and the configured storage/upload quotas remain in
force. Progress can be read and cancellation requested between decode/validation
blocks. A cancellation that arrives after a successful record commit can leave a
completed job; it does not roll back a valid recording.

The new request variants are:

- `POST /api/workspace/captures?background=true`: multipart recording/options,
  optional paired SigMF metadata; returns HTTP 202 and a `capture-import` job.
- `POST /api/workspace/datasets?background=true`: validated labeled manifest;
  returns HTTP 202 and a `dataset-import` job.
- Existing `GET /jobs/{id}` exposes phase/progress and the completed capture or
  dataset identifier. `GET /jobs/{id}/result` returns import provenance for these
  job kinds. Existing synchronous POST variants are unchanged.
- `GET /datasets?limit=50&offset=0&search=...` now supports pagination/search.
- `GET /jobs/{id}/export?format=pdf` returns a batch/evaluation technical summary;
  import jobs export JSON provenance instead.

All abbreviated endpoints above remain under `/api/workspace`. Authentication,
body limits and request IDs apply to both new variants. Model errors are logged
server-side, without exposing filesystem paths in prediction responses.

No SQL schema change was needed: existing `records` kinds remain capture,
analysis, dataset and job; the job payload adds `phase`, result identifiers and
staged-file descriptors. Capture and job IDs are distinct, and deleting a
terminal import job cannot delete its recording. Restart cleanup touches only
UUID-named managed temporary/canonical files without committed capture records
inside `SATMOD_DATA_DIR`. Do not share that directory between server processes.
Retention of committed captures/reports remains explicit, manual deletion.

## Verification and measured performance

Evidence is isolated under `integration/phase2/followup/`. Original audit logs,
figures, historical evaluations and model weights are preserved. Fresh final
local results: **209 Python / 43 frontend / 11 real contracts**; production build,
TypeScript and syntax/undefined-name lint passed. ESLint has zero errors and the
18 existing Fast Refresh warnings. The final production browser run passed all
29 scenarios in 2.5 minutes.

The [current application CI run](https://github.com/rvit24bec005rvitm-dev/satellite-modulation-classification/actions/runs/43072081077)
passed all eight jobs for source `b4f1c209e8118870d3030a8a47d3c9dfffeb5917`.
Linux Chrome and Windows Chromium each executed all 29 production browser
scenarios. The container built and verified protected asynchronous capture and
dataset import, real evaluation, both PDF export paths and retained state after
named-volume recreation. `github-ci-run.json`, `github-ci-jobs.json` and the
platform/container logs retain the actual results under the follow-up evidence
directory. Later documentation/evidence commits do not change that application
source. No permanent host was published.

The updated native preview at **http://127.0.0.1:8162** passed model health, all
eight SPA routes, referenced assets, real generated/uploaded inference, invalid
upload rejection and API error handling. A separate protected temporary native
workspace verified recording/job/dataset/evaluation/report persistence across a
complete server restart; see `protected-native-restart.json`. This is local
serving evidence, not a permanent hosted deployment.

The first follow-up browser execution passed 28 scenarios and failed a newly
added cancellation scenario because its exact label selector included option
text. The selector now uses the actual combobox accessible name; the failure log
is retained as `browser-first-followup.log`. No production assertion was removed.
The earlier remote Windows trace stalled in a 15-event synchronous pointer move;
the test now uses three pointer events and additionally asserts camera azimuth.
The earlier Linux job failed during package-mirror DNS resolution. Its existing
remote-branch fix uses the hosted image's installed Chrome; this was retained.

`scripts/profile_workspace.py` measured a one-million-sample synthetic NPY import
at **187.59 ms**, with **3,418,907 bytes** peak traced allocation, and the last
128-sample window at **20.79 ms**. This exercises the shared decoder, excluding job
queue overhead. Values are host-specific; tracing excludes mapped pages, OS
cache and pre-existing arrays. They are not RF accuracy or throughput guarantees.

## Files changed in this follow-up

- Backend: `backend/api/__init__.py`, `backend/api/workspace.py`, `backend/main.py`,
  `backend/services/captures.py`, `operations.py`, `technical_reports.py`,
  `advanced_dsp.py`.
- Frontend: `external-workspace.tsx`, `external-evaluation.tsx`,
  `lib/operations-api.ts`, and diagnostic telemetry in `mission/earth-globe.tsx`.
- Verification: new `tests/test_import_jobs.py`, extended operational unit and
  browser tests, and isolated evidence/screenshots.
- Operations/documentation: `.env.example`, CI browser selection,
  `PHASE2_IMPLEMENTATION.md`, this report, and the README follow-up runbook.

## Local and production commands

From the repository root, with dependencies installed:

```bash
.venv/bin/python scripts/dev.py
# The launcher prints the actual available UI/API ports.
.venv/bin/python -m pytest tests -q
npm --prefix frontend test
npm --prefix frontend run test:integration
npm --prefix frontend run typecheck
npm --prefix frontend run lint
npm --prefix frontend run build
```

For a same-origin native production server, configure a private persistent data
directory and set `SATMOD_ACCESS_TOKEN` through the deployment environment:

```bash
npm --prefix frontend run build
export SATMOD_SERVE_FRONTEND=1
export SATMOD_PUBLIC=1
export SATMOD_DATA_DIR=/absolute/path/to/private-persistent-data
.venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Enter the access token in Settings. Run one process/replica. With the same token
set in the shell, container deployment is `docker compose up --build -d`; the
named `/data` volume must persist across recreations. Hosting needs a Python or
container service, `PORT`, the access token and a persistent data mount. Provision
those resources before claiming durable public deployment.

Raw complex NPY, real `(N,2)` / `(2,N)` NPY and explicit supported binary formats
support long captures within the configurable 128 MiB default upload cap.
CSV/NPZ/MAT/WAV/JSON retain bounded existing parser limits; MAT v7.3, multichannel
or unsupported SigMF types are rejected. The SigMF subset follows the
[primary specification](https://sigmf.org/), preserves raw metadata and never
guesses unsigned-integer offsets, sample rates or scaling.

Independent acquisition data is still required before fitting production
calibration, validating OOD thresholds, expanding classifier outputs or promoting
an experimental model. Synthetic fixtures verify implementation behavior only.
The remaining dataset/hardware/account limitations are not implementation
successes. No real orbital tracking, communication packet decoding or new RF
observations are claimed.

## Prior implementation record

The following prior-source report and its CI evidence are retained. The current
follow-up status table above supersedes capabilities subsequently extended.


Branch: `feat/phase2-signal-operations`. Starting source: `70c411b`. The working tree was clean at the start. This phase preserves the original production CNN, experimental checkpoints, historical evidence, TanStack routes and Earth/galaxy scene. No model was promoted and no new training or real-world accuracy improvement is claimed.

## Status by requirement

| Capability | Status | Executed evidence / boundary |
|---|---|---|
| Existing CNN, canonical preprocessing, generation, explanation, eight pages | EXISTING AND VERIFIED | Baseline: 149 Python, 35 frontend, 11 client contracts; previous remote 25-browser run recorded in original audit. Current regression includes those cases. |
| Complex/real NPY long recordings | COMPLETE AND VERIFIED | Mapped NPY vectors, N×2 and 2×N arrays; 100,000-sample tests and one-million-sample profile. |
| Raw complex binary | COMPLETE AND VERIFIED | Nine explicit signed/float encodings; LE/BE, IQ/QI, planar/interleaved, scale, clock, offset and length. 18 encoding/layout cases. No format guessing. |
| SigMF ingestion | COMPLETE AND VERIFIED for documented subset | Paired metadata/data, sample clocks, frequency segments, UTC timestamps, SHA-512, description/license and raw metadata preservation; malformed/unsupported formats rejected. |
| Full SigMF standards/library coverage | PARTIALLY IMPLEMENTED | Bounded specification-based reader, not a full conformance library. Unsigned/real/multichannel datasets, embedded headers, trailing bytes, required extensions and nonzero global offsets are rejected. |
| Other existing formats | EXISTING AND VERIFIED | CSV/NPZ/MAT/WAV/JSON retain existing 4-MiB/65,536-sample limits. MAT v7.3 remains unsupported. |
| Long-capture navigation | COMPLETE AND VERIFIED | Disk-backed canonical IQ; ≤512 overview bins; explicit bounded windows; selected-segment NPY export; capture search/pagination. Import cap defaults to 128 MiB, not unlimited multi-GB uploads. |
| Windowed batch inference | COMPLETE AND VERIFIED | Real CNN results, raw confidence, entropy, timing, positions and valid UTC timestamps; bounded result pages and full JSON/CSV export. Up to 10,000 windows per job. |
| Activity and quality diagnostics | COMPLETE AND VERIFIED | Welch PSD, DC/IQ indicators, known-threshold clipping, power/envelope trajectory, conditional M-PSK CFO estimate, contiguous energy activity segments. Activity is not packet decoding. |
| Domain-shift report | COMPLETE AND VERIFIED | Amplitude/power quantiles, I/Q statistics, normalized spectral centroid, actual class/confidence distributions; 300-demo reference versus ≤32 sampled external windows. No OOD verdict. |
| Dataset registry / provenance | COMPLETE AND VERIFIED | Imported sources, sessions, licenses, source/canonical hashes, sample clocks, labels, splits and manifests; existing demo/synthetic references remain distinct. |
| Leakage protection | COMPLETE AND VERIFIED | Session contamination, duplicate frames, overlapping windows, cross-format canonical capture overlap, and forbidden demo fitting/evaluation tests. Cannot independently prove user-declared acquisition groups are truthful. |
| External labeled evaluation | COMPLETE AND VERIFIED | User-supplied test labels, real inference, confusion matrix, macro metrics, per-class metrics and diagnostic calibration; SNR bins only from supplied SNR values. Validation uses synthetic fixtures, not independent real satellite RF. |
| SQLite persistence | COMPLETE AND VERIFIED | Schema v1, WAL, parameterized queries, bounded metadata and file references; persistence checked from a separate Python interpreter. |
| History operations | COMPLETE AND VERIFIED | Search, class filter, pagination, JSON/CSV/PDF reports, delete and clear. Legacy `/api/history` remains compatible and now persists. |
| Background jobs | COMPLETE AND VERIFIED | Two workers / four slots, queued/running/completed/failed/cancelled, cooperative cancellation, truthful failures, interrupted-job recovery, progress polling and terminal-result deletion. |
| Background dataset import / full-capture DSP | PARTIALLY IMPLEMENTED | Batch classification and evaluation are jobs; bounded capture import, dataset registration, overview and per-window DSP remain synchronous with loading states. No web training handler. |
| Mission-control integration | COMPLETE AND VERIFIED | Actual managed-capture/job counters and persistent recent history; original globe source unchanged. |
| Reports | COMPLETE AND VERIFIED for individual analyses | Versioned JSON/CSV/PDF with identity/time, active model hash, preprocessing, input hash, metadata, uncalibrated scores, metrics, limitations and vector I/Q plot. Batch/evaluation JSON/CSV available. |
| Batch PDF / persistent attribution PDF | NOT IMPLEMENTED | Existing genuine IG/saliency and explanation JSON export preserved. No new attribution method or batch PDF renderer. |
| Training infrastructure / model comparison | EXISTING AND VERIFIED | Existing grouped CNN/ResNet/CLDNN/RF tools/checkpoints retained; no extra expensive fitting without appropriate independent data. |
| Registry-backed training adapter | NOT IMPLEMENTED | Existing offline NPZ/group-based training tools remain; managed capture manifests are not directly consumed by a new web or offline trainer. No training is launched in request handlers. |
| External-dataset multi-model evaluation | NOT IMPLEMENTED | New external evaluation runs the active production CNN. Existing experimental Model Arena comparisons remain on their documented synthetic/reference domains; no external RF leaderboard is claimed. |
| Persistent comparison PDF / metadata report | NOT IMPLEMENTED | Existing real two-signal comparison and JSON exports remain. No new persisted multi-capture comparison report renderer. |
| Expanded production classes / calibrated production / OOD | BLOCKED | Missing independent acquisition/training/validation/OOD corpus. Production remains three-class, uncalibrated and OOD-not-validated. |
| New low-SNR training experiments | NOT IMPLEMENTED | Existing synthetic experiments retained; no new data justification for changing production. |
| Public access controls | COMPLETE AND VERIFIED | Optional per-origin session token in Settings; constant-time server token check; public mode fails closed without configuration. Shared workspace key, not multi-user authorization. |
| Resource limits and observability | COMPLETE AND VERIFIED | Streamed disk spooling, two import slots, file/record/quota bounds, request IDs, structured logs, safe error responses, job state/counts and real inference summaries. |
| Production dependencies | COMPLETE AND VERIFIED | Production npm audit: zero findings after removing unused Zod and updating Seroval. Resolved Python web audit: zero findings over 30 dependencies. |
| Full development-tool remediation | PARTIALLY IMPLEMENTED | All-dependency npm audit still reports 14 development-tool findings (10 moderate, 2 high, 2 critical). Test/build toolchain upgrades were not forced into this feature phase; test servers remain local. Do not interpret runtime audit as a complete security guarantee. |
| Docker persistence/security | COMPLETE AND VERIFIED | Non-root image, `/data` volume, fail-closed public mode, Compose named volume, secret configuration, pinned dependencies and health check. Local image download exceeded the initial 180-second build budget. Remote Linux source `cd5eec7` built the current image and passed real inference/upload checks, protected capture import/batch/PDF checks, then removed/recreated the container and reloaded the same named-volume data. This is image execution evidence, separate from permanent hosting. |
| Permanent hosting | BLOCKED | No authenticated hosting connection. Render/Railway configuration is prepared. Free/ephemeral host filesystems do not guarantee persistence across redeployments; provision a persistent volume or keep operation local. |
| Linux / Windows browser execution | COMPLETE AND VERIFIED | Current source `cd5eec7`: 40 frontend tests, 11 contracts, type/lint/build and 28 production browser scenarios passed on each OS. Linux uses recorded runner Chrome; Windows uses Playwright Chromium. |

## Architecture and storage

New modules: `workspace_store.py` (SQLite/files), `captures.py` (ingestion/windowing), `operations.py` (jobs/evaluation/leakage), `advanced_dsp.py`, `domain_shift.py`, `technical_reports.py`, and `backend/api/workspace.py`. React adds external recording, dataset evaluation and history components to existing pages. `operations-api.ts` uses the existing centralized transport. No new router or external queue was introduced.

SQLite v1 uses `records(id PRIMARY KEY, kind, created_at, payload)` with `(kind,created_at)` index; kinds are capture, analysis, dataset and job. Schema creation is guarded by `PRAGMA user_version`; a newer version is refused. Serialized access and a small connection cache keep WAL open between window reads. One application process / one replica is supported. Two worker threads perform bounded jobs; SQLite stores job state, while results and IQ remain managed files. This is deliberately a lightweight standard-library SQLite design rather than an unnecessary ORM layer.

Data defaults to `.satmod/`, configurable with `SATMOD_DATA_DIR`. Only generated UUID filenames are used. Canonical sample files are little-endian float32 `(N,2)`; original datatype/scaling and hashes are retained. Temporary uploads are removed on normal success/failure. The default managed quota is 1 GiB with conservative metadata reserves and a 10,000-record cap. Retention is manual: delete captures, terminal job results or analyses to free space. SQLite may retain reusable pages; stop the app before database maintenance or filesystem backups. Back up the entire data directory, including WAL state, or use SQLite's backup API.

Legacy classification reports retain their input hash and bounded diagnostics, not full raw input. Keep the original file or use managed import for raw-data retention. Deleting a managed capture is explicit and irreversible; referenced datasets and active jobs prevent deletion. Historical reports retain provenance snapshots but cannot reconstruct deliberately deleted raw samples.

## Scientific boundaries

The production input remains I-first `(128,2)` with exactly one saved scalar training z-score. Explicit raw-file scaling is an importer-selected unit conversion, not automatic unit-power normalization. SigMF integer units are not guessed. A missing sample clock uses the existing normalized Fs=1 convention; physical timestamps are emitted only from valid metadata. No SNR is estimated from arbitrary mixtures. CFO requires an explicit BPSK/QPSK assumption and has modulo/bias limitations. IQ imbalance metrics are distribution indicators, not receiver calibration. No sample alignment, resampling, packet decoding, reliable OOD classification or real ephemeris tracking was added.

Ground truth comes from manifests, never model predictions. Evaluation exports include the frozen manifest/hash, explicit class order and ground truth for each prediction; the UI downloads the complete report rather than just its first prediction page. Test-only evaluation does not fit temperature or alter production. Session and hash checks cannot establish independence of undocumented or deliberately misidentified acquisitions. Domain diagnostics use a demo reference with unknown historical split membership and must not be interpreted as evidence of original training-domain identity.

## APIs

All additions use `/api/workspace`:

- `GET/POST /captures`, `GET/DELETE /captures/{id}`.
- `GET /captures/{id}/window`, `/diagnostics`, `/domain-shift`, `/activity`, `/segment`.
- `POST /captures/{id}/classify`.
- `POST /jobs/batch`; `GET /jobs`, `/jobs/{id}`, `/jobs/{id}/result`, `/jobs/{id}/export`; `POST /jobs/{id}/cancel`; `DELETE /jobs/{id}`.
- `GET/POST /datasets`, `DELETE /datasets/{id}`, `POST /datasets/{id}/evaluate`.
- `GET/DELETE /history`, `DELETE /history/{id}`, `GET /history/{id}/report?format=json|csv|pdf`.
- `GET /status`.

Existing endpoints remain. Protected operations require `X-Satmod-Token` when `SATMOD_ACCESS_TOKEN` is set. `SATMOD_PUBLIC=1` without a token returns 503 on protected APIs. Model health remains publicly readable but hides filesystem exception details. The shared key grants access to the whole workspace; use a private ingress or authenticated gateway for separate-user isolation. Tokens stay in browser session storage scoped to the configured API address, not URLs or committed preferences.

## Executed validation

- Baseline: 149 Python / 35 frontend / 11 real client contract tests executed before implementation; original CI/browser evidence checked.
- Phase 2 local Python: **196 passed**.
- Frontend: **40 passed**, including 5 new operational UI tests.
- Real TypeScript-to-Python contracts: **11 passed**.
- Current production browser suites: **28 passed on macOS Chrome**, **28 on Linux Chrome**, **28 on Windows Chromium**. These include the three new external-capture workflows and all original navigation/globe/scientific scenarios. macOS used an exclusive test server (`CI=1`) to prevent reuse of another editor’s server.
- TypeScript, production build and ESLint passed; 18 pre-existing Fast Refresh warnings remain.
- Original weights/statistics/demos/evaluation files and prior audit screenshots were preserved; current screenshots and logs are under `integration/phase2/`.
- Browser test authoring failures (missing jsdom environment and unawaited download path) were corrected; no production predictions or regression assertions were weakened.

All eight jobs in [remote run 43060024518](https://github.com/rvit24bec005rvitm-dev/satellite-modulation-classification/actions/runs/43060024518) passed on implementation source `cd5eec7`: Linux Python 3.10/3.11/3.12, Windows Python 3.12 (196 tests each), Linux/Windows frontend plus real contracts/production browser checks, runtime security audit, and the Docker image/volume-recreation smoke. Source and job metadata are stored in `integration/phase2/github-ci.json`. Later report/evidence commits do not alter application code.

A protected native server was also restarted and its capture, batch output and PDF reloaded using `scripts/check_workspace_deployment.py`; this is separate from container volume evidence. The first Linux browser setup failed on Ubuntu mirror DNS and was corrected by using the runner’s recorded installed Chrome. A local attempt required selecting installed Chrome; a subsequent reused-server attempt lost its shared server after 19 passing cases. The final exclusive-server macOS run passed all 28. These infrastructure attempts are not reported as application passes.


## Performance

`python scripts/profile_workspace.py` reproduces a seeded synthetic NPY profile. For one million samples (8,000,128 source bytes), the recorded import took 170.22 ms with 3,418,714 bytes of peak traced allocation; a selected final 128-sample window took 23.89 ms and its quality diagnostics 4.51 ms. For 100,000 samples, import took 52.43 ms with 2,423,305 traced bytes. These are host-specific observations, not throughput guarantees. Tracing excludes existing arrays, mapped pages and OS cache. The profile exposed repeated SQLite WAL setup; cached connections and range-only mapping avoid unnecessary repeated setup. Full samples remain on disk and plots are bounded.

## Exact execution

Local setup (from repository root):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
npm --prefix frontend ci
python scripts/dev.py
```

The launcher prints available URLs. Use the URLs printed by the launcher; it selects free ports when defaults are occupied. Restart it after backend edits. For explicit independent servers, set `SATMOD_API_PORT` / `SATMOD_FRONTEND_PORT` on the combined launcher. Persistent data defaults to `.satmod/`; do not commit it.

Production, same-origin native server:

```bash
npm --prefix frontend run build
export SATMOD_SERVE_FRONTEND=1
export SATMOD_DATA_DIR=/path/to/private-persistent-data
export SATMOD_PUBLIC=1
# Set SATMOD_ACCESS_TOKEN through your secret manager/environment; never commit it.
.venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Enter the same secret in Settings. For Docker, set `SATMOD_ACCESS_TOKEN` in the shell first, then:

```bash
docker compose up --build -d
python scripts/check_deployment.py http://127.0.0.1:8000 --wait-seconds 90
```

Both deployment verification scripts use only Python’s standard library. Compose binds localhost and uses a named volume owned by container UID 10001; host bind mounts must grant that UID write access. A hosting provider must set `PORT`, the access token and persistent `/data` mount. Keep one process/replica. Render's free plan configuration does not create a persistent disk. No permanent production URL was created in this phase.

## Additional reliability checks

Final storage review added regression coverage for dataset references beyond the first 200-record page, UUID case normalization, one report write per managed classification, and rejection of oversized legacy-format files before reading them into the legacy parser. Dataset/capture deletion and report/result reads are serialized with metadata changes. Evaluation exports retain supplied labels and manifest identity after registration deletion. The container smoke now explicitly removes and recreates the container against the same named volume.

New direct run/check commands:

```bash
# Windows environment activation instead of the POSIX source command:
.venv\Scripts\activate

python -m pytest tests/ -q
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run lint
npm --prefix frontend run test:integration
npm --prefix frontend run build
# Production browser tests use the SATMOD_TEST_PRODUCTION / SATMOD_TEST_BASE_URL
# and SATMOD_TEST_API_URL variables documented in frontend/playwright.config.ts.
python scripts/profile_workspace.py
```

## Source isolation

Another active editor changed the shared checkout to `feat/phase2-readiness-followup` during final testing. Final Phase 2 verification was isolated in a separate Git worktree on `feat/phase2-signal-operations`; the other checkout and its uncommitted work were preserved. The review branch is available in draft PR #1. Main was not changed by this phase.

## CI browser infrastructure correction

The first Linux browser job failed before browser execution: optional font packages could not be fetched because `azure.archive.ubuntu.com` DNS resolution failed. Application types/tests/contracts/build had passed. Linux CI now records and uses the runner’s preinstalled Google Chrome; Windows installs Playwright Chromium without a Linux package-manager step. The full browser assertions are retained. This is an infrastructure correction, not a suppressed application test. The corrected final Linux run passed all 28 cases. Linux Chrome 156.0.8078.79 was recorded by the runner.

## Upload deadline

`SATMOD_BODY_TIMEOUT_SECONDS` defaults to 120 seconds for complete request-body reads. Slow/chunked requests return 408 and release import slots and spool files. This limits upload reading, not background job execution. A regression test covers managed and ordinary API bodies, request IDs, and a successful subsequent import.

## File inventory and preserved artifacts

[`integration/phase2/files.json`](integration/phase2/files.json) lists created and modified files. [`integration/phase2/preserved-artifacts.json`](integration/phase2/preserved-artifacts.json) records six original artifacts and all four existing experimental checkpoint hashes, verified unchanged. Historical audit reports and screenshot files were preserved; new screenshots/logs have their own Phase 2 directory.

The current local preview is `http://localhost:8162`, serving the compiled React build and real API from one local process. Its temporary preview data is separate from the normal `.satmod/` workspace. This is a local verification service, not durable public hosting. Start the normal launcher for your persistent local workspace.

For an exclusive same-origin production browser run on macOS (after installation/build):

```bash
CI=1 PLAYWRIGHT_CHANNEL=chrome SATMOD_PYTHON="$PWD/.venv/bin/python" SATMOD_TEST_PRODUCTION=1 SATMOD_TEST_BASE_URL=http://127.0.0.1:18374 SATMOD_TEST_API_URL=http://127.0.0.1:18374 npm --prefix frontend run test:e2e
```

Chrome must be installed for that command. Alternatively run `npx playwright install chromium` in `frontend/` and omit `PLAYWRIGHT_CHANNEL`. Select a free port; `CI=1` refuses to reuse an unrelated server. Windows CI executes its own installed Chromium browser rather than inferring verification from Linux.

## Additional export and responsive validation

A separate browser inspection downloaded the full evaluation report for 201 known synthetic BPSK windows through the UI, confirming all 201 predictions, manifest entries and ground-truth labels survived beyond the 200-row result page. The expanded external-capture workspace was inspected at 1440px and 390px; mobile document width and scroll width were both 390px. Corrected inspection-script selectors did not change application behavior. Screenshots and `export-layout.json` are retained with Phase 2 evidence. These fixtures do not establish real-RF accuracy.
