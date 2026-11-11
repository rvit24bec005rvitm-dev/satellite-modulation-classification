# Phase 2 — external signal operations and deployment readiness

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
| Expanded production classes / calibrated production / OOD | BLOCKED | Missing independent acquisition/training/validation/OOD corpus. Production remains three-class, uncalibrated and OOD-not-validated. |
| New low-SNR training experiments | NOT IMPLEMENTED | Existing synthetic experiments retained; no new data justification for changing production. |
| Public access controls | COMPLETE AND VERIFIED | Optional per-origin session token in Settings; constant-time server token check; public mode fails closed without configuration. Shared workspace key, not multi-user authorization. |
| Resource limits and observability | COMPLETE AND VERIFIED | Streamed disk spooling, two import slots, file/record/quota bounds, request IDs, structured logs, safe error responses, job state/counts and real inference summaries. |
| Production dependencies | COMPLETE AND VERIFIED | Production npm audit: zero findings after removing unused Zod and updating Seroval. Resolved Python web audit: zero findings over 30 dependencies. |
| Full development-tool remediation | PARTIALLY IMPLEMENTED | All-dependency npm audit still reports 14 development-tool findings (10 moderate, 2 high, 2 critical). Test/build toolchain upgrades were not forced into this feature phase; test servers remain local. Do not interpret runtime audit as a complete security guarantee. |
| Docker persistence/security | COMPLETE AND VERIFIED | Non-root image, `/data` volume, fail-closed public mode, Compose named volume, secret configuration, pinned dependencies and health check. Local image download exceeded the initial 180-second build budget. Remote Linux source `d48a17b` built the current image and passed real inference/upload checks, protected capture import/batch/PDF checks, then removed/recreated the container and reloaded the same named-volume data. This is image execution evidence, separate from permanent hosting. |
| Permanent hosting | BLOCKED | No authenticated hosting connection. Render/Railway configuration is prepared. Free/ephemeral host filesystems do not guarantee persistence across redeployments; provision a persistent volume or keep operation local. |
| Windows browser | COMPLETE AND VERIFIED for initial Phase 2 source | Windows source `41ebc4f`: 40 frontend, 11 contracts, build and 28 production Chromium tests passed. Final source `d48a17b` is being rechecked separately. This is Windows Chromium execution, not a macOS/Linux browser substitute. |

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
- Phase 2 local Python: **195 passed**.
- Frontend: **40 passed**, including 5 new operational UI tests.
- Real TypeScript-to-Python contracts: **11 passed**.
- Production Playwright: **28 passed**, including three new external-capture workflows; final small pagination/deletion changes are additionally covered by type/unit/API checks and awaiting current remote regression.
- TypeScript, production build and ESLint passed; 18 pre-existing Fast Refresh warnings remain.
- Original weights/statistics/demos/evaluation files and prior audit screenshots were preserved; current screenshots and logs are under `integration/phase2/`.
- Browser test authoring failures (missing jsdom environment and unawaited download path) were corrected; no production predictions or regression assertions were weakened.

First Phase 2 remote checks on `41ebc4f` passed Python on Linux 3.10/3.11/3.12 and Windows 3.12, the image smoke, resolved Python security audit, and 28 actual Windows Chromium scenarios. Follow-up `d48a17b` passed all four Python matrix jobs, the security audit and the strengthened named-volume recreation/image check; final browser jobs remain pending until observed. A protected native server was also restarted and its capture, batch output and PDF successfully reloaded using `scripts/check_workspace_deployment.py`; this native check is separate from container volume evidence.

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

The first Linux browser job failed before browser execution: optional font packages could not be fetched because `azure.archive.ubuntu.com` DNS resolution failed. Application types/tests/contracts/build had passed. Linux CI now records and uses the runner’s preinstalled Google Chrome; Windows installs Playwright Chromium without a Linux package-manager step. The full browser assertions are retained. This is an infrastructure correction, not a suppressed application test. The final run must pass before claiming current Linux browser verification.

## Upload deadline

`SATMOD_BODY_TIMEOUT_SECONDS` defaults to 120 seconds for complete request-body reads. Slow/chunked requests return 408 and release import slots and spool files. This limits upload reading, not background job execution. A regression test covers managed and ordinary API bodies, request IDs, and a successful subsequent import.
