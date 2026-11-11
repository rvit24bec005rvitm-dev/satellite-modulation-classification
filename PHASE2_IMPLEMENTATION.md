# Phase 2 implementation tracker

## Readiness follow-up (2026-11-11)

This continuation starts from the existing Phase 2 implementation (`41ebc4f`) and
preserves the in-flight fixes subsequently committed as `7b3bc0a`. Work is on
`feat/phase2-readiness-followup`; no existing changes or artifacts were discarded.
Fresh baseline: **195 Python / 40 frontend / 11 real client contracts passed**.
The original scientific audit and its checklist were read before implementation.

| Remaining requirement | Audited state | Follow-up status |
|---|---|---|
| Background capture import and progress/cancellation | Missing; imports block their request until decoded | Implemented and tested — same bounded pool, staged bytes, cooperative cancellation |
| Restart cleanup of interrupted staged imports | Partially implemented for inference jobs only | Implemented and tested — partial imports cleaned, committed recordings preserved |
| Capture overview concurrent deletion / mapped-file cleanup | Partially implemented | Implemented and tested — bounded read blocks and explicit mapping closure |
| Report completeness and batch PDF | Individual PDF omits input identity/position; no batch PDF | Implemented and tested — reproducible CSV, PDF summaries and complete JSON |
| Dataset/job navigation beyond first page | Partially implemented | Implemented and tested — pagination and direct selected-job polling |
| Current remote CI / Windows browser / container | Earlier evidence exists; latest runs in progress | Pending actual results |
| Independent satellite validation / production calibration / OOD | No independent labeled corpus | Blocked by data |
| Permanent hosting | No authenticated persistent host | Blocked by account authorization / persistent storage |

The sections below preserve the prior implementation history. New validation
evidence is kept under `integration/phase2/followup/`.

Final follow-up local checks: **208 Python / 43 frontend / 11 real contracts /
29 production browser scenarios passed**. TypeScript, production build,
syntax/undefined-name checks and ESLint passed (18 existing warnings).
Background labeled-dataset registration, split interval indexing, safe missing
model responses and upload deadlines are also implemented and tested. The
latest container/cross-platform CI for this branch remains pending until its
actual result is recorded; prior-source successes are kept separate.

## Prior implementation tracker


Starting commit: `70c411b`; branch `feat/phase2-signal-operations`. Working tree was clean. Original weights, experiments, routes and globe must remain unchanged.

| Requirement | Initial state | Current evidence/status |
|---|---|---|
| Canonical CNN, DSP, explanation, eight workspaces | Existing | Existing and verified: baseline 149 Python / 35 frontend / 11 contracts; original artifacts preserved |
| External small NPY/NPZ/CSV/MAT/WAV/JSON | Existing | Existing and verified; original parsers/preprocessing preserved |
| Explicit binary and SigMF import | Missing | Implemented and tested for explicit binary and documented SigMF subset |
| Long recordings / window navigation | Missing | Implemented and tested: bounded upload, on-disk canonical samples, memory-mapped windows |
| Batch prediction timeline / aggregate votes | Missing | Implemented and tested; descriptive aggregation only |
| Capture/dataset provenance and leakage checks | Partially implemented offline | Implemented and tested: capture/dataset provenance, duplicate/session/overlap checks |
| External evaluation / domain shift | Missing online | Implemented and tested diagnostics/evaluation workflow; independent real labeled data remains blocked by data |
| Training architectures/checkpoints/group splits | Existing | Existing and verified; no retraining or model promotion |
| Welch / quality / burst diagnostics | Partially implemented | Implemented and tested Welch, quality, conditional CFO, power trajectory and energy activity |
| SQLite history, managed files, migrations | Missing | Implemented and tested with standard-library sqlite3, schema v1 and managed files |
| Persistent bounded jobs / cancellation | Missing | Implemented and tested: bounded workers, cancellation, recovery; imports remain bounded synchronous operations |
| JSON / CSV / PDF reports | JSON partially exists | Implemented and tested individual JSON/CSV/PDF plus batch/evaluation JSON/CSV; batch PDF not implemented |
| Mission-control integration | Existing UI | Implemented and tested capture/history/evaluation panels and actual dashboard counters; shell/globe preserved |
| Container / hosting | Existing Docker, remote image check passed | Implemented volume/public-access configuration; current source image and named-volume recreation passed on remote Linux; permanent host blocked |
| Request IDs / observability / public access controls | Partially implemented | Implemented and tested request IDs, structured logs, limits and shared workspace access key |
| Windows backend / Linux browser | Existing remote verification | Existing backend verification repeated successfully; current Windows Chromium and Linux Chrome suites passed 28 cases each; macOS Chrome passed 28 |
| Production calibration / validated OOD | Tools exist, no independent corpus | Blocked by data; keep honest unavailable states |

This file is updated with actual executed checks as implementation proceeds. `PHASE2_REPORT.md` records final scope and limitations.


## Executed implementation checkpoint

- Baseline 149 Python / 35 frontend / 11 real contracts passed.
- Implemented and tested: external decoding/SigMF subset, mapped windows, real batch timeline, persistent SQLite/report files, cancellation/recovery, dataset provenance/leakage, external evaluation, Welch/quality/activity/domain diagnostics, history JSON/CSV/PDF, protected public-mode APIs and request IDs.
- Local verification: 196 Python, 40 frontend, 11 contracts, 28 production browser cases; build/type/lint passed. Runtime npm and resolved Python audits report zero findings. See PHASE2_REPORT.md for exact boundaries.
- Existing and verified: model artifacts/checkpoints, preprocessing, eight routes, original DSP/IG, training tools and prior CI records.
- Partially implemented: imports/overview/registration remain bounded synchronous operations; restricted SigMF coverage; development-tool vulnerability remediation. Batch/attribution PDF exports are not implemented.
- Blocked by data: independent real-world accuracy, production calibration/OOD, scientifically justified class expansion or production model improvements.
- Blocked by authorization/storage provisioning: permanent public host with persistent storage. No live deployment claim.
- Not implemented: batch/persisted attribution/comparison PDFs, a registry-backed training adapter, external multi-model evaluation, new training experiments or production promotion.

The detailed final status table, architecture, API/storage contracts, run commands and known limits are in PHASE2_REPORT.md. This document is a tracker, not evidence of unexecuted remote jobs.
