# Phase 2 implementation tracker

Starting commit: `70c411b`; branch `feat/phase2-signal-operations`. Working tree was clean. Original weights, experiments, routes and globe must remain unchanged.

| Requirement | Initial state | Current evidence/status |
|---|---|---|
| Canonical CNN, DSP, explanation, eight workspaces | Existing | Existing and verified: baseline 149 Python tests; frontend/contracts recorded below after execution |
| External small NPY/NPZ/CSV/MAT/WAV/JSON | Existing | Existing; preserve parser/preprocessing |
| Explicit binary and SigMF import | Missing | In implementation |
| Long recordings / window navigation | Missing | In implementation: bounded upload, canonical on-disk samples, memory-mapped windows |
| Batch prediction timeline / aggregate votes | Missing | In implementation; descriptive aggregation only |
| Capture/dataset provenance and leakage checks | Partially implemented offline | Extend to imported captures and labeled manifests |
| External evaluation / domain shift | Missing online | Implement diagnostics; independent real labeled data remains blocked by data |
| Training architectures/checkpoints/group splits | Existing | Preserve; no expensive retraining or model promotion |
| Welch / quality / burst diagnostics | Partially implemented | Extend existing actual-sample analytics |
| SQLite history, managed files, migrations | Missing | Implement with standard-library sqlite3 (transactional lightweight storage, no new ORM dependency) |
| Persistent bounded jobs / cancellation | Missing | Implement single-process bounded worker pool and restart recovery |
| JSON / CSV / PDF reports | JSON partially exists | Add versioned reproducible reports |
| Mission-control integration | Existing UI | Add functional operations panels; preserve visual shell and globe |
| Container / hosting | Existing Docker, remote image check passed | Extend persistent volume and public access configuration; permanent account connection still blocked |
| Request IDs / observability / public access controls | Partially implemented | Extend without logging raw signals |
| Windows backend / Linux browser | Existing remote verification | Preserve and extend CI; Windows browser pending execution resources |
| Production calibration / validated OOD | Tools exist, no independent corpus | Blocked by data; keep honest unavailable states |

This file is updated with actual executed checks as implementation proceeds. `PHASE2_REPORT.md` records final scope and limitations.


## Executed implementation checkpoint

- Baseline 149 Python / 35 frontend / 11 real contracts passed.
- Implemented and tested: external decoding/SigMF subset, mapped windows, real batch timeline, persistent SQLite/report files, cancellation/recovery, dataset provenance/leakage, external evaluation, Welch/quality/activity/domain diagnostics, history JSON/CSV/PDF, protected public-mode APIs and request IDs.
- Local verification: 193 Python, 40 frontend, 11 contracts, 28 production browser cases; build/type/lint passed. Runtime npm and resolved Python audits report zero findings. See PHASE2_REPORT.md for exact boundaries.
- Existing and verified: model artifacts/checkpoints, preprocessing, eight routes, original DSP/IG, training tools and prior CI records.
- Partially implemented: bounded synchronous imports/overview/registration (not all job types); restricted SigMF; development-tool vulnerability remediation; cross-platform/current-image CI until actually observed.
- Blocked by data: independent real-world accuracy, production calibration/OOD, scientifically justified class expansion or production model improvements.
- Blocked by authorization/storage provisioning: permanent public host with persistent storage. No live deployment claim.
- Not implemented: batch PDFs, persisted attribution PDFs, new training experiments or production promotion.

The detailed final status table, architecture, API/storage contracts, run commands and known limits are in PHASE2_REPORT.md. This document is a tracker, not evidence of unexecuted remote jobs.
