# SatModClass Phase 3 implementation and validation

Phase 3 extends the existing React/FastAPI research workspace. The production
CNN, three-class mapping, saved scalar preprocessing, eight routes and rotating
Earth remain in place. No production model was fitted or promoted. All generated
test/profile captures are explicitly synthetic; their results establish software
correctness, not independent satellite RF accuracy.

## Starting state and preservation

The starting commit was `0bf7c29fbdb6610ee4fa13484767a53afb32ea1e` on `main`.
Before implementation, the baseline actually passed **209 Python tests, 43
frontend tests, 11 real TypeScript-to-Python contracts and 29 production-browser
scenarios**. TypeScript, lint and production build passed. The baseline full npm
audit reproduced **14 findings: 10 moderate, 2 high and 2 critical**.

Baseline logs, dependency investigation, screenshots and executed performance
measurements are under [integration/phase3/baseline](integration/phase3/baseline/).
The existing user edit to `frontend/src/styles/mission-control.css` was preserved
byte for byte against its starting diff and is included unchanged. It supplies
the current galaxy styling; it was not rewritten as part of this implementation.

[preserved-artifacts.json](integration/phase3/preserved-artifacts.json) verifies
all six original production/data/evaluation files and four experimental
checkpoints against both the Phase 2 hashes and starting Git blobs. Original
audit and Phase 2 evidence is unchanged. New profiling and serving-check defaults write to
`integration/profiles/`, rather than the original audit destination.

## Implemented workflows

### All eight pages and shared controls

The existing page architecture and mission-control styling were retained.
The functional review and regression coverage are mapped below. Data-dependent
controls are disabled or show an explicit empty/error state when their required
capture, dataset, model or service is absent.

| Page | Reviewed actions and resulting behavior | Executed evidence |
|---|---|---|
| Overview | Real shared CNN result; demonstrative satellites clearly identified; Earth rotation/pause, drag, zoom/reset, selection, geographical fallback, reduced motion and offline states | Existing mission-dashboard browser tests and four viewport screenshots |
| Live Classifier | Upload/demo/classify; sample-clock/window synchronization; managed import, inspect, activity, DSP, batch/cancellation/history/export; new SigMF options and persistent explanations | Existing workflows plus Phase 2, Phase 3 controls and research-report browser tests |
| Satellite Studio | Genuine seeded waveform generation, impairments, scale calibration, CNN and scientific inspection; no fabricated predictions | Existing scientific workflows and numerical generator tests |
| Model Evaluation | Historical artifacts remain identified; external dataset registration, actual model/version selection, confusion matrix, per-class/SNR/calibration diagnostics, provenance and offline export | Research model API/unit/browser tests |
| Model Arena | Existing synthetic experiments remain separate; new comparison only within an exact registered dataset/test-window protocol | Research model comparison API/browser tests |
| Signal Intelligence | Existing diagnostics/ephemeral comparison; new managed A/B windows, historical explanation differences and persistent JSON/PDF | Scientific workflows and research-report browser tests |
| Satellite Theory | Existing ideal alphabets, sample-SNR formula, orbit/Doppler request and TWTA transfer controls | Phase 3 controls browser tests and existing numerical orbit tests |
| Settings | Connection check/save validation, session token and persisted theme/chart/generation preferences | Phase 3 controls browser tests; existing frontend route/API tests |

The additional rendered-control inventory opens native detail sections on all
eight routes at 1440 and 390 px and records actual accessible trees, control names,
disabled states, alerts and overflow. All 16 final route/viewport snapshots had
zero horizontal overflow, unnamed controls or alerts. A narrow-screen overflow
in the history controls was fixed by allowing the Clear history row to wrap.
The inventory complements action tests; it is not a
claim to have tested every input permutation or every assistive technology.

Verified bugs fixed include stale imported sample clocks, delayed inspection or
classification responses overwriting newer selections, stale legacy comparison
results, Plotly purge/zoom resets during job polling, an asynchronous plot-update race
where an older completion purged newer selected data, and a saved-comparison
selector losing its ID before an asynchronous callback. Saved generation
preferences now initialize before Studio mounts, and audio export propagates
the session token with a visible loading state, timeout and service error.
Mobile navigation now contains keyboard focus, closes with Escape and restores
the opener. A skip link
and current-route semantics were added. Interactive plots expose their toolbar
to screen readers, retain keyboard focus and provide an accessible numeric data
preview. Local PNG/JSON/PDF exports remain; Plotly cloud sharing is removed.
The original strict Overview console assertion also caught a timed-out Google
Fonts request.
The same Inter, JetBrains Mono and Space Grotesk faces are now bundled locally
(16 WOFF2 files, 333,068 bytes), with original SIL licenses, source URLs and
SHA-256 manifest. No typography redesign or external font request is required.
The serving check verifies every font and rejects missing font assets with HTTP 404.

### SigMF compatibility and export

The implementation was checked against the official
[SigMF 1.2.6 specification](https://sigmf.org/).
It supports all 28 core real/complex signed, unsigned and floating sample
datatype/endian combinations, explicit interleaved channel selection, multiple
capture segments, annotations, capture/global sample origins, hardware counters,
UTC timing and non-conforming-dataset capture headers/trailers.

Metadata is bounded at 64 KiB and depth 64, with up to 1,024 capture segments
and 4,096 annotations. Derived UTC values have Python datetime microsecond
precision; original timestamp strings remain in source provenance. Managed
imports need at least 128 samples; exported windows span 16–65,536 samples.

Unsigned samples require an explicit ADC zero; real input requires explicit
`i-only` conversion; multichannel input requires a selected zero-based channel.
Scaling and these choices are recorded. Ambiguity is rejected. Decode blocks
are bounded, including large component widths and 64-channel source layouts.
`core:offset` is a sample origin, not a byte skip. File-relative samples,
SigMF-relative samples and hardware global indices remain distinct.

Window export creates a paired canonical `cf32_le` dataset and fresh consistent
derived metadata/SHA-512, clips annotations and capture ranges, and includes the
complete original `source.sigmf-meta`. Conversion provenance describes selected
channel, zero, scaling and real conversion. This preserves source metadata;
it does not claim byte-lossless recovery of the original multichannel recording.
Metadata nesting, timestamp and hardware-index overflow return descriptive
validation errors before unsafe persistence or response encoding. Unknown
required extensions are rejected; optional extension metadata is opaque and
retained. Archives/collections, arbitrary required extension semantics and
full external schema certification are not implemented. Missing version,
captures or annotations remain accepted for legacy compatibility; this does
not imply full schema conformance.

### Genuine external multi-model evaluation

The catalog exposes production CNN, experimental RF, ResNet and CLDNN with
artifact hash/version, class map, preprocessing, lifecycle and runtime status.
Actual inference is enabled only when its artifact and runtime load successfully.
This host executes production NumPy CNN and the supplied Random Forest. The
optional TensorFlow neural runtime is unavailable here; ResNet/CLDNN stay
disabled with the reason, rather than returning substitute output.

Each queued evaluation freezes model identity, exact dataset/hash, ordered
held-out windows, preprocessing and class order. Persistent reports contain
actual predictions, accuracy, per-class metrics/confusion matrix, inference
timings and raw-score calibration diagnostics. Accuracy-by-SNR includes only
known supplied SNR. Forest vote proportions are labeled raw model scores;
legacy `rawSoftmax` CSV values remain specific to the production CNN.

Arena groups results by exact dataset/hash/test-window/class protocol and
rejects incompatible comparisons. Synthetic historical benchmarks remain
separate from registered acquisition-domain evaluations. Generated fixture
labels are explicitly synthetic and do not establish independent RF quality.

### Dataset registry to offline fitting

Validated exports produce a reproducible training manifest, deterministic NPZ,
artifact hash and compatible experiment configuration for the existing offline
scripts. Class compatibility, duplicate data/source overlap, source/session
independence and train/validation/test membership are checked. Test-only data
cannot be repartitioned into fitting data. Retained evaluation provenance also
protects known held-out windows after a registry entry is deleted.

The normal minimum is 1,000 frames with at least nine independent sessions and
every class represented in every split. An explicit small-fixture override
relaxes size only, preserving independence and split checks. Missing SNR is
`NaN` plus a required `snrKnown` mask; it never becomes invented 0 dB evidence.
The loader, training runner and replay metrics preserve that distinction.
Tests actually fit and replay a bounded four-tree offline fixture. There is no
public training endpoint and no automatic production promotion.

### Persistent explanations and managed comparison

Saved analysis explanations compute genuine saliency and Integrated Gradients
for a selected class logit. Reports preserve full 128-by-2 signed attribution,
baseline/steps, IG completeness residual (null for saliency), positive/negative
channel summaries, important regions and descriptive per-target method rank correlation derived
from stored maps without new gradients. A new computation replaces the same
method/target pair; different targets remain, without claiming a revision archive.
Model hash, preprocessing and full input identity are checked before recomputation. The
explanation wrapper preserves every signed/raw row and target summaries; full
analysis-history JSON additionally retains preprocessing and capture metadata.
`sourceHash` identifies uploaded bytes, capture hash identifies canonical managed
samples, `inputHash` identifies the exact selected float32 input, and
`modelWindowHash` identifies the first 128 comparison samples. These verify
software identity, not the truth of operator-supplied acquisition metadata.
Historical records without reconstructable original input fail explicitly.

Analysis-history JSON retains persisted explanations, prediction, metadata,
model identity, class scores, full attribution arrays and scientific limitations;
PDF adds vector I/Q attribution plots. Re-export of an unchanged saved report after reload is byte-identical
in the tested renderer/runtime, without new inference. Recomputing a report
intentionally changes timestamps/timings. A new comparison has a new ID;
recomputed explanations retain their analysis ID. Example PDFs/JSON in
[integration/phase3/examples](integration/phase3/examples/) use labeled generated
fixtures, not independent RF evidence.

Managed comparison accepts two stored recordings/windows, computes actual
prediction/power/phase/amplitude/constellation/PSD data and persists JSON/PDF.
DSP uses the full 16–65,536-sample selection; the CNN uses its first complete
128-sample window when available. Saved predictions require exact model,
preprocessing and 128-sample identity. Identical windows avoid duplicate inference.
Physical axes are shared only where valid; unknown or unequal sample rates use explicit normalized frequency
and density units. Equal sample rates do not establish clock synchronization.
No resampling or physical alignment is silently performed. Attribution
differences require compatible saved methods, targets, model/preprocessing and
supplied rates (or both unknown rates for relative-index subtraction only),
and use an explicit relative-index interpretation. Neither establishes
synchronization. Comparison JSON contains complete bounded diagnostics and
hashes; independent replay still requires separately retained raw recordings.

### Scientific visualization

New STFT controls select FFT size, overlap and window. The heatmap/waterfall,
nearest-bin frequency marker and computed spectral peaks use actual complex
signal density. No detrending, padding or resampling is hidden. FFT size is
bounded at 1,024, displayed frequencies at 256 bins and selected complete time
windows at 128; the response records reduction and exact selected positions.
The -300 dB display floor is identified as a display convention, not a measured
noise floor. Selected windows and computed activity ranges are highlighted on
capture plots. Large-capture and neural input limits remain explicit.

## API and source inventory

All new routes use the existing API access policy/token middleware. They do not
accept model paths, executable configuration or unrestricted fitting commands.

| API | Purpose |
|---|---|
| `GET /api/workspace/captures/{id}/sigmf?start=0&length=128` | Metadata-preserving derived SigMF ZIP |
| `GET /api/research/models` | Validated model/runtime catalog |
| `POST /api/research/evaluations` | Persistent actual model/dataset job |
| `GET /api/research/evaluations` | Completed reports and compatible groups |
| `POST /api/research/evaluations/compare` | Exact-protocol comparison |
| `GET /api/research/datasets/{id}/training-manifest` | Validated provenance/split/export manifest |
| `GET /api/research/datasets/{id}/training.npz` | Compatible deterministic NPZ |
| `GET /api/research/datasets/{id}/experiment-config` | Offline trainer configuration |
| `POST/GET /api/research/analyses/{id}/explanations` | Compute/save or inspect full explanation wrapper |
| `GET /api/workspace/history/{id}/report?format=json\|csv\|pdf` | Full saved analysis, metadata/preprocessing and persisted explanation export |
| `POST/GET /api/research/comparisons` | Create/list managed comparisons |
| `GET/DELETE /api/research/comparisons/{id}` | Inspect/delete persisted comparison |
| `GET /api/research/comparisons/{id}/export?format=json\|pdf` | Reproducible report |
| `POST /api/research/time-frequency` | Bounded configurable complex PSD/STFT |

New backend modules are `sigmf_io`, `capture_windows`, `research_evaluation`,
`training_exports`, `saved_explanations`, `managed_comparisons`, `research_pdf`
and `time_frequency`, with three research routers. Existing capture, job,
registry, store, PDF and training modules were extended. New UI components are
`saved-explanations`, `managed-comparison`, `matched-evaluations`,
`training-dataset-export` and `time-frequency-inspector`, with a typed
`research-api` client. The shared shell/charts and existing workflow components
carry focused fixes. New Python, component and browser tests live alongside
the originals. `integration/phase3/changed-files.txt` records the exact inventory.

## Dependency and security review

Vite moved from 7.1.12 to patched 7.3.8; Vitest from 2.1.9 to 4.1.12;
`typescript-eslint` to 8.72.2. The vulnerable transitive matcher is scoped to
`@typescript-eslint/typescript-estree` with compatible `minimatch` 11.0.1,
and `source-map-js` was refreshed. No forced audit repair was used. The initial
attempt to install a newer matcher failed because its new transitive package
was unavailable; the failure and successful compatible selection are retained.

Version/engine metadata, the baseline audit and final full audit are committed.
The final audit reports **zero findings across production and development**.
CI now audits both. Official [Vitest migration](https://vitest.dev/guide/migration/)
and [Vite migration](https://vite.dev/guide/migration) guidance informed the
selection; this is a tested patch/minimal-compatible upgrade, not a Vite 8
build-system migration.

Upload limits, safe UUID-managed paths, no-pickle external NPZ loading,
allowlisted artifact loaders, SQLite transactions and atomic report-file replacement
with rollback on observed metadata-write failure, bounded jobs,
explanation slots, two concurrent training-export builds and file-handle cleanup remain enforced.
Historical evaluation provenance that is missing, unusable or over the 16 MiB
review bound blocks training export for offline review. Deleting both registry
and historical reports removes that recoverable evidence; it is not an immutable
external data ledger. Trusted operators
can supply labels and metadata; software cannot establish that their claims
are true or detect undeclared real-world session relationships. Keep protected
deployments configured with the existing access token. Tokens are session-only
in the UI and are not included in report exports.

## Executed validation and measurements

Final validation results and exact evidence paths are recorded here after the
last regression run. Failed integration attempts are retained separately so
they cannot be mistaken for successful final runs.

Implementation source: [`2f63d7f`](https://github.com/rvit24bec005rvitm-dev/satellite-modulation-classification/commit/2f63d7f48d78386776fe514a2f7a5e20483701c8).
The exact application/static/test/configuration and reproduction-helper hashes
are in `source-manifest.json` (261 files). Machine-readable results are in
`validation-summary.json`.

| Check | Executed final result | Evidence under `integration/phase3/` |
|---|---|---|
| Python regression | 340 passed | `python-release.log` |
| Frontend component/routes | 60 passed | `frontend-release.log` |
| Real API-client contracts | 11 passed | `contracts-complete.log` |
| Production Chrome E2E | 44 passed, no retries, 4.1 min | `browser-final-release.log` |
| TypeScript | Passed | `typecheck-release.log` |
| ESLint | 0 errors; 18 pre-existing fast-refresh warnings | `eslint-release.log` |
| Python severity lint | 0 findings in checkout Python source | `repository-source-release-lint.log` |
| Final production build | Passed | `build-final-release.log` |
| SPA/API/static assets | 8 routes, 16 local fonts, correct missing font/API 404 | `production-serving.json` |
| Full npm audit | 0 vulnerabilities | `npm-audit-complete.json` |
| Expanded desktop/mobile inventory | 16 snapshots, 0 overflow/unnamed controls/alerts | `control-inventory.json` |
| Protected native restart | 36 HTTP requests; all six JSON/PDF exports byte-identical; only GET verification after restart; unauthorized requests 401 | `protected-restart-smoke.json` |

Every original test remains: additions total 131 Python, 17 frontend and 15
browser cases. The final browser run retains the original strict console and
scientific assertions. New checks synchronize actual service completion;
no original assertion was weakened. The deferred-Plotly regression reproduces
the ownership race before the fix. Earlier failed runs are retained as diagnostic
history, including the external-font timeout and pre-fix mobile measurements.
A broad local lint attempt included installed `.venv` dependencies; the final
clean-checkout source lint excludes those untracked packages and passes.

The implementation and all evidence are published directly to `main` without
another Phase3 branch, force push or history rewrite. [Repository CI run 43253979275](https://github.com/rvit24bec005rvitm-dev/satellite-modulation-classification/actions/runs/43253979275)
passed all eight jobs on pushed `main` commit `76aef24`: Python on Ubuntu
3.10/3.11/3.12 and Windows 3.12, compiled frontend/contracts/browsers on Linux
and Windows, the runtime security audit, and the protected Docker upload,
inference and persistent-volume recreation checks. Exact job/step conclusions
are retained in `integration/phase3/ci-validation.json`. Python 3.11/3.12 on
Linux and Python 3.12 on Windows each passed 340 tests. Python 3.10 passed 338
with two honest RF runtime skips: it installs scikit-learn 1.7.2, while the frozen
RF artifact requires its recorded 1.9.1 version. Both frontend platforms passed
60 component tests, 11 contracts and 44 compiled browser scenarios (Linux 4.0 min,
Windows 4.1 min). No unavailable model was substituted. Subsequent documentation
and evidence publication leaves all 261 source/static/test/configuration/helper
hashes unchanged.

The controlled reader optimization alternated the old per-window mapping path
and the bounded eight-map job LRU three times with identical cached metadata,
model, 65,536-sample seed and 512 windows. Median throughput was **173.05 →
463.88 windows/s (2.68×)**, with **maximum probability difference 0** across
all 512 full vectors. The portable reproduction is
`scripts/profile_research_windows.py`; evidence and precise scope are in
[window-reader-reproducible.json](integration/phase3/window-reader-reproducible.json).
This measures one worker including serialization, excluding upload/HTTP, and
is not a capacity guarantee.

The identical actual-browser probe reproduced a polling bug before the fix:
45 chart SVG removals and lost zoom. After the fix: **zero removals and exact
zoom retained**, with 13 real poll responses over 6.5 s. Heartbeat p95 was
17.7 → 17.1 ms; maximum was 41.8 → 30.7 ms. The separate network/batch figures
are host-load-sensitive observations, not a controlled backend speedup claim.
See baseline/final `browser-performance.json` and the committed probe.

Same host/runtime (macOS x86_64, Python 3.14.7, NumPy 2.5.3, CPU, batch 1),
separate executed baseline/final processes. Timings below are observations;
the unchanged inference can also become slower with host scheduling/load.

| Measurement | Baseline | Final | Scope |
|---|---:|---:|---|
| CNN inference median / p95 | 1.13 / 1.48 ms | 1.59 / 1.89 ms | 20 warm calls, unchanged weights |
| DSP 128 median | 7.55 ms | 6.68 ms | 20 calls |
| DSP 65,536 | 84.77 ms | 78.93 ms | Untraced elapsed; traced memory measured separately |
| ASGI classify median / p95 | 20.12 / 25.84 ms | 22.39 / 37.29 ms | 10 real TestClient calls, no network RTT |
| Actual network classify median | 33.97 ms | 27.32 ms | 10 HTTP requests through browser context |
| Actual HTTP batch throughput | 58.83 windows/s | 411.26 windows/s | 512 windows, includes polling; host-load-sensitive |
| 1M-sample NPY import | 173.43 ms | 138.18 ms | Tracemalloc enabled, generated 8 MB source |
| 1M-sample import traced peak | 3.261 MiB | 3.260 MiB | Excludes source/mapped pages/OS cache |
| 65,536 DSP traced peak | 10.837 MiB | 10.837 MiB | Separate traced call |
| Process peak RSS | 126.98 MiB | 125.95 MiB | Python/model/DSP/TestClient process |

Evidence: baseline/final `performance-services.json`, `performance-import.json`
and `browser-performance.json`. Only the paired reader ablation above isolates
that optimization; these process/network figures do not establish a general
speedup across machines or workloads.

Store-cache microbenchmarks are separately scoped in `performance-store.json`.
Traced allocations exclude pre-existing arrays, mapped pages and OS cache;
process peak RSS includes imported Python/ML modules. The large Plotly bundle
remains lazy loaded; it is not presented as a small initial asset.

## Local execution and reproduction

From the repository root, use Python 3.10+ (optional TensorFlow training/runtime
requires a compatible separate Python installation; the supplied RF requires
its recorded scikit-learn 1.9.1 on Python 3.11+) and Node >=22.12:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
npm --prefix frontend ci
npm --prefix frontend run build
SATMOD_DATA_DIR="$PWD/.satmod" .venv/bin/python scripts/serve.py
```

Open `http://localhost:8000`. The refreshed existing local preview is also
available at `http://localhost:8162`; its matching Phase3 backend retains the
same `.satmod` workspace and all 294 existing history records. Counts and
health are verified in `integration/phase3/local-preview.json`. Set `SATMOD_ACCESS_TOKEN` for a protected instance;
enter the same token in Settings. Windows uses `.venv\Scripts\python.exe` and
the equivalent environment-variable syntax. Separate development servers use
`.venv/bin/python scripts/dev.py`.

```sh
.venv/bin/python -m pytest tests -q
npm --prefix frontend run typecheck
npm --prefix frontend run lint
npm --prefix frontend test
npm --prefix frontend run test:integration
npm --prefix frontend run build
npm --prefix frontend audit
```

Production browser tests (from `frontend/`, with installed Chrome; otherwise
install Playwright Chromium and omit `PLAYWRIGHT_CHANNEL`):

```sh
CI=1 SATMOD_DATA_DIR=/tmp/satmod-phase3-e2e \
SATMOD_TEST_PRODUCTION=1 SATMOD_TEST_API_URL=http://127.0.0.1:18384 \
SATMOD_TEST_BASE_URL=http://127.0.0.1:18384 PLAYWRIGHT_CHANNEL=chrome \
node node_modules/@playwright/test/cli.js test --reporter=list
```

Run profiles separately from CPU-intensive suites for comparable observations:

```sh
.venv/bin/python scripts/profile_services.py --output /tmp/services.json
.venv/bin/python scripts/profile_workspace.py > /tmp/import.json
.venv/bin/python scripts/profile_research_windows.py --output /tmp/reader.json
node integration/phase3/browser-profile.mjs http://127.0.0.1:8000 /tmp/browser.json
node integration/phase3/control-inventory.mjs http://127.0.0.1:8000 /tmp/controls.json
```

The browser/profile commands create labeled synthetic managed test records;
use a disposable `SATMOD_DATA_DIR`. The offline training UI exports the NPZ,
config and manifest; fit/replay with `scripts/train.py` and `scripts/evaluate.py`
using a new output directory. No existing checkpoint directory should be reused.

## Remaining boundaries

Independent acquisition-domain RF quality, receiver calibration, real SNR truth,
fitted probability calibration/OOD rejection and production model promotion
remain blocked by suitable independent datasets and controlled validation.
Optional ResNet/CLDNN runtime execution remains blocked on this host. Model
version selection enumerates currently registered artifact versions; it does
not invent a version archive. Unknown required SigMF extensions, collection
archives and universal external-schema certification are not implemented.
Bounded previews and PDF plots complement full machine-readable data; they do
not claim full-resolution multi-million-sample browser rendering. Accessibility
and responsive regression checks are executed, without claiming a comprehensive
screen-reader or WCAG certification. Permanent hosting was not newly provisioned.

Requirement classifications are recorded in [PHASE3_CHECKLIST.md](PHASE3_CHECKLIST.md).
