# SatModClass technical implementation audit

Work starts from `ed8a78a` on `feat/scientific-reliability-audit`. The premium React workspaces, navigation, routes and interactive Earth remain in place. The source changes implement scientific/backend features rather than replacing the interface. Concurrent local Earth/CSS visual edits, including the later starfield changes, were retained; the initial visual edits were not authored by this audit. The subsequent requested galaxy polish adds seeded galactic-plane stars, procedural dust texture, batched constellation links and pause-aware background drift, preserving that work.

## Verified defects and implemented changes

| Defect | Actual fix and principal files |
| --- | --- |
| Legacy upload preprocessing differed from saved samples | `utils/preprocessing.py` is the canonical validated I-first 128×2 float32 window/z-score contract. Legacy upload padding, arbitrary reshape and unit-power scaling were removed. API uploads, generation handoff, demo requests, benchmarks, comparison, explanation and Streamlit use this contract. |
| Loader could replace a requested model with sibling weights | `utils/model_backend.py` now loads exactly the requested artifact, validates its architecture/tensors and fails clearly instead of substituting another model. Optional frameworks load lazily. |
| Legacy Arena fabricated ResNet/CLDNN/RF predictions and fixed comparisons | Replaced `pages/4_🔬_Model_Arena.py` with the shared real registry and measured benchmark. Removed probability transforms, invented accuracies/latencies and random fallback captures. |
| Incomplete real model metadata/benchmarking | `backend/services/model_registry.py` records versions, class order, preprocessing, artifact identity/size, parameter counts, training/evaluation provenance and availability. Timings warm up, repeat, report median/p95, separate preprocessing and record hardware/batch size. |
| Fixed GMSK/BFSK parameters and duplicated constellation definitions | `utils/modulation.py` provides a shared natural-binary mapping and configurable APSK rings. BFSK has actual ±frequency states, Hz deviation and phase-continuity controls; GMSK has BT-controlled Gaussian shaping and h=0.5. Gaussian span adapts to low BT. Rectangular 32-QAM was added. |
| Fading enable flag and output normalization obscured physical behavior | `generate_iq_signal` respects disable independently of K, supports block Rician/Rayleigh fading and IID/Wiener phase errors, uses independent seeded streams, checks frequency bounds and reports applied shaping/channel power stages. Output either preserves gain or explicitly calibrates final RMS. |
| Entropy was used as unsupported reliability/OOD evidence in legacy pages | Removed arbitrary entropy threshold decisions and gauge risk bands. Classifier responses/UI separate numeric validity, uncalibrated softmax confidence, entropy, calibration status and OOD status. OOD decisions stay unavailable. |
| No genuine attribution feature | `backend/services/explain.py`, `/api/explain`, and `classification-insights.tsx` implement analytic class-logit saliency/Integrated Gradients, signed I/Q maps, baseline/target/step metadata, completeness residual, loading/cancellation and export. |
| Missing relative-power/variance/crest metrics and undefined zero-signal handling | `signals.py`/DSP return true power/RMS/variance/PAPR/crest, phase/frequency traces, real spectra/PSD, relative units and unavailable undefined values. Arbitrary-capture SNR remains unavailable. |
| Signal comparison mixed physical/normalized axes and omitted classification/timing | Comparison now reports explicit axis units with proper PSD density conversion, overlap/rate/length metadata, actual CNN results and times, analysis-only uploads and reproducible raw-capture export. It never guesses resampling/alignment. |
| Legacy eye/reference plots implied unknown receiver timing/amplitude | Unknown capture timing is explicitly unavailable; optional exploratory legacy traces require an assumed clock. CPM does not show a fake symbol eye or fixed constellation. Raw receiver clouds are not overlaid with uncalibrated unit-energy ideal symbols. |
| Orbit pass used inconsistent range/elevation/Doppler and could exceed 90° latitude | All quantities now derive from one spherical circular propagation, rotating Earth and reference station. Coordinates, peak elevation, range derivative and Doppler are tested. |
| Resource/error edge cases | Bounded body buffering and headers retain CORS; JSON I/Q is strict; duplicate/encrypted archives, reversed/empty CSV and invalid WAV clocks are rejected; exact class enums and explanation schemas are enforced. Production static misses return 404 and server ports are configurable. |
| Avoidable analysis/inference work | Convolution vectorization, one FFT, vectorized contiguous PSD reductions and bounded plot data retain numerical/scientific correctness. Reduced motion now also ends decorative infinite CSS animation. |

The earlier React preprocessing, SAME padding/ReLU-before-BN correction, constant-envelope CPFSK implementations and corrected 32-APSK point count were verified rather than falsely reported as new repairs. See `CHECKLIST.md` for issue-by-issue statuses.

## Model/data fidelity and controlled measurements

The six original model/data/evaluation artifacts are byte-identical to `ed8a78a`, with SHA-256 hashes in `preserved-artifacts.json`. Production still uses the original supplied CNN. No checkpoint was automatically promoted.

| Check | Actual measurement |
| --- | --- |
| Saved-stat transform on the supplied 300 demos | 71.00% descriptive accuracy |
| Legacy unit-power-before-z-score path on the same demos | 33.33% |
| No transform on the same demos | 33.33% |
| Loop versus vectorized NumPy, 300 frames | 100% class agreement; maximum probability difference 5.81×10⁻⁷ |
| Fresh Keras versus NumPy, 43 frames | 100% class agreement; maximum absolute probability error 4.17×10⁻⁷ |
| Analytic logit gradients versus TensorFlow, all three class targets | Maximum absolute error 2.39×10⁻⁷ |
| Matched 50-repeat inference median / p95 | Loop 4.76 / 9.03 ms; vectorized 1.26 / 4.11 ms |
| Descriptive production-demo calibration | ECE 0.0867; NLL 0.6000; Brier 0.3545 |

The demo subset's original split membership is unknown. Those observations neither reproduce nor replace the historical **72.44% / 900-observation** aggregate evaluation. Demo frames are prohibited for training/calibration, including renamed or repackaged copies.

The historical CSV verifies the reported low-SNR collapse: at −20 dB accuracy is 20%, with 16-QAM recall 0%; at −18 dB accuracy is 33.33%, again with 16-QAM recall 0%. This is class-dependent, not a uniform confidence reduction. On the separate descriptive demo subset, 43 of 100 true 16-QAM frames were predicted as QPSK and 30 of 100 BPSK frames as QPSK. The recorded calibration error confirms that raw softmax confidence should not be interpreted as calibrated correctness. The much lower accuracy on newly generated frames also exposes domain dependence; the available evidence does not isolate one causal channel mismatch.

The host profile in `performance.json` measures model initialization, preprocessing, inference, generation, full-resolution analysis, API request handling and peak memory. It explicitly distinguishes ASGI request timing from real network RTT and measures untraced analysis latency separately from allocation tracing. Plot payloads stay bounded at 1,024 waveform/constellation/spectral points and 64 STFT time bins, while the full capture is retained for calculations.

A fresh short headless Chrome probe of the final galaxy scene observed **55.27 requestAnimationFrame callbacks/s**, versus **59.50** on the sequential empty-page control on this Intel host. This is cadence evidence, not a GPU frame-time benchmark or a general 60 FPS guarantee. Hidden-tab rotation paused exactly and resumed without a jump; sidebar collapse, fullscreen and runtime-error checks passed. See `browser-performance.json`.

## Real experimental models and reproducibility

`experiments/data/controlled-synthetic-v1.npz` contains **1,800 new synthetic frames**, generated from **300 independent source sessions** over six SNR levels. Each source sequence remains grouped across SNRs. All four experiments share protocol `c7c74ee438f937fdcf4e409028c3abf4802d9f275c400d631c3c3ba1e594db44`, including the identical **360-frame test partition**. Normalization fits training only; augmentation affects training only; checkpoint/temperature selection uses validation only. Group IDs and exact duplicate frames cannot cross partitions.

| Candidate | Actual held-out synthetic accuracy | Macro F1 | CNN on the identical test |
| --- | ---: | ---: | ---: |
| RF control | 66.94% | 65.07% | 65.83% |
| RF + training AWGN | 66.67% | 64.81% | 65.83% |
| Residual CNN, eight bounded epochs | 49.44% | 42.38% | 65.83% |
| CLDNN, eight bounded epochs | 68.89% | 66.87% | 65.83% |

RF augmentation improved some low-SNR bins but reduced high-SNR/overall accuracy. CLDNN improves lower-SNR performance but also loses high-SNR accuracy. ResNet underperformed in this bounded experiment. These results are **experimental**, not independent satellite-capture validation or a justification for replacing production. Complete per-SNR/per-class results, calibrated versus raw probabilities, splits, seeds, configuration, environment and artifact hashes are in each versioned run and `experiments/SUMMARY.json`.

Actual new artifacts:

- `experiments/runs/rf-control-v1/random-forest.joblib`
- `experiments/runs/rf-awgn-v1/random-forest.joblib`
- `experiments/runs/resnet-synthetic-v1/best.keras` — 189,635 total / 188,483 trainable parameters
- `experiments/runs/cldnn-synthetic-v1/best.keras` — 23,747 total / 23,555 trainable parameters

RF checkpoint replay differs by at most 2.22×10⁻¹⁵; both neural checkpoint replays have zero probability difference from their saved outputs. The default NumPy interpreter genuinely serves the experimental RF when its recorded sklearn version matches. Neural online inference is explicitly Unavailable without optional compatible TensorFlow; their genuine stored experiment metrics remain available. No CNN-probability substitute is used.

## Executed validation

| Check | Actual final result |
| --- | --- |
| Python suite | 149 passed after the reliability follow-up |
| Frontend unit/route/API-transport suite | 35 passed |
| TypeScript API client → real Python model/DSP | 11 passed |
| TypeScript check | Passed |
| ESLint | 0 errors; 18 existing Fast Refresh export warnings |
| Production build | Passed, including the final requested galaxy/constellation code |
| Full production browser suite | 25 passed (4.2 minutes), against the compiled production app |
| Latest galaxy/globe browser regression | 12 passed against the final development source |
| Fresh CI-style production server/browser smoke | 4 passed; runner started and stopped its own compiled-app server |
| Python syntax/undefined-name CI lint | Passed, 0 errors |
| Scientific browser workflows | 5 passed before full regression |
| Production SPA/API serving | Passed: all 8 SPA routes, referenced assets, real model health and unknown API 404 |
| Keras/NumPy and gradient parity | Passed with actual TensorFlow 2.16.2 |
| Original artifact preservation | All six byte-identical |
| Legacy Streamlit | All 6 pages render without errors; real Arena benchmark also passed |

The browser scenarios include impairments → visualization → real CNN → class-specific explanation → export, real file upload/sample clock validation, GMSK/BFSK diagnostics, rate-aware two-signal comparison, real evaluation/experimental model evidence, model benchmarking, offline/invalid-file states, WebGL fallback, continuous globe rotation/interaction, all routes and responsive layouts. `validation-summary.json` collects the executed results. `changed-files.json` inventories every added/modified file and distinguishes the shared visual edits; `preserved-artifacts.json` records the six original hashes. Raw final logs and scientific screenshots are stored here alongside the report; responsive mission screenshots remain under `../validation/`. The older browser request-dispatch race was repaired by awaiting the actual response and validating its HTTP status/window before the UI assertion; both BPSK/QPSK checks passed. One follow-up attempt recorded a roughly 15-minute gap between browser events and timed out; the repeat completed in 31.2 seconds. Both logs are retained rather than concealing the failed attempt.

## Remaining limitations and final status

- **BLOCKED — original reproduction:** original training corpus, source groups, training configuration, split IDs/seeds and all 900 individual evaluation observations were not provided.
- **BLOCKED — validated production low-SNR improvement/calibration/OOD:** no sufficiently independent original-domain training/validation/test and labeled OOD data exists. Production remains uncalibrated; OOD remains not validated. A confident class can be wrong.
- **IMPLEMENTED — feasible training/evaluation:** real RF/residual/CLDNN architectures and checkpoints, grouped training, controlled augmentation, checkpointing, validation-only fitting, replay and metrics export work. The supplied toy corpus was not used for fitting.
- **NOT YET COMPLETED — Windows execution:** a Python 3.12 CI job and portable commands/paths are configured, but remote execution has not been observed in this session.
- The orbital/channel models remain educational approximations; fading is not a Doppler-correlated atmospheric channel and tracks are not live ephemerides. Semantic I/Q swaps and absolute RF power cannot be inferred from arbitrary sample arrays.
- History is process-local and browser preferences are local. Authentication/multi-user persistent storage were outside this local application audit.

## Run commands

From `satellite-modulation-classification/` after installation:

```bash
.venv/bin/python scripts/dev.py
```

The launcher prints actual free ports. Current development uses **http://localhost:5173**, backend **http://127.0.0.1:8003**, and API docs **http://127.0.0.1:8003/docs**; Docker occupies 8000–8002. For separate launch, set the Vite proxy to the API port (the ignored local configuration does so here).

```bash
# Production
npm --prefix frontend run build
.venv/bin/python scripts/serve.py

# Validation
.venv/bin/python -m pytest tests -q
npm --prefix frontend test
npm --prefix frontend run test:integration
npm --prefix frontend run typecheck
npm --prefix frontend run lint
# From frontend/ with installed Chrome and the current API:
SATMOD_TEST_API_URL=http://127.0.0.1:8003 PLAYWRIGHT_CHANNEL=chrome npm run test:e2e
```

The main README documents installation, optional TensorFlow, fresh external/grouped data training, checkpoint replay, OOD score validation and interpretation. Full file inventory: `changed-files.json`; preserved original artifacts: `preserved-artifacts.json`.

## Requested galaxy follow-up

The Earth remains the main illuminated subject. Its sky now includes a seeded Milky Way star band, restrained procedural nebula/dust lanes and fine decorative constellation graphs. Two corner patterns are positioned outside Earth's opening silhouette; the remaining patterns appear as the camera rotates. These are stylized scenery, not measured astronomical coordinates. Constellation links use one batched draw call. Background drift follows the auto-rotation control, clamps resume deltas, and stays stopped for hidden/reduced-motion views. No new dependency or network texture was added.

Visual evidence: `cosmic-mission-dashboard.png` and `cosmic-earth-detail.png`. Browser console and runtime checks are recorded in `cosmic-visual.log`. All 12 final development globe/responsive regressions passed, and the final compiled production scene also passed the rendering/visibility/sidebar/fullscreen probe.

## Reliability and CI follow-up

Controlled failure injection verified that an unexpected `/api/system/info` error returned HTTP 500 without an `Access-Control-Allow-Origin` header; a negative `Content-Length` also incorrectly reached the health endpoint with HTTP 200. The centralized error handler now preserves the configured origin allowlist (including configured wildcard behavior), varies explicit-origin responses correctly, and keeps private exception details in backend logs. The body limiter rejects signed, empty and malformed lengths before request processing. Nine regression cases cover these paths; the complete Python suite passed 149 tests.

The Playwright runner now has an explicit production mode with one self-managed FastAPI server serving the compiled frontend and API. It rejects conflicting API/frontend origins. A fresh `CI=1` run started that server and passed real WebGL, upload/CNN, measured benchmark and invalid-file scenarios (4 passed, 48.7 seconds), then shut the test server down. TypeScript and ESLint passed; Python CI's syntax/undefined-name lint reported zero errors. Full earlier browser/build evidence remains recorded separately above.

GitHub Actions now includes Windows Python 3.12 alongside Linux 3.10–3.12, uses `npm ci`, installs Chromium with its system dependencies, runs the compiled-app browser suite, and retains diagnostic artifacts. The configuration was syntax-checked locally; this is not a claim that unexecuted remote jobs passed. The original model and mission-control/galaxy source remain unchanged by this follow-up.
