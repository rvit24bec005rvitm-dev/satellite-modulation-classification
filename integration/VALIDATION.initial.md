# Actual validation record

Latest live retry after local networking/process permissions were enabled: the frontend responds at http://localhost:5173 and the backend runs on http://127.0.0.1:8003. Docker reserves ports 8000–8002. Both direct frontend HTTP and proxied model health checks passed. Installed Chrome rendered the overview with no runtime errors, then successfully generated BPSK at 18 dB, transferred the full 256-sample capture, and displayed a real CNN classification. Screenshots and machine-readable results are in `validation/running-overview.png`, `validation/running-classifier.png`, and `validation/live-retry.json`. The launcher now selects available ports and prints READY only after both services respond. The full browser suite and strict TypeScript check were not rerun during this connection retry.

The tables below preserve the **initial restricted-environment validation**. The initial TCP/browser restrictions no longer describe the running services.

| Check | Actual result |
| --- | --- |
| `python -m pytest tests -q` | **66 passed**; original tests plus real model/API/upload/DSP/security coverage |
| `npm test` | **22 passed** across eight routes, API transport/errors, and server-rendered feedback components |
| `npm run test:integration` | **8 passed** against the real Python app/model/DSP through the actual TypeScript API client |
| `npm run build` | **Passed**, Vite 7.1.12 production output in `frontend/dist` |
| `npm run lint` | **0 errors**, 18 React Fast Refresh export warnings |
| Independent TensorFlow/Keras comparison | **43 frames**, 100% class agreement, maximum absolute probability difference **5.662441253662109e-7** |
| Model weight identity | NumPy NPZ weights exactly equal weights extracted from original Keras archive |
| `python scripts/check_production.py` | **Passed**: all eight compiled SPA routes, JS/CSS assets, real API health, and unknown-API 404 through FastAPI TestClient |
| Original asset preservation | Six model/data/evaluation artifacts unchanged from Git; original Lovable folder clean; legacy working edits preserved |
| Python syntax compilation | `python -m compileall -q backend scripts utils` passed |
| `npm run typecheck` | **Failed**: locally recovered TanStack JavaScript package omits its `.d.ts` files; missing declarations and downstream contextual typing errors remain |
| `npm run test:e2e` | **Blocked before any browser test**: backend TCP bind on 127.0.0.1:8000 denied with EPERM |
| Separate development-server launches | Both FastAPI and Vite initialization attempted; TCP binds on 8000/5173 denied with EPERM |
| Browser launch probe | Cached Playwright browser absent; installed system Chrome launch aborts under sandbox restrictions |
| Clean dependency installation | npm/PyPI network requests denied. Escalation was rejected because sandbox approvals are disabled. Local packages/caches enabled executed build/tests; a clean online install was not verified |

Raw final logs, with the workspace path normalized, are in `validation/`. Machine-readable results are in `validation-summary.json`; model comparison results are in `keras-parity.json`. `preservation.json` records SHA-256 artifact hashes and preservation assertions.

## Requested workflows

| Workflow | Executed verification | Browser status |
| --- | --- | --- |
| 1. Valid upload → CNN → probabilities/analysis | Real saved CSV frame through TypeScript client, Python multipart parser, real CNN, actual FFT/waveform response. Python tests also cover NPY/NPZ/MAT/WAV/JSON and complex/transposed forms | Not executed |
| 2. Generate BPSK → classify | Actual 1,024-sample generated capture transferred unchanged to the real model through TypeScript client; response contract and preprocessing window verified | Not executed |
| 3. Generate QPSK at selected SNR → classify | Same real contract workflow; Python tests verify SNR/impairment controls change samples reproducibly | Not executed |
| 4. Evaluation counts, metrics, SNR curves | Actual files loaded; 900 counts and trace/precision/recall/F1/SNR arithmetic checked by Python and TypeScript client | Not executed |
| 5. CNN benchmark | Real timed iterations; latency arrays, mean, median/P95, throughput checked; untrained models rejected | Not executed |
| 6. Backend disconnection | TypeScript transport test rejects a failed connection without manufacturing a result; rendered error/empty components and offline health behavior tested separately | Full disconnect/visual flow not executed |
| 7. Invalid upload | Actual malformed NPY request through TypeScript client and real backend; useful error propagated; Python tests cover malformed, oversized, pickle and compressed-archive cases | Not executed |

These are client/backend integration checks, **not browser end-to-end passes**. The Playwright suite covers the seven workflows plus narrow viewport/navigation checks but could not get past server startup here.

## Remaining acceptance checks

1. Perform a clean `npm install` with registry access, then rerun `npm run typecheck` using the complete published TanStack packages. No permissive router typing shim was added to conceal missing declarations.
2. Run `npx playwright install chromium` and `npm run test:e2e` in a terminal/environment that permits local TCP ports and Chrome processes. Inspect narrow/wide layouts and chart controls there.
3. Verify a clean dependency installation and Windows startup in those environments. Windows commands and relative paths are provided; Windows execution was not tested here.

## Data and product limitations

- Only BPSK, QPSK, and 16-QAM are trained CNN classes. Other Studio modulations are for simulation/analysis. The classifier is a closed-set model; entropy does not reliably identify unknown modulations.
- ResNet, CLDNN, and Random Forest have no independent supplied checkpoints. They remain Not Trained; the trusted model registry can accept real future loaders, preprocessing, and independently measured evaluations.
- The 900-observation evaluation has aggregate counts and rounded SNR percentages. The demo library contains 300 frames. Full split provenance and individual errors for the 900-observation dataset are unavailable; no SNR-sliced full-dataset matrix is invented.
- Arbitrary capture noise/SNR is not uniquely identifiable without a reference. Those metrics remain unavailable. Synthetic waveforms may differ from the training distribution despite explicit RMS calibration.
- History is the most recent 100 classifications in backend memory and clears on restart. Preferences are browser-local.
- Orbital/channel simulations are educational approximations. Legacy Streamlit remains available, including historical simulated Arena comparisons documented in README. Its existing user edits were preserved.
- The existing working tree contains pre-existing trailing-whitespace findings; these were preserved along with the user's legacy edits.
