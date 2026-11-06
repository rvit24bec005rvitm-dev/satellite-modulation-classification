# SatModClass — Satellite Signal Intelligence

A unified React + FastAPI application for classifying and inspecting complex baseband satellite signals. The React interface preserves the Lovable mission-control design. The supplied trained CNN recognizes **BPSK, QPSK, and 16-QAM only**. Streamlit remains available as a legacy reference.

## Local installation

Use Python 3.10+ (3.11 or 3.12 recommended), Node.js **22.12+**, and npm. TensorFlow is optional; normal application inference uses the supplied NumPy weights.

From this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
cd frontend
npm install
cd ..
python scripts/dev.py
```

Open **http://localhost:5173**. FastAPI is at **http://localhost:8000** and its interactive documentation is at **http://localhost:8000/docs**. The development launcher automatically selects a free nearby port if a default is occupied, configures the frontend proxy, and prints the actual URLs only after both services respond. It stops both processes when interrupted. Set `SATMOD_API_PORT` and `SATMOD_FRONTEND_PORT` to require specific ports.

Windows PowerShell uses:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
cd frontend
npm install
cd ..
python scripts/dev.py
```

To launch services separately, activate the Python environment, run the following from the repository, and run the npm command in another terminal:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```bash
cd frontend
npm run dev
```

`frontend/.env.example` documents `VITE_API_BASE_URL`. Leave it blank to use the Vite `/api` proxy. Settings can override the API base URL for the current browser. A separate backend origin must allow the frontend origin in `SATMOD_CORS_ORIGINS`. `scripts/dev.py` reads the root `.env`; direct Uvicorn launches can use `--env-file .env` or exported environment variables.

## Production build and local serving

```bash
cd frontend
npm run build
cd ..
python scripts/serve.py
```

This serves the compiled frontend and API together at **http://localhost:8000**. All eight frontend routes have a SPA fallback. Frontend hosting elsewhere must provide a history fallback and a backend URL. The development interface and API are intended for a trusted local environment; authentication and multi-user storage are not included.

## Workspaces

| Workspace | Connected behavior |
| --- | --- |
| Overview | Interactive 3D Earth, simulated spacecraft and matching ground tracks, real backend/model health, capture telemetry, actual classification probabilities and recent history |
| Live Classifier | Real CNN classification, saved sample filters, file uploads, explicit 128-sample window selection, probabilities, entropy, scientific plots, JSON export |
| Satellite Studio | Eight modulation generators, RRC, AWGN, Doppler/CFO, phase jitter, Rician/Rayleigh fading, Saleh AM/AM + AM/PM distortion, CSV/audio download, capture handoff |
| Evaluation | Actual 900-observation confusion matrix, recomputed precision/recall/F1, SNR series, per-class counts, aggregate error analysis, JSON/CSV export |
| Model Arena | Actual warm CNN inference timings, median/P95/throughput, artifact parameter count, truthful alternative-model availability, benchmark export |
| Signal Intelligence | Power, RMS, occupied bandwidth, I/Q moments, amplitude/phase histograms, FFT/PSD/STFT, two-capture comparison |
| Satellite Theory | Modulation equations, ideal constellation explorer, sample-SNR/noise calculations, approximate LEO Doppler pass, Saleh transfer curves, CNN explanation |
| Settings | API status/address, persisted theme/chart preferences, default signal parameters, loaded model metadata, version |

Scientific views use a locally bundled Plotly.js build, including zoom/pan/export controls. Waveform and constellation previews are limited to 1,024 points; full-resolution samples remain available for processing and transfers. A spectrogram has a bounded number of time bins. Eye traces appear only for generated signals with known transmitter symbol timing; they are not a recovered receiver eye diagram.

## Mission-control dashboard

The Overview uses a compact three-column operational layout, a collapsible sidebar, and a UTC header. Its Three.js / React Three Fiber globe includes local 2K Earth textures, clouds, night lights, atmospheric shading, three selectable simulated spacecraft, orbital paths, and an animated station link when the spacecraft clears the 10° elevation mask. Drag to orbit the camera, scroll or use the buttons to zoom, reset the camera, or pause automatic rotation. Global View restores the camera. Reduced-motion preferences disable automatic animation initially; unsupported or lost WebGL contexts show an explicitly labeled static geographical fallback.

The display turntable rotates once per **108 seconds**, using frame delta time. Circular Keplerian orbits propagate at **30× simulated time**; Earth uses sidereal rotation for the matching ECEF ground tracks on the local Leaflet map. This educational two-body scenario omits perturbations and uses a simulated Bengaluru ground station. All spacecraft, station geometry, and links are labeled simulated; they are independent of saved or generated RF captures. Dragging temporarily pauses propagation and display rotation, and both resume after interaction. Hidden tabs suspend the scene without a catch-up jump.

Telemetry comes from the active capture or the indicated saved demo frame. Probabilities appear only after a real classification in the current browser session. Unmeasured RF link strength and carrier frequency remain unavailable. Backend/model status is polled; a backend outage stays visible while the independent orbital demonstration remains usable. Earth textures and country boundaries are bundled locally, with upstream references in `frontend/public/textures/README.md`; no live imagery or tracking service is implied.

The globe and map are loaded separately from the other workspaces. Rendering caps the device pixel ratio at 1.5, uses bounded sphere geometry and a 600-point starfield, and avoids postprocessing. On smaller screens, the panels stack around a full-width globe.

## Input and preprocessing contract

The model consumes `(128, 2)` arrays with **I first, Q second** and class order `BPSK, QPSK, QAM16` (displayed as `16-QAM`). Every classification source uses the same saved scalar transform:

```
model_input = (raw_window - saved_mean) / saved_std
```

The mean/std come from `normalization_stats_all_snr.npz`. Captures are **not** silently normalized to unit power, resampled, reshaped arbitrarily, or padded. Longer captures retain all samples; a selected complete 128-sample window is classified. Captures shorter than 128 samples can be analyzed but cannot be classified.

Studio generates a unit-power waveform and explicitly scales the output to a selected RMS amplitude. Its default is `sqrt(2) * training_std` to place the simulator near the training amplitude scale. This does not establish equivalence to the training distribution. Prediction accuracy on newly synthesized waveforms is not guaranteed. Captures far outside the training scale receive an amplitude warning.

Supported uploads (4 MiB default, 65,536 samples maximum):

- `.npy`: numeric `(N,2)`, `(2,N)`, or a complex vector.
- `.npz`: one numeric array, or an `iq`, `samples`, or `signal` capture array. A multi-frame dataset is not a single capture.
- `.csv`: exactly two numeric columns, optionally headed `I,Q`.
- `.mat`: numeric capture; name the variable `iq`, `samples`, or `signal` when the file has other variables. MAT v5 is supported; v7.3/HDF5 is rejected.
- `.wav`: stereo I/Q or mono I with zero Q. Integer PCM is converted to its standard floating full-scale convention; no training-scale recalibration is inferred.
- `.json`: numeric pairs or `{i,q}` samples, optionally inside a `samples` property.

NaN/Inf, object/pickle arrays, excessive dimensions, oversized bodies, large expanded archives, and malformed files are rejected. The upload filename is never used as a filesystem destination. No uploaded code is executed.

Power and PSD use relative sample units unless the capture is calibrated. Arbitrary captures do not identify noise power or SNR uniquely; those values remain unavailable rather than being invented. Zero-variance skew/kurtosis and zero-signal PAPR are undefined and display as unavailable.

## Model and evaluation integrity

The NumPy inference engine was corrected to match the saved Keras architecture: **same** Conv1D padding, and **ReLU before batch normalization**. The original NumPy implementation used valid padding and reversed activation/normalization. Original weight files were not changed.

The independent check in `scripts/verify_keras_parity.py` loads the saved Keras model weights and compares real TensorFlow outputs to NumPy. Legacy disabled configuration fields and initializer metadata are removed from an in-memory configuration clone for TensorFlow 2.16 compatibility; the archive stays unchanged.

`evaluation_results.json` contains a 900-observation confusion matrix. Its trace is 652, giving **72.4444% accuracy**. Precision, recall, and F1 are recomputed from those counts. `snr_accuracy_results.csv` contains 20 SNR bins and rounded source percentages. The included demo NPZ is a **300-frame subset**, not the complete evaluation dataset. Split provenance and per-observation errors for the full evaluation were not supplied.

The trusted-code extension point is `backend/services/model_registry.py`: register a real loader, its own preprocessing function, and an optional independent evaluation provider. This can enable later model benchmarks without changing the frontend. Uploaded files cannot register loaders.

ResNet, CLDNN, and Random Forest have no supplied trained artifacts. Their predictions/accuracy/latency are unavailable in the integrated app. The original Streamlit Arena's alternative outputs are transformations of CNN probabilities, not independently evaluated models. Treat the legacy Arena's comparison values as historical illustrations.

The generator now uses true continuous-phase BFSK/GMSK instead of BPSK aliases, and the 32-APSK reference has all 32 points. APSK references and generators share unit-average-energy definitions. The PSD uses `|FFT|²/(N*Fs)`, so integration over frequency recovers signal power. The channel simulator is educational: it uses constant short-window frequency offsets and independent fading/phase jitter, not a full standards-compliant satellite propagation model.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Load/check model and training stats; 503 on failure |
| GET | `/api/system/info` | Model contract, app metadata, limits |
| GET | `/api/models` | Actual model registry and parameter counts |
| POST | `/api/classify` | JSON I/Q classification with optional window offset |
| POST | `/api/classify/upload` | Multipart capture parsing + classification |
| GET | `/api/samples` | Saved sample metadata |
| GET | `/api/samples/{id}` | Full capture and bounded analysis |
| POST | `/api/signal/generate` | Reproducible modulation/channel generation |
| POST | `/api/signal/analyze` | Scientific signal diagnostics |
| POST | `/api/signal/compare` | Two-capture diagnostics and unaligned overlap MSE |
| POST | `/api/signal/audio` | Audible heterodyne WAV (illustration, not decoded payload) |
| GET | `/api/evaluation` | Recomputed stored evaluation statistics |
| GET | `/api/evaluation/snr` | Stored SNR evaluation series |
| POST | `/api/benchmark` | Real CNN latency measurement; alternatives return 409 |
| GET | `/api/history` | Most recent 100 classifications, in memory |
| GET | `/api/orbit` | Approximate educational orbital-pass geometry |

Request schemas, bounds, and response details are available through `/docs`. Bodies are bounded before parsing, including chunked requests. File decompression and array dimensions are checked before materializing uploaded captures. Model loading is cached, inference is serialized where required, CORS is configurable, and request logs include method/path/status/timing without capture contents.

## Tests

With the Python environment activated:

```bash
python -m pytest tests -q
cd frontend
npm run typecheck
npm run lint
npm test
npm run test:integration
npm run build
npx playwright install chromium
npm run test:e2e
```

`npm test` covers all file routes, API errors/uploads/cancellation, rendered loading/error/metric components, and orbit/ground-track mathematical consistency. `test:integration` runs the actual TypeScript API client against the real FastAPI app in a separate Python process via an ASGI TestClient transport. It uses no mocked model predictions, evaluation scores, or DSP output and needs no listening socket. Set `SATMOD_PYTHON` if the environment is not at `.venv`.

Playwright starts both TCP servers, or reuses running services outside CI. Its 20 tests cover the seven signal workflows, real WebGL rendering, continuous rotation, camera and satellite controls, reduced motion, backend failures, WebGL fallback, and layouts from 390px to 1440px. To test existing services on alternate ports with installed Chrome:

```bash
SATMOD_TEST_API_URL=http://127.0.0.1:8003 PLAYWRIGHT_CHANNEL=chrome npm run test:e2e
```

`SATMOD_TEST_BASE_URL` overrides the default frontend URL. Omit `PLAYWRIGHT_CHANNEL` to use Playwright's installed Chromium. See [the actual validation record](integration/VALIDATION.md) for executed results and remaining limits.

Optional independent Keras parity check (requires TensorFlow + h5py in that interpreter):

```bash
python scripts/verify_keras_parity.py
```

## Architecture and preserved files

```text
satellite-modulation-classification/
├── frontend/                  # Integrated Lovable UI, Vite, TanStack file routes
│   ├── src/components/        # Existing UI primitives, shared shell and plots
│   ├── src/pages/             # Eight connected workspaces
│   ├── src/hooks/             # React Query operations, shared capture/preferences
│   ├── src/lib/               # Typed API client and circular-orbit mathematics
│   ├── public/                # Bundled Earth textures and geographical boundaries
│   ├── src/routes/            # Directly addressable TanStack routes
│   ├── e2e/                   # Browser workflows
│   ├── scripts/               # Route-tree generation
│   └── vendor/                # Plotly.js distribution with upstream license
├── backend/
│   ├── main.py                # FastAPI endpoints, CORS, errors, optional SPA hosting
│   ├── api/schemas.py          # Typed/bounded request and response contracts
│   ├── core/config.py         # Environment configuration
│   └── services/              # Classification, capture parsing, analysis, evaluation
├── scripts/                   # Dev/production launch, API test bridge, Keras parity
├── integration/               # Audit, feature mapping, actual validation evidence
├── utils/                     # Preserved ML/DSP/Plotly helpers, correctness fixes
├── tests/                     # Original and new Python tests
├── models/cnn_weights.npz     # Preserved model weights
├── pages/                     # Preserved legacy Streamlit workspaces
├── app.py                     # Preserved legacy Streamlit entry
├── classifier_samples.npz
├── normalization_stats_all_snr.npz
├── satellite_modulation_cnn_all_snr.keras
├── evaluation_results.json
└── snr_accuracy_results.csv
```

The copied frontend runs as a local SPA rather than through the original Lovable/Cloudflare server wrapper. TanStack Router and file-based route sources are retained. `frontend/scripts/generate-routes.mjs` regenerates the flat route tree before development, builds, tests, and type checks; edit route sources, not the generated file. It rejects duplicate paths. New nested/dynamic route requirements should use TanStack's official generator/plugin rather than expanding this flat-route helper.

The original sibling `pixel-perfect-replica/` folder is unchanged and is not needed to run this repository. No nested Git repository was copied. Existing uncommitted Python changes were preserved; an ignored local rollback patch is in `integration/pre-integration-working-tree.patch`. Work is on `feat/satmod-fullstack-integration` without rewriting published history.

Legacy launch:

```bash
streamlit run app.py
```

## Troubleshooting

- **Backend disconnected:** start Uvicorn, inspect `/api/health`, check the Settings URL and browser/backend origin. The UI never generates fallback predictions.
- **Missing model/stats:** restore the supplied artifacts in the paths above; model errors return 503. `SATMOD_MODEL_PATH` can point at a real compatible artifact.
- **Upload rejected:** use one capture, two numeric channels or a complex vector, at least 128 samples for classification, finite values, and the upload limit. WAV full-scale amplitudes may require explicit receiver calibration before classification.
- **Port already in use:** stop the existing process or change the backend port, frontend proxy, and API address together.
- **Sandbox EPERM:** run development/browser commands from an ordinary terminal that permits local TCP sockets and browser processes. ASGI integration tests can still validate the backend/client contract without sockets.
- **Offline dependency/type failures:** run a clean `npm install` with registry access. Stripped copies of production libraries may contain JavaScript but omit TypeScript declarations; do not replace missing router types with `any` shims.
- **Charts:** Plotly is bundled locally. Web fonts have system fallbacks when offline. Large Plotly code is loaded separately from the application chunk.
