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

The Overview includes an interactive, continuously rotating Earth with a restrained procedural galaxy, seeded stars and decorative constellation links. Earth stays the primary illuminated subject; the star patterns are scenery, not an astronomical catalog. Globe pause, reduced-motion preferences and hidden-tab handling also stop background drift.

## Workspaces

| Workspace | Connected behavior |
| --- | --- |
| Overview | Interactive 3D Earth, simulated spacecraft and matching ground tracks, real backend/model health, capture telemetry, actual classification probabilities and recent history |
| Live Classifier | Real CNN classification, saved sample filters, file uploads, explicit 128-sample window selection, probabilities, entropy, explicit trust status, genuine logit saliency/Integrated Gradients, scientific plots, JSON export including attribution |
| Satellite Studio | Nine modulation generators including rectangular 32-QAM, configurable GMSK BT/BFSK deviation/APSK rings, RRC, AWGN, Doppler/CFO, IID or Wiener phase noise, block Rician/Rayleigh fading, Saleh AM/AM + AM/PM distortion, explicit amplitude policy, CSV/audio download and capture handoff |
| Evaluation | Actual 900-observation confusion matrix, recomputed precision/recall/F1, SNR series, per-class counts, aggregate error analysis, JSON/CSV export |
| Model Arena | Actual registered-model timings with separate preprocessing, median/P95/hardware/batch size, artifact identity, honest availability and matched synthetic experiments |
| Signal Intelligence | Power, RMS, occupied bandwidth, I/Q moments, amplitude/phase histograms, FFT/PSD/STFT, rate-aware two-capture comparison with actual CNN predictions and timing |
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

Studio first calibrates transmitter RMS, applies the configured channel, and then either calibrates final capture RMS or preserves channel gain, as selected. Pre-calibration power stages and the exact channel order are included in the response. Its default is `sqrt(2) * training_std` to place the simulator near the training amplitude scale. This does not establish equivalence to the training distribution. Prediction accuracy on newly synthesized waveforms is not guaranteed. Captures far outside the training scale receive an amplitude warning.

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

The original corpus supplies only the CNN. The legacy Arena's probability transformations and fixed latency/accuracy comparisons have now been removed. Both interfaces use the real registry. Original-domain alternatives remain Not Trained until a compatible independently evaluated artifact exists. Separately named synthetic experiments contain genuinely trained candidates with their own artifacts, preprocessing, class mapping, recorded groups and metrics. Their evidence is explicitly limited to a generated educational domain; it is not comparable to the historical 900-observation score. A missing dependency or incompatible recorded model runtime produces an Unavailable state, never replacement CNN predictions.

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
| POST | `/api/benchmark` | Real registered-model timing, preprocessing separately; untrained models return 409 |
| POST | `/api/explain` | Genuine class-logit saliency or Integrated Gradients on the selected 128-sample window |
| POST | `/api/signal/upload` | Safely parse/analyze a capture without requiring CNN inference |
| GET | `/api/constellations` | Shared unit-energy transmitter symbol tables |
| GET | `/api/evaluation/diagnostics` | Fresh descriptive 300-frame demo diagnostics; no fitting |
| GET | `/api/experiments` | Published actual experimental reports and matched-test evidence |
| GET | `/api/history` | Most recent 100 classifications, in memory |
| GET | `/api/orbit` | Approximate educational orbital-pass geometry |

Request schemas, bounds, and response details are available through `/docs`. Bodies are bounded before parsing, including chunked requests. File decompression and array dimensions are checked before materializing uploaded captures. Model loading is cached, inference is serialized where required, CORS is configurable, and request logs include method/path/status/timing without capture contents.

## Scientific audit, uncertainty and explainability

See [the full audit and executed results](integration/audit/REPORT.md), [the implementation checklist](integration/audit/CHECKLIST.md) and [machine-readable controlled audit](integration/audit/model-diagnostics.json). The production model artifacts and supplied datasets were preserved.

The canonical input contract is real I-first `(N,2)` samples, a complete explicitly selected 128-sample window, float32 arithmetic, and the saved scalar z-score exactly once. Upload parsing may decode a complex vector or transpose a `(2,N)` capture; it never infers semantic channel order, pads a short capture, rescales uploaded power, or manufactures a class. React, Streamlit, benchmarks, comparisons and attributions use this contract. The original training source and split definitions are absent, so its provenance cannot be reconstructed completely. A controlled descriptive comparison on all **300 demo frames** measured **71.00%** with the existing z-score, **33.33%** with the legacy upload unit-power step before z-scoring, and **33.33%** with no transform. These values support retaining the existing model-compatible transform; they do not independently reproduce the stored 72.44% / 900-observation report.

Production confidence is **uncalibrated**. Entropy measures concentration within the three trained classes. Input validity means the numeric/shape/window checks passed; it does not certify modulation, channel order, or an in-distribution capture. OOD detection is **not validated**, with no detected/not-detected decision. No production temperature or OOD threshold was fitted to the demo library. Descriptive reliability diagrams, ECE, NLL and multiclass Brier scores are available in Evaluation. The initial descriptive demo ECE was 0.0867, NLL 0.6000, and Brier 0.3545 (mean summed class error, range 0–2). These are dataset-specific observations with unknown split independence.

Live Classifier exposes analytic class-logit saliency and [Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html). IG integrates along the path from a zero **raw** capture transformed by the same training statistics, using trapezoidal integration (16–256 steps, adaptive refinement). Its response reports the target class, baseline, signed I/Q contributions, calculation time and completeness residual. Saliency is a local derivative per raw sample unit. Neither method proves physical causality, classification correctness or OOD detection. Result export includes the latest computed explanation. Numerical gradient and completeness tests use the actual CNN weights.

[Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html) is implemented for experiments with a suitable independent validation split. The training pipeline fits it only on validation, then reports raw/calibrated metrics on the untouched test split. `scripts/evaluate_ood.py` accepts documented ID/OOD validation/test scores with source-group IDs, chooses a validation-only threshold, and reports test AUROC and frozen-threshold FPR/TPR. It does not activate a production detector.

## Simulator definitions and units

| Scheme | Actual implementation |
| --- | --- |
| BPSK, QPSK, 8-PSK | Explicit natural-binary symbol-table indexing; unit average ideal symbol energy |
| 16-QAM | Square 4×4 alphabet, normalized by √10 |
| 32-QAM | Rectangular 8×4 alphabet, normalized by √26; no cross-QAM/DVB claim |
| 16-APSK | 4+12 points; configurable increasing radii and angular offsets |
| 32-APSK | 4+12+16 points; the generator, reference API and visualizer share the same definition |
| BFSK | Two real frequency states ±Δf; default Δf=Rs/4, configurable deviation and continuous-phase option |
| GMSK | Binary NRZ → Gaussian frequency shaping → continuous phase integration, h=0.5; BT configurable 0.05–1 (default 0.3) |

The Gaussian FIR span grows for small BT; its normalized power -3 dB bandwidth is B=BT/T. Sustained GMSK bits produce ±π/2 phase per symbol. See the [modulator definition](https://www.mathworks.com/help/comm/ref/comm.gmskmodulator-system-object.html). CPM plots show trajectories, phase and instantaneous frequency, with no static symbol alphabet or fake linear symbol eye. Linear-modulation eye overlays use known transmitter timing; they do not imply receiver clock recovery.

The channel order is transmitter calibration → Saleh TWTA → flat Rician/Rayleigh fading → phase error → constant Doppler/CFO → complex AWGN → selected output calibration. Frequency parameters use Hz, or cycles/sample when Fs=1. FSK states and carrier offsets are checked against Nyquist. Rician K is linear; K=0 is Rayleigh when enabled. Fading is independent per configured sample block, not a Doppler-correlated scattering process. Phase error is either IID jitter (σ radians) or a Wiener walk (σ radians/sample increment). Independent seeded random streams make parameter comparisons reproducible. Unit-RMS output calibration removes absolute gain; select Preserve channel gain when studying amplifier/fading gain and inspect the recorded pre-calibration powers.

Arbitrary captures have no inferred SNR. Synthetic `realizedSnrDb` is known from the separate clean/noise components, not estimated from the mixture. PSD integrates to relative sample power; undefined zero-signal logarithmic power/bandwidth/phase and zero-variance moments are unavailable. The API bounds plotting arrays while full samples remain available. Comparison uses common physical axes when both sample rates are known, otherwise normalized axes and PSD units for both captures; no guessed resampling or time alignment is applied. The educational orbit pass now derives position, range, range rate, elevation, Doppler and latitude/longitude from the same spherical circular orbit with sidereal Earth rotation. It is explicitly simulated.

## Reproducible training and evaluation

The original training data, group identifiers, train/validation/test assignments, seeds and full training configuration were not supplied. The 300-frame demo file is explicitly refused for training even if renamed. The historical 900-frame evaluation cannot be independently rerun from aggregate files. Production weights are not automatically replaced.

An external dataset must be an `allow_pickle=False` NPZ with raw `signals (N,128,2)`, canonical string `labels`, numeric `snrs`, and original source-capture/session `groups`. Optional `splits` contain `train`, `validation`, `test`. Without them, the deterministic group split is approximately 60/20/20. Source groups and exact duplicate frames cannot cross partitions. All classes must occur in each split. The provider remains responsible for meaningful grouping and provenance. Normalization fits training data only; augmentation affects training only; checkpoint/LR/temperature selection uses validation only; final metrics use test only.

```bash
# Create a new explicitly synthetic educational dataset (never the supplied demo).
python scripts/create_synthetic_dataset.py --output experiments/data/my-synthetic-v2.npz --sessions-per-class 100 --seed 2026

# Real RF baseline; each output directory must be new.
python scripts/train.py --dataset experiments/data/my-synthetic-v2.npz --config experiments/configs/random-forest.json --output experiments/runs/my-rf-v2

# Replay a saved candidate on its recorded untouched test split, without refitting.
python scripts/evaluate.py --run experiments/runs/my-rf-v2 --dataset experiments/data/my-synthetic-v2.npz --output experiments/runs/my-rf-v2/replay.json
```

Keras CNN, residual CNN and CLDNN (convolutions → dense per time step → LSTM → classifier) architectures are in `training/models.py`. Install optional `requirements-training.txt` in a TensorFlow-supported Python environment; the pinned Intel macOS configuration is `requirements-training-macos-intel.txt` with Python 3.12. Then use `experiments/configs/cnn.json`, `resnet.json` or `cldnn.json`. Deep training supports class-balanced loss, optional controlled AWGN/phase/CFO augmentation, dropout, validation-loss early stopping, learning-rate reduction, checkpoint saving, and deterministic seeds. Outputs include preprocessing statistics, split indices, dataset/checkpoint hashes, configuration, environment, test probabilities and metrics. `experiments/rf-environment.txt` records the RF artifact runtime. Trusted experimental pickles are hash-checked and version-checked; uploaded pickle/object payloads remain forbidden.

Completed synthetic RF runs are in `experiments/runs/rf-control-v1` and `rf-awgn-v1`, using 1,800 frames from 300 independent seeded sessions and an identical 360-frame test partition. The control RF measured **66.94%**, AWGN-augmented RF **66.67%**, and the unchanged CNN **65.83% on that same synthetic test**. At −6 dB, augmentation improved RF accuracy from 48.33% to 56.67%, while +6 dB fell from 76.67% to 70.00% and overall accuracy fell by one correct frame. This does not justify production replacement. These scores must not be compared directly with the historical 72.44% score from a different domain. The experimental RF is separately identified as `rf-experimental`; original-domain alternatives remain unavailable until genuine compatible evidence exists.

Two real bounded neural experiments were also completed for eight epochs using the same source groups and untouched test frames. `resnet-synthetic-v1` measured **49.44%** and `cldnn-synthetic-v1` **68.89%**, versus the same CNN **65.83%** baseline. CLDNN improved low-SNR accuracy but reduced high-SNR accuracy; ResNet underperformed in this bounded run. Neither was promoted. Both have real `.keras` checkpoints, separate normalization, recorded validation-selected checkpoints/temperatures, and replayable test probabilities. They appear under separate synthetic experiment IDs. On the default NumPy-only interpreter, their online inference is explicitly Unavailable until compatible optional TensorFlow dependencies are installed; their genuine stored experiment evidence remains visible.

Reproduce the preprocessing/engine audit and measure this host with:

```bash
python scripts/audit_model.py
python scripts/profile_services.py
```

The controlled implementation comparison found all 300 class decisions identical and maximum probability difference 5.81×10⁻⁷ after convolution vectorization. In the same 50-repetition experiment, median inference changed from 4.76 ms (reference loop) to 1.26 ms (vectorized), and p95 from 9.03 to 4.11 ms. These are host-dependent measurements, not fixed latency guarantees.

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

Playwright starts both TCP servers, or reuses running services outside CI. Its browser tests cover the seven signal workflows, real WebGL rendering, continuous rotation, camera and satellite controls, reduced motion, backend failures, WebGL fallback, and layouts from 390px to 1440px. To test existing services on alternate ports with installed Chrome:

```bash
SATMOD_TEST_API_URL=http://127.0.0.1:8003 PLAYWRIGHT_CHANNEL=chrome npm run test:e2e
```

`SATMOD_TEST_BASE_URL` overrides the default frontend URL. Omit `PLAYWRIGHT_CHANNEL` to use Playwright's installed Chromium. See [the actual validation record](integration/VALIDATION.md) for executed results and remaining limits.

For browser tests against the compiled application, build first and select a free port. Production mode starts one FastAPI server serving both API and frontend, so the two URLs must have the same origin:

```bash
npm run build
SATMOD_TEST_PRODUCTION=1 SATMOD_TEST_API_URL=http://127.0.0.1:8100 SATMOD_TEST_BASE_URL=http://127.0.0.1:8100 npm run test:e2e
```

CI runs the Python suite on Linux (3.10–3.12) and Windows (3.12), then checks frontend types/lint/unit tests, real client contracts and the production build on Linux. Its Chromium job exercises the compiled application, enables software WebGL when no GPU is available, and retains failure traces/screenshots. These jobs are configured; local validation does not establish that the remote Windows/Linux jobs have passed.

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

The original sibling `pixel-perfect-replica/` folder is unchanged and is not needed to run this repository. No nested Git repository was copied. Existing uncommitted Python changes were preserved; an ignored local rollback patch is in `integration/pre-integration-working-tree.patch`. The original integrated application is preserved at commit `ed8a78a`; the scientific audit is isolated on `feat/scientific-reliability-audit` without rewriting history.

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
