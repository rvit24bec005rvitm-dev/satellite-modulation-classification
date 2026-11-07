# Actual validation record

The latest scientific audit and expanded regression results are in [audit/REPORT.md](audit/REPORT.md). The mission-control implementation record below is retained as earlier validation evidence.

The integrated development application responds at **http://localhost:5173**, with the real NumPy CNN backend at **http://127.0.0.1:8003**. Docker reserves ports 8000–8002; the combined launcher selects available ports and configures the proxy automatically. A separate built-app test server used port 8010 during verification.

## Mission-control redesign

The Overview's static hero has been replaced by the dense mission dashboard. The shared shell retains all eight TanStack routes. The new Earth uses Three.js, React Three Fiber, and drei; its day/night/normal/specular textures, cloud layer, atmospheric shader, Keplerian orbital paths, clickable demo spacecraft, and station links are rendered in WebGL. The Leaflet map uses bundled Natural Earth boundaries and the same ECI/ECEF calculations as the globe. Simulation is labeled throughout. Real health, capture analysis, stored accuracy, latest classification, and history remain API driven; unavailable measurements stay unavailable.

| Check | Actual result |
| --- | --- |
| `python -m pytest tests -q` | **66 passed**, including actual model, preprocessing, uploads, API validation, and DSP |
| `npm test` | **32 passed**, including ten orbital/rotation mathematical checks |
| `npm run test:integration` | **8 passed**, actual TypeScript API client → FastAPI → real CNN/DSP |
| `npm run typecheck` | **Passed**, official complete TanStack declarations; no permissive type shims |
| `npm run lint` | **0 errors**, 18 existing React Fast Refresh export warnings |
| `npm run build` | **Passed**, including separate Earth, map, and Plotly chunks |
| Production Playwright suite | **20 passed**, built SPA served with FastAPI on port 8010 (Chrome, 2.6 min) |
| `python scripts/check_production.py` | **Passed**, all eight compiled SPA routes, index assets, real health, and API 404 |
| Browser supplemental probes | Hidden-tab pause/resume without catch-up, fullscreen enter/exit, sidebar collapse/expand, no runtime errors; connected → offline → reconnected status polling |
| Original preservation | Original Lovable sibling clean; backend/ML artifacts and legacy Streamlit retained |

The ten mathematical tests verify the three circular orbit radii/periods, matching globe/map coordinates, sidereal longitude drift, antimeridian splitting, station visibility, and frame-rate-independent rotation at 20/30/60/120 FPS. Display rotation uses 108 seconds per revolution; physical propagation runs at 30× simulated time.

The browser suite checks actual WebGL and local geographic assets, automatic rotation, pause/resume, drag interruption, wheel/button zoom, camera reset, marker and selector interaction, real-result probability handoff, reduced motion, backend offline behavior, WebGL fallback, and widths 1440/1024/820/390px. Responsive assertions check horizontal overflow, panel overlap, clipped text, and mobile navigation. The tablet breakpoint was moved to 1100px after a real 1024px clipping failure. The mobile camera now fits the entire Earth. Scientific-chart checks allow the separately loaded Plotly bundle to finish; an initial linked-UMD loader issue was fixed in Vite's dependency and CommonJS configuration.

## End-to-end signal workflows

| Workflow | Actual browser verification |
| --- | --- |
| Valid I/Q upload | Saved frame → real CSV upload → CNN probabilities/entropy/latency → scientific charts → exported result |
| BPSK synthesis | 1,024 samples at 18 dB → Studio → unchanged full capture sent to CNN |
| QPSK synthesis | 1,024 samples at 6 dB → Studio → unchanged full capture sent to CNN |
| Evaluation | Stored 72.44% / 900 observations, confusion matrix, SNR/per-class charts and row percentages |
| Arena | Real CNN benchmark latency/throughput; three alternative models remain Not Trained |
| Backend failure | Explicit offline state, no fabricated probabilities or successful inference |
| Invalid upload | Malformed NPY rejected with a useful visible error |
| Navigation | All eight routes plus mobile sidebar navigation |

## Rendering performance and limits

A four-second desktop headless Chrome sample on Intel Iris Plus/ANGLE Metal measured **28.5 requestAnimationFrame callbacks/s**, with a 33.5ms P95 interval. The same browser's blank-page baseline measured **30.0 callbacks/s**. These are observed browser cadence values, not a guarantee of 60 FPS on other hardware. The scene caps DPR at 1.5, bounds geometry, lazily loads its bundle, and stops rendering while hidden. Raw observations are in `validation/mission-browser-probe.json` and `validation/mission-raf-baseline.json`. A separate production connectivity probe confirmed that an established connection switches offline in 11.1 seconds after blocking API traffic and reconnects in 9.9 seconds after restoration, with no fabricated probabilities (`validation/mission-connectivity-probe.json`).

The orbital scenario is an educational circular two-body simulation, not live spacecraft tracking. The RF capture and simulated spacecraft are independent. A browser without WebGL receives a labeled static map fallback. Windows startup is documented but was not executed here. The supplied reference screenshots were not accessible as image attachments in this session; the implementation follows the detailed written layout and palette.

Screenshots: `validation/mission-1440.png`, `mission-1024.png`, `mission-820.png`, and `mission-390.png`. Raw final check logs use the `mission-` prefix in `validation/`. The original restricted-environment report is archived in `VALIDATION.initial.md`; its earlier missing-types/network/browser failures were superseded by the executed checks above. The initial Keras parity check remains valid: 43 real frames, 100% class agreement, maximum absolute probability difference 5.662441253662109e-7.

## Data and product limitations

- Only BPSK, QPSK, and 16-QAM are trained CNN classes. Other Studio modulations are for simulation/analysis. The classifier is a closed-set model; entropy does not reliably identify unknown modulations.
- ResNet, CLDNN, and Random Forest have no independent supplied checkpoints. They remain Not Trained; the trusted model registry can accept real future loaders, preprocessing, and independently measured evaluations.
- The 900-observation evaluation has aggregate counts and rounded SNR percentages. The demo library contains 300 frames. Full split provenance and individual errors for the 900-observation dataset are unavailable; no SNR-sliced full-dataset matrix is invented.
- Arbitrary capture noise/SNR is not uniquely identifiable without a reference. Those metrics remain unavailable. Synthetic waveforms may differ from the training distribution despite explicit RMS calibration.
- History is the most recent 100 classifications in backend memory and clears on restart. Preferences are browser-local.
- Orbital/channel simulations are educational approximations. Legacy Streamlit remains available, including historical simulated Arena comparisons documented in README. Its existing user edits were preserved.
- The existing working tree contains pre-existing trailing-whitespace findings; these were preserved along with the user's legacy edits.
