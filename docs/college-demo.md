# SatModClass: college demonstration, submission and viva guide

SatModClass is a local signal-analysis and modulation-classification laboratory. It combines a React interface, a Python/FastAPI backend, real trained model artifacts, synthetic waveform generation, scientific plots and persistent research reports. A hosted website or cloud account is not required for a college demonstration.

For a slower introduction with diagrams and a quiz, open the [standalone teaching guide](satmodclass-teaching-guide.html). That HTML guide works offline. This document is the practical presentation and submission companion.

## Start the project locally

Run commands from the extracted project/repository root, where `requirements-web.txt`, `scripts/` and `frontend/` exist. Install Python and, when building the frontend from source, Node.js **22.12+** with npm. Python 3.12 is a suitable starting point for the pinned web requirements. The normal application runs all four trained models without TensorFlow.

### Build from source on macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-web.txt
npm --prefix frontend ci
npm --prefix frontend run build
python scripts/serve.py
```

### Build from source in Windows PowerShell

These commands use the environment's Python directly, so PowerShell activation-policy changes are unnecessary.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
.\.venv\Scripts\python.exe scripts/serve.py
```

Open the URL printed by the launcher, normally **http://localhost:8000**. The server binds to the local computer. If its default port is occupied, the launcher selects a free nearby port and prints it. Stop the server with **Ctrl+C**.

### Use a prebuilt college release

`scripts/serve.py` serves the API and an **already-built `frontend/dist`** together; it does not build React. If the prepared college release includes `frontend/dist`, install the Python requirements and run `scripts/serve.py` directly. Node/npm are only needed when rebuilding that frontend. On macOS/Linux, the compiled-release launch is:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-web.txt
.venv/bin/python scripts/serve.py
```

For a prebuilt release in Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe scripts/serve.py
```

Opening `frontend/dist/index.html` as a file does not start the Python API. Create a new virtual environment on each computer; do not copy `.venv` between operating systems.

For code development after installing the Python and npm dependencies, run `python scripts/dev.py` using the activated environment. It starts Vite and FastAPI, prints their actual URLs, and stops both on Ctrl+C. Use the compiled path for the presentation.

Local use needs no token by default. Keep `SATMOD_PUBLIC` unset or `0` for this local workflow. An existing configured access token will cause the unlock screen to appear. Browser Settings can override the API address; a blank address uses the local same-origin API. Workspace data defaults to `.satmod/`; an optional `SATMOD_DATA_DIR` selects a separate demo folder.

## Understand the three different counts

**Nine generators, four runnable trained artifacts, three classification labels** describe different capabilities.

The generator supports **BPSK, QPSK, 8-PSK, 16-QAM, 32-QAM, 16-APSK, 32-APSK, BFSK and GMSK**. These are educational waveform implementations. APSK settings are illustrative, 32-QAM is rectangular, and this is not a complete DVB-S2 transmitter.

| Trained artifact | Registry ID | What to say in the viva |
|---|---|---|
| Supplied 1D-CNN | `1d-cnn` | Preserved original classifier and normalization statistics |
| Random Forest | `rf-experimental` | A separately trained synthetic experiment with its own feature preparation |
| ResNet | `resnet-synthetic-v1` | A real saved synthetic-experiment checkpoint, executed by the fixed-graph NumPy runtime |
| CLDNN | `cldnn-synthetic-v1` | A real saved synthetic-experiment checkpoint, including recurrent inference, executed by NumPy |

Their classification labels are **BPSK, QPSK and 16-QAM**. Generating GMSK or 32-APSK does not train a corresponding classifier. Use Signal Intelligence to inspect these additional waveforms. The registry can also show explicitly untrained original-domain architecture slots; these are not additional trained artifacts. Do not relabel the synthetic experiments as independently validated original-domain models.

## The eight workspaces

| Workspace and route | Demonstrable purpose |
|---|---|
| Overview `/` | Rotate, pause and zoom the globe; inspect simulated spacecraft and real backend/classification status |
| Live Classifier `/classifier` | Load/upload I/Q, select a complete 128-sample window, run the CNN, explain its logits and inspect persistent history |
| Satellite Studio `/studio` | Generate seeded signals, apply channel impairments, export CSV/audio and send captures to other workspaces |
| Model Evaluation `/evaluation` | Inspect stored confusion matrices and SNR evidence; distinguish it from registered external evaluations |
| Model Arena `/arena` | Inspect actual artifact availability, run measured model benchmarks and examine compatible experiment evidence |
| Signal Intelligence `/intelligence` | Inspect waveform, constellation, power, PSD, phase and time-frequency views; compare stored captures |
| Satellite Theory `/theory` | Explore constellations, noise formulas, educational orbital Doppler and amplifier curves |
| Settings `/settings` | Check the API, inspect model metadata, and persist theme/chart/default-generator preferences |

## A nine-minute demonstration

Install dependencies and build once before the presentation. Open the application and allow the first model/plot load to finish. Use the displayed results even when a generated example is misclassified; an unexpected prediction is useful evidence of the model's limits.

| Time | Action and explanation |
|---|---|
| 0:00–0:45 | **Overview:** rotate/pause the globe. Say that spacecraft and orbital links are simulated and independent of the RF samples. Point out the actual backend status. |
| 0:45–1:15 | **Theory:** switch the ideal constellation from QPSK to 16-QAM. Explain I/Q as the horizontal/vertical coordinates of a complex-signal arrow. |
| 1:15–2:30 | **Studio:** choose QPSK, 1,024 samples, 8 samples/symbol, an 8,000 Hz clock and a fixed seed. Generate once at a higher SNR and once at a lower SNR. Regenerate after changing controls; compare the returned plots. |
| 2:30–3:30 | **Classifier:** use Studio's **Send to classifier**, then **Run CNN classification**. Explain the three probabilities, entropy and uncalibrated confidence. Show the selected 128-sample window; move its start and rerun if time permits. |
| 3:30–4:15 | **Explanation:** choose a target class and press **Explain classification**. Show genuine saliency/Integrated Gradients, then export the result JSON. Attribution describes model sensitivity, not physical causation. |
| 4:15–5:15 | **Intelligence:** inspect the active capture. Show PSD and the spectrogram controls; change FFT/window/overlap and explicitly recompute. A frequency marker is an operator annotation. Optional waterfall rendering uses the same computed density. |
| 5:15–6:00 | **Evaluation:** show the confusion matrix, class metrics and provenance. Explain that these are retained/recomputed results, not a new independent test performed by opening the page. |
| 6:00–7:30 | **Arena:** identify the four runnable artifacts. Select a model, use five timed iterations and run the benchmark; repeat for another model if time allows. These timings describe this computer. Compare accuracy only when dataset and test protocol match. |
| 7:30–8:15 | **History:** return to Classifier, expand persistent history and export a saved JSON/PDF report. Explain that replay needs retained managed recordings or the original input samples. |
| 8:15–9:00 | **Settings:** change a preference and reload to show persistence. Finish by opening the offline teaching guide for questions. |

For a five-minute slot, prioritize Overview → Studio → Classifier → Intelligence → Arena. Cover the other workspaces during the viva. For a longer demonstration, import a generated CSV/NPY into **External recordings**, supply a session ID and sample clock, inspect a later window, and run a small batch. A supplied/generated test file is a reproducible demonstration fixture, not proof of real satellite acquisition.

## Scientific claims to keep honest

- A modulation label describes the waveform family. The project does not decode satellite payloads, recover arbitrary packet formats, identify spacecraft from RF, or decrypt messages. The WAV export is audible heterodyne illustration.
- The supplied corpus has incomplete training/split provenance. The saved demo subset and historical evaluation are different evidence sources; do not present the subset as the complete original test set or independent new validation.
- RF/ResNet/CLDNN experiment evidence is synthetic. A synthetic held-out result does not establish real satellite performance and cannot be ranked directly against a score from another dataset.
- CNN input is I-first `128 × 2`, transformed once with the saved mean/std. Uploads are not silently resampled, padded or normalized to unit power. Studio's explicit output-amplitude policy is a separate generation choice.
- Softmax confidence is uncalibrated; entropy measures uncertainty among the three trained classes. Neither provides a validated unknown-signal detector.
- Power is in relative sample units unless calibrated. An arbitrary recording does not uniquely reveal noise power or SNR. Unknown sample clocks require normalized axes. Time-frequency views sample a bounded number of windows, and the −300 dB floor is a display convention.
- SigMF import requires explicit choices for unsigned zero, real I-only handling and multichannel selection. Derived exports preserve managed float32 samples and source metadata, not original ADC bytes. Required unknown extensions and collection/tar ingestion remain outside the documented reader scope.

For established measurements and their conditions, cite the [Phase 2 report](../PHASE2_REPORT.md) and [Phase 3 report](../PHASE3_REPORT.md). These are dated milestone reports: the Phase 3 runtime-availability limitation predates the current fixed-checkpoint NumPy execution of ResNet/CLDNN. Current completion artifacts are under [integration/completion](../integration/completion/). Do not invent a new accuracy score from a successful UI demonstration.

## Architecture and code-reading order

Follow one complete request before reading every component: **React control → typed HTTP request → FastAPI validation → preprocessing/model or DSP → returned JSON → plot/report**. SQLite stores metadata; UUID-named files store managed I/Q and reports.

1. [Teaching guide](satmodclass-teaching-guide.html): I/Q, modulation and the end-to-end explanation.
2. [App shell](../frontend/src/components/satmod-app.tsx) and [workspace pages](../frontend/src/pages/): navigation, controls and shared capture handoffs.
3. [Typed frontend API](../frontend/src/lib/satmod-api.ts) and [FastAPI app](../backend/main.py): trace generation/classification requests.
4. [Preprocessing](../utils/preprocessing.py) and [classifier service](../backend/services/classifier.py): shape, class order, saved normalization and genuine predictions.
5. [Model registry](../backend/services/model_registry.py), [model backend](../utils/model_backend.py) and [fixed neural runtime](../backend/services/neural_numpy.py): real artifacts, preprocessing and model availability.
6. [Modulation mappings](../utils/modulation.py) and [signal processing](../utils/signal_processing.py): seeded bits, pulse shaping, amplifier/fading/phase/CFO/Doppler/noise sequence.
7. [Signal diagnostics](../backend/services/signals.py) and [time-frequency computation](../backend/services/time_frequency.py): plot formulas, units and bounds.
8. [Capture import](../backend/services/captures.py), [SigMF reader](../backend/services/sigmf_io.py) and [workspace store](../backend/services/workspace_store.py): source metadata, hashes, windows and persistence.
9. [Training runner](../training/runner.py), [Python tests](../tests/) and [browser scenarios](../frontend/e2e/): controlled offline fitting and the evidence behind application behavior. Training is separate from the demonstration; no web action promotes a replacement production model.

## Short viva answers

**Why two I/Q columns?** They preserve the two coordinates needed to describe baseband amplitude and phase.

**Why exactly 128 samples for classification?** That is the trained model input contract. Longer recordings provide selectable complete windows.

**Why generate nine schemes but classify three?** Generation and supervised recognition are different tasks; adding a generator does not supply training evidence for a new label.

**What makes a model comparison valid?** Matching labels, dataset/test windows and protocol, while retaining each model's documented preprocessing and artifact identity.

**Why can NumPy run a trained neural model?** Training produced fixed weights. The supported inference equations can execute those weights without retraining or installing TensorFlow; this runtime supports the published graphs, not arbitrary uploaded networks.

**What is the main research limitation?** Independently acquired, well-labeled satellite captures and suitable calibration/OOD validation are not supplied.

## Minimal submission and instructor checklist

- Include the source, four required model artifacts and their metadata, normalization/data evidence, this guide, the offline HTML teaching guide and the Phase reports. Include `frontend/dist` when promising a prebuilt release. Keep `.env`, access tokens, `.satmod`, private backups, `.venv` and `node_modules` out of the submission.
- Test the extracted copy in its own virtual environment. Confirm the printed local URL, `/api/health`, all eight routes and the four runnable model IDs in `/api/models`.
- Show a seeded generated capture, genuine inference and at least one exported scientific report. Keep the actual parameters and outputs; do not prepare fabricated probabilities or expected benchmark timings.
- Explain the input contract, three-label scope, dataset provenance and synthetic limitations without prompting.
- Record the submitted Git revision and retain the existing test/validation reports. Preserve original model/data files; use a separate demo workspace for presentation history when needed.
