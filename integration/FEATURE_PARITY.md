# Audit and implementation checklist

Both existing local repositories were inspected before changes. The original frontend AGENTS.md prohibits rewriting public history, hand editing generated files, and changing Lovable-managed external systems. No such external changes were made. The shared file-route shell instruction is preserved. The roadmap, source UI, router, tests, API skeleton, styles, asset, Python pages/helpers/tests, model configs/weights, normalization statistics, sample arrays, and evaluation files were audited.

| Original capability | React equivalent | API/data path | Status |
| --- | --- | --- | --- |
| Landing model summary / signal preview | Overview | health, info, sample, evaluation | Implemented |
| Live Classifier saved sample / class / SNR selection | Classifier demo filters | samples and samples/{id} | Implemented |
| Uploaded NPY/NPZ/CSV/MAT/WAV | Classifier upload + errors | classify/upload | Implemented; secure parsing |
| Signal decision / confidence / all probabilities | Classifier result | classify | Implemented |
| Entropy uncertainty gauge | Classifier entropy in bits | shannon_entropy | Implemented, three-class limitation stated |
| I/Q / constellation / spectrum / spectrogram | Shared inspector | signal analysis | Implemented with real data |
| Eye traces | Studio inspector | generation with known sps | Implemented; unavailable for unknown timing |
| Generated modulation / AWGN / RRC | Studio controls | signal/generate | Implemented |
| Doppler / CFO / phase jitter / Rician / TWTA | Studio channel controls | signal/generate | Implemented |
| Signal export / classification handoff | CSV download / shared context | full-resolution samples | Implemented |
| Audible heterodyne | Studio audio download | signal/audio | Implemented; no payload-decoding claim |
| Evaluation accuracy / report / confusion matrix | Evaluation | evaluation | Implemented from actual counts |
| SNR curve / per-class metrics / exports | Evaluation | evaluation/snr + stored report | Implemented |
| SNR-sliced confusion matrix | No invented slice | Full per-frame 900-item data unavailable | Explicit aggregate-only result |
| Model comparison / latency profiling | Arena | models + benchmark | CNN measured; other models not trained |
| Expert features / distribution diagnostics | Intelligence | signal/analyze | Implemented |
| Two-signal comparison | Intelligence capture A/B | signal/compare | Implemented |
| Theory / constellation education | Theory explorer and equations | analytic references | Implemented |
| Orbital Doppler / TWTA education | Theory interactive curves | orbit + analytic Saleh model | Implemented; educational approximation |
| Service / theme / defaults | Settings | info / health / localStorage | Implemented |

## Material audit findings resolved

- The only registered frontend page was a blank placeholder. Eight routes now render connected workspaces.
- Hardcoded offline indicators, disconnected evaluation screens, simulated local input, and unavailable scientific chart placeholders were replaced with real API operations.
- The original frontend stylesheet referred to the orbit image through the wrong relative path; corrected in the integrated copy.
- The original Arena derives alternatives from the CNN probabilities. The integrated Arena exposes no such synthetic comparison results.
- The saved Keras architecture uses same padding and Conv ReLU before BN; the NumPy implementation did neither. Fixed and independently checked against TensorFlow.
- Original upload handling added unit-power normalization to already scaled data. The new backend has one amplitude-preserving training transform for every source.
- GMSK/BFSK aliases, 32-APSK reference cardinality, APSK reference scaling, and PSD density scaling were corrected.
- The final simulator exposes TWTA input drive, since varying only output gain would disappear during final RMS scaling.
- Constant-channel moments and zero-energy PAPR display as undefined, not fabricated zero-valued measurements.

The original Lovable folder and all ML/data artifacts are preserved. Legacy Streamlit files retain the user's pre-existing changes. The integrated React app is the primary interface; the legacy Arena remains a historical illustration as described in README.
