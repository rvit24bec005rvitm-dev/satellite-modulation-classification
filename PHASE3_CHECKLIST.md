# Phase 3 requirement checklist

Starting source: `0bf7c29` on `main`. Before changes, the executed baseline was
209 Python / 43 frontend / 11 real API contracts / 29 production browser tests.
All original research evidence and ten binary/model artifacts are preserved.
The pre-existing user galaxy CSS edit is included unchanged.

| Requirement | Classification | Evidence and practical boundary |
|---|---|---|
| Baseline repository/source inspection and original tests | EXISTING AND VERIFIED | `integration/phase3/baseline/`; original reports read, no research artifact replacement |
| Overview controls, mission design and rotating Earth | EXISTING AND VERIFIED | Real WebGL/rotation/drag/zoom/pause/selection/fallback/reduced-motion browser scenarios and responsive screenshots |
| Live Classifier, Studio, Evaluation, Arena, Intelligence, Theory and Settings | COMPLETE AND VERIFIED | Existing workflows retained; added clock/stale-response fixes, Theory formulas, preference reload and actual token-aware WAV checks |
| Every route and rendered control inventory | COMPLETE AND VERIFIED | Desktop/mobile eight-route inventory plus source/handler review and action regressions; permutation limits below |
| Core SigMF datatype/endian matrix, real/unsigned and channel selection | COMPLETE AND VERIFIED | 28 datatypes; explicit zero/i-only/channel choices; exact canonical round trips and malformed-input tests |
| SigMF metadata, capture segments, offsets and optional provenance | COMPLETE AND VERIFIED | File/source/hardware indices separated; NCD headers/trailers, clipping, hashes, segment boundaries and opaque optional metadata tested |
| SigMF derived-window paired export | COMPLETE AND VERIFIED | Consistent canonical cf32_le ZIP plus original source metadata; conversion is explicitly not original ADC byte-lossless |
| Arbitrary required extensions, collections/tar ingestion and universal schema certification | NOT IMPLEMENTED | Rejected/unsupported explicitly; no ambiguous decoder fallback |
| Production/RF external evaluation with model/version/hash/preprocessing | COMPLETE AND VERIFIED | Actual inference, persisted predictions/metrics, class-map checks, known-only SNR and honest raw model scores |
| Optional ResNet/CLDNN execution on this host | BLOCKED | Compatible TensorFlow runtime unavailable; catalog disables these models with reason; no substitute output |
| Exact-dataset/split Arena comparison | COMPLETE AND VERIFIED | Protocol identity checks and actual CNN/RF browser comparison; synthetic historical scores kept separate |
| Registry training manifest/NPZ/config bridge | COMPLETE AND VERIFIED | Deterministic exports, class/session/source/split/hash checks, real bounded offline fit/replay; no web fitting or promotion |
| Held-out-data and duplicate protection | COMPLETE AND VERIFIED | Active registry and retained completed-evaluation provenance prevent test reassignment, including deletion/re-registration regressions; declared acquisition truth cannot be proven |
| Missing SNR honesty in offline scripts | COMPLETE AND VERIFIED | NaN + snrKnown mask propagated; old unmasked input requires finite SNR; metrics use measured/known bins only |
| Persistent saliency/IG panel and full explanation export | COMPLETE AND VERIFIED | Actual signed128×2 maps, selected-target summaries, IG-only residual, baseline/steps, regions and exact identity checks |
| History technical PDFs | COMPLETE AND VERIFIED | Vector plots, model/input provenance and limitations; six saved JSON/PDF exports byte-identical across protected process restart |
| Managed recording/window comparison | COMPLETE AND VERIFIED | Actual I/Q/constellation/PSD/phase/amplitude/power/predictions; exact saved prediction reuse; equal/unequal/unknown rates and compatible attribution differences |
| Reproducible comparison JSON/PDF | COMPLETE AND VERIFIED | Persist/reload/restart and hashes verified; replay still requires retained raw recordings |
| STFT, waterfall, marker, peaks, activity and selection overlays | COMPLETE AND VERIFIED | Real bounded complex PSD; SciPy difference ≤5.69e−14 dB and power error ≤1.72e−8 on analytic fixtures; no carrier-identification claim |
| Plot zoom/pan, polling stability and asynchronous updates | COMPLETE AND VERIFIED | Before/after actual browser probe, genuine job-poll regression, deferred-render unit reproduction/fix and resource cleanup |
| Actual API/model/DSP/import/batch/memory/browser profiling | COMPLETE AND VERIFIED | Genuine baseline/final measurements and portable controlled512-window reader ablation; full prediction difference0 |
| Development dependency audit | COMPLETE AND VERIFIED | Initial14 findings resolved; final full npm audit0; compatible upgrades and full regression, CI audits all dependencies |
| Upload/storage/access/temp/logging safeguards | COMPLETE AND VERIFIED | Bounded metadata/decode, depth/index/time overflow rejection, two export slots, token-aware downloads, protected HTTP401/restart smoke and Windows handle cleanup tests |
| Mobile keyboard/navigation and chart accessibility improvements | COMPLETE AND VERIFIED | Skip link, current route, contained focus/Escape/opener restore, plot toolbar/group and numeric preview; original appearance retained |
| Exhaustive device/control-permutation/assistive-technology certification | PARTIALLY IMPLEMENTED | Automated route/action/responsive/accessibility-tree checks exist; physical mobile/GPU, Firefox/WebKit, full screen-reader and WCAG/PDF-UA certification not performed |
| Production CNN and original evidence preservation | EXISTING AND VERIFIED | `preserved-artifacts.json`: all ten hashes match Phase2 and starting Git blobs; original audit/Phase2 evidence unchanged |
| Independent acquisition-domain accuracy, calibrated probabilities/OOD | BLOCKED | Suitable independently sourced/split RF validation and calibration data absent; synthetic fixtures never presented as that evidence |
| Production promotion or unrestricted server training | NOT IMPLEMENTED | Intentionally excluded; production weights unchanged, fitting remains offline |
| Newly provisioned permanent hosting | NOT IMPLEMENTED | No permanent host provisioned; existing local/server/container launch paths retained |
| Final Python/frontend/contracts/types/lint/build/browser suite | COMPLETE AND VERIFIED | Final counts/logs in `PHASE3_REPORT.md` and `integration/phase3/validation-summary.json` |
| Code and evidence published directly to main | COMPLETE AND VERIFIED | No Phase3 feature branch, force push or history rewrite |

The exact supported scope, API/source inventory, commands, measured values and
remaining limitations are detailed in [PHASE3_REPORT.md](PHASE3_REPORT.md).
