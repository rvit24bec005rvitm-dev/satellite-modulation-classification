# College-project completion record

The final target is a self-contained **college project**, as requested after the optional hosting discussion. No Render account, permanent deployment or paid service is required. Source changes are published directly to **main** without rewriting history. This record supplements the historical Phase 2/3 reports; those reports and their original evidence remain unchanged.

## Finished behavior

- All eight React workspaces retain the shared application shell, real FastAPI backend, scientific plots and rotating simulated Earth.
- Four genuinely trained models execute in the lightweight default runtime: the supplied production CNN and separately named synthetic RF, ResNet and CLDNN experiments. The fixed neural engines read the original hash-verified tensors; they do not retrain, transform CNN probabilities, substitute another model or promote an experiment.
- Protected mode has public discovery and readiness, real token verification, tab-scoped credentials, explicit locking and cleared private state. Failed downloads/audio authorization also lock the workspace. Delayed token checks cannot undo a lock; failed background discovery preserves known protection. Trusted local demonstration remains token-free by default.
- Completed recording imports clear old diagnostics/activity; non-SigMF imports no longer submit a hidden stale sidecar; all six loaded-batch export actions fit a 390-pixel viewport.
- Offline backup/restore consolidates SQLite plus referenced files, validates identities and SHA-256 hashes, bounds archive extraction, rejects active jobs/overwrites, and publishes a validated private restore atomically. No public backup or unrestricted training endpoint was added.
- Startup and readiness validate singleton settings, token configuration, actual writable storage and a real CNN forward pass. The portable Docker image drops privileges before the API starts; optional Render configuration attaches a durable disk.
- Python tests, API-client contracts and newly started browser-test servers use disposable workspaces. Browser output defaults no longer replace historical Phase 3 screenshots. The skip link stays available while access discovery is loading.
- The standalone HTML teaching guide remains intact. The college demonstration guide provides setup, architecture, demo order, viva preparation and limitations.

## Scientific verification

| Evidence | Executed result |
| --- | --- |
| Independent TensorFlow 2.16 oracle, unchanged neural checkpoints | Both stored 360-frame test probability arrays reproduced exactly |
| Default NumPy engines versus independent TensorFlow golden vectors | All 360 class decisions match for each model; ResNet max probability difference 1.79×10⁻⁷, CLDNN 4.77×10⁻⁷ on the local host |
| NumPy engines in Linux, 512 MiB / 0.5 CPU container | Both complete golden-vector replays pass; no class mismatches; ResNet max difference 1.79×10⁻⁷, CLDNN 3.58×10⁻⁷ |
| Current Linux TensorFlow 2.21 independent reproduction | All 300 supplied CNN frames and both 360-frame neural test sets pass probability tolerance, with no class mismatches |
| Actual protected API | Real fresh synthetic evaluation jobs for all four models and an identical-test-protocol comparison pass |
| Artifact preservation | All ten production/experimental/data/score files retain their original SHA-256 hashes |

See [the independent reference metadata](neural-tf-golden.json), [native parity](neural-native-parity.json), [native API evidence](neural-numpy-api.json), [Linux container proof](neural-numpy-container.json), [current TensorFlow reproduction](neural-linux-verification.json), and [preserved artifacts](preserved-artifacts.json). The exporter obtains its oracle directly from TensorFlow, independently of the native registry loader. Float32 tolerances were declared before comparison; probability parity is not a new accuracy estimate.

The production classifier still recognizes **BPSK, QPSK and 16-QAM**. Nine generator schemes do not imply nine trained classes. Synthetic test scores retain their original domain, splits and limitations. Original training-corpus provenance, independently acquired satellite validation, validated production OOD decisions and probability calibration remain unavailable. Neural forward engines do not expose fabricated gradients; genuine saliency and Integrated Gradients remain available for the supplied CNN.

## Storage preservation and recovery

The existing private workspace was stopped, backed up outside the repository, restored into a new private directory and compared. All **294 analyses**, all logical metadata and all 294 JSON report hashes match. Archives are mode 0600; the restore is mode 0700. The preview was restarted using the original workspace. Private recordings, metadata databases, tokens and backup archives are excluded from publication and the submission bundle.

The full audit caught pre-existing test defaults that could write fixture analyses into the operator workspace. The 18 introduced fixtures were moved to private recovery storage, every original record/report was checked against the verified backup, and all test transports now use temporary storage. Historical screenshot defaults were likewise corrected and the original PNG bytes restored. Final preservation checks confirm no changes to original research evidence.

See [private backup proof](private-workspace-backup-proof.json) and [the recovery instructions](../../docs/deployment.md). The archive exists locally; no transfer to a cloud host is claimed.

## Validation

| Check | Result |
| --- | --- |
| Complete Python suite | **440 passed**, including stale-readiness-probe recovery |
| Frontend TypeScript | Passed |
| Frontend ESLint | 0 errors; 18 existing warnings |
| Frontend unit/component tests | 72 passed |
| Real TypeScript-to-Python contracts | 11 passed |
| Production frontend build | Passed; fonts, textures and Plotly are bundled locally |
| Production browser suite | **51 passed** in the final full run; the original keyboard regression was fixed without removing its assertions |
| Python web dependency audit | 32 resolved packages, zero advisories/skips |
| Frontend dependency audit | Zero advisories |
| Optional current TensorFlow image audit | 60 installed packages, zero advisories/skips after pip-only remediation |
| Thin final image dependency audit | **33 installed packages, zero advisories/skips**; final source image has the byte-identical audited package freeze |
| Final portable Docker image | UID 10001, four genuine backends, shipped backup CLI, access rejection, identical persisted JSON/PDF across recreation |
| Extracted college submission | All source hashes verified; all eight routes, readiness, four model availability and actual CNN classification pass through `scripts/serve.py` with bundled frontend |

The [complete Python run](python-final-verified.log), [51-scenario browser run](browser-final-verified.log), [72 frontend tests](frontend-unit-final.log), [11 real client contracts](client-contract-final.log), [compiled serving check](production-serving.json) and [final installed-image audit](thin-final-audit-provenance.json) accompany this record. The first browser failure and its correction are retained separately. Container resource evidence is scoped to one worker, small real evaluation jobs and full fixed test-vector replays. Peak server RSS was **212,021,248 bytes (about 202 MiB)** under an actual 512 MiB / 0.5 CPU limit with no OOM. This is a bounded measured workload, not a general concurrent-load guarantee.

The optional Intel macOS TensorFlow reproduction profile depends on old protobuf with a documented advisory. It is labeled private compatibility-only and is not the default application or hosting profile. The default web dependency audit and audited current Linux TensorFlow profile have no reported advisories at the recorded audit time. Installed image installer vulnerabilities were remediated by updating pip to 26.3; the actual installed-package audit is retained.

The [final container smoke check](final-portable-container.json) and [extracted submission verification](college-package-verification.json) exercise the actual packaged application. The college ZIP includes source, original models, compiled frontend and documentation, with a per-file SHA-256 manifest. Virtual environments, node_modules, Git credentials, tokens, operator workspace and private backups are excluded. Git history identifies the published revision; the final archive manifest records that commit. Python dependencies must be installed on the presentation computer; Node is needed only to rebuild the frontend.

## Run and present

Follow [the college demonstration guide](../../docs/college-demo.md). For source development, use the root README's virtual environment and Node setup, then `python scripts/dev.py`. The prepared submission bundle includes compiled `frontend/dist`, so after installing the pinned Python web requirements, `python scripts/serve.py` serves frontend and backend together on localhost. No hosting account or TensorFlow is needed for the normal demo. Open [the standalone HTML explanation](../../docs/satmodclass-teaching-guide.html) directly in a browser for offline teaching and presentation.

Permanent hosting was deliberately left aside at the user's request. Optional configuration is retained for future use; it is not part of the college acceptance criteria. Exhaustive physical-device/screen-reader certification and new independently collected RF datasets are outside the executed evidence. No new performance score, original-domain model equivalence or production promotion is claimed.
