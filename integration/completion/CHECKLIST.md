# Final college-project checklist

This supplements the historical `PHASE2_REPORT.md`, `PHASE3_REPORT.md` and `PHASE3_CHECKLIST.md`. In particular, the older optional-neural-runtime blocker is resolved by the verified fixed NumPy engines.

| Requirement | Status | Evidence |
| --- | --- | --- |
| Preserve original CNN, normalization, data, scores and experimental checkpoints | COMPLETE AND VERIFIED | [Ten SHA-256 checks](preserved-artifacts.json); original research files unchanged |
| Connected eight-workspace application and rotating Earth | EXISTING AND VERIFIED | Production browser suite, serving check |
| Actual CNN/RF/ResNet/CLDNN inference in normal lightweight runtime | COMPLETE AND VERIFIED | [Native API](neural-numpy-api.json), [independent parity](neural-native-parity.json), [512 MiB container](neural-numpy-container.json) |
| Preserve model identity, preprocessing, experimental labels and untrained placeholders | COMPLETE AND VERIFIED | Research model tests, missing-artifact rejection, unchanged hashes |
| Recording import context, sidecar selection and mobile batch actions | COMPLETE AND VERIFIED | Three component regressions and three actual API/browser workflows |
| Shared workspace access, token validation, locking, revocation and pending-verification race | COMPLETE AND VERIFIED | API/component tests and four actual protected browser workflows |
| Deployment readiness, singleton guard and runtime privilege drop | COMPLETE AND VERIFIED | Readiness tests and actual container startup checks |
| Offline workspace backup/restore with integrity checks | COMPLETE AND VERIFIED | Backup tests; existing 294-analysis restore proof |
| Preserve existing private analysis history | COMPLETE AND VERIFIED | [Logical metadata and all report hashes match](private-workspace-backup-proof.json); archive outside Git |
| Isolate automated tests from operator data and historical screenshots | COMPLETE AND VERIFIED | Session workspace fixture, isolated contracts/browser server, current screenshot defaults |
| Offline HTML project explanation | EXISTING AND VERIFIED | [Standalone teaching guide](../../docs/satmodclass-teaching-guide.html), unchanged |
| College demonstration and viva instructions | COMPLETE AND VERIFIED | [College demo guide](../../docs/college-demo.md), all local links checked |
| Full source on main and clean submission bundle | COMPLETE AND VERIFIED | Main-branch Git history; [extracted package smoke](college-package-verification.json), SHA-256 manifest in the archive |
| Independently acquired satellite-data generalization, validated OOD/calibration | BLOCKED ON EXTERNAL DATA | Not replaced by synthetic scores; no fabricated evidence |
| Exhaustive physical-device, screen-reader and PDF-UA certification | PARTIALLY IMPLEMENTED | Automated browser/accessibility checks only; no certification claimed |
| Unrestricted public training or automatic production promotion | NOT IMPLEMENTED, INTENTIONALLY | Offline trusted training remains separate |
| Permanent cloud hosting | OUTSIDE FINAL COLLEGE SCOPE | User explicitly requested local college-project completion; no paid service created |

Executed counts, resource scope, dependency audit details and remaining scientific limits are recorded in [REPORT.md](REPORT.md).
