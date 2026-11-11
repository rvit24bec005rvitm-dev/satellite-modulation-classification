import { operationsApi } from "@/lib/operations-api";
import { lazy, Suspense, useState, type ReactNode } from "react";
import { Link } from "@tanstack/react-router";
import { useIsMutating, useQuery } from "@tanstack/react-query";
import {
  Activity,
  ArrowRight,
  Box,
  Braces,
  Check,
  ChevronRight,
  CircleDot,
  Cpu,
  Crosshair,
  FlaskConical,
  Radio,
  Satellite,
  ScanLine,
  Signal,
  Waves,
} from "lucide-react";
import { satmodApi, SUPPORTED_MODULATIONS } from "@/lib/satmod-api";
import { useEvaluation, useHealth, useInfo, useSignal } from "@/hooks/use-satmod";
import { useMissionClock, useReducedMotion } from "@/hooks/use-mission-clock";
import {
  DEMO_SATELLITES,
  GROUND_STATION,
  elapsedLabel,
  elevationDeg,
  orbitalPeriod,
  subsatellitePoint,
} from "@/lib/orbital-simulation";
import { numeric, percent } from "@/components/workspace-ui";
const EarthGlobe = lazy(() => import("@/components/mission/earth-globe"));
const GroundTrackMap = lazy(() => import("@/components/mission/ground-track-map"));
function Panel({
  title,
  tag,
  children,
  className = "",
}: {
  title: string;
  tag?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`mission-panel ${className}`}>
      <div className="mission-panel-heading">
        <h2>{title}</h2>
        {tag && <span>{tag}</span>}
      </div>
      {children}
    </section>
  );
}
const TECHNICAL_NOTES: Record<string, string> = {
  Inclination: "Angle of the simulated orbital plane relative to Earth's equator.",
  "Ground visibility":
    "Simulated elevation exceeds the 10° station mask; this does not establish an RF link.",
  "Signal power":
    "Computed from the actual I/Q capture, in relative sample units; not calibrated dBm.",
  SNR: "A measured estimate only when supplied; otherwise source metadata or configured synthesis SNR.",
  "Input window":
    "128 time samples × two channels: I then Q. The backend applies the saved training transform.",
};
function DataRow({ label, value, kind = "" }: { label: string; value: ReactNode; kind?: string }) {
  return (
    <div className={`mission-data-row ${kind}`} title={TECHNICAL_NOTES[label]}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
export function OverviewPage() {
  const health = useHealth(),
    info = useInfo(),
    evaluation = useEvaluation();
  const workspace = useQuery({
    queryKey: ["workspace-status"],
    queryFn: () => operationsApi.status(),
    refetchInterval: 5000,
    retry: 1,
  });
  const online = health.isSuccess && health.data.connected;
  const { capture, classification } = useSignal();
  const history = useQuery({
    queryKey: ["history"],
    queryFn: ({ signal }) => satmodApi.history(signal),
    refetchInterval: 5000,
    retry: 1,
  });
  const preview = useQuery({
    queryKey: ["preview"],
    queryFn: ({ signal }) => satmodApi.sample(42, signal),
    retry: 1,
    enabled: !capture,
  });
  const input = capture ?? preview.data;
  const [selectedId, setSelectedId] = useState("SAT-001");
  const selected = DEMO_SATELLITES.find((s) => s.id === selectedId) ?? DEMO_SATELLITES[0]!;
  const reduced = useReducedMotion();
  const [motionChoice, setMotionChoice] = useState<boolean | null>(null);
  const autoRotate = motionChoice ?? !reduced;
  const { clock } = useMissionClock(autoRotate);
  const position = subsatellitePoint(selected, clock.seconds);
  const elevation = elevationDeg(selected, clock.seconds);
  const latest = history.data?.[0];
  const result =
    classification && (!latest || latest.id === classification.id) ? classification : null;
  const processing = useIsMutating({ mutationKey: ["classification"] }) > 0;
  const modelReady = online && health.data?.modelId === "1d-cnn";
  const normReady = online && !!info.data?.normalization.std;
  const actualSnr = input?.analysis.estimatedSnrDb ?? input?.snrDb;
  const snrKind =
    input?.analysis.estimatedSnrDb != null
      ? "MEASURED"
      : input?.snrDb != null
        ? "SOURCE METADATA"
        : "UNAVAILABLE";
  const nodes = [
    {
      label: "I/Q INPUT",
      icon: Waves,
      ready: !!input,
      note: input ? `${input.samples.length} samples` : "Awaiting capture",
    },
    {
      label: "PREPROCESS",
      icon: Braces,
      ready: normReady,
      note: normReady ? "Training z-score" : "Unavailable",
    },
    {
      label: "1D-CNN",
      icon: Cpu,
      ready: modelReady,
      note: modelReady ? "Model loaded" : "Model unavailable",
    },
    {
      label: "SOFTMAX",
      icon: Box,
      ready: online,
      note: online ? "3-class output" : "Service offline",
    },
    {
      label: "CLASSIFY",
      icon: ScanLine,
      ready: !!latest,
      note: processing ? "Processing…" : (latest?.predictedModulation ?? "Awaiting inference"),
    },
  ];
  return (
    <div className="mission-dashboard">
      <div className="mission-page-line">
        <div>
          <span className="panel-eyebrow">WORKSPACE / 01</span>
          <h1>Mission Control</h1>
        </div>
        <div className="mission-summary">
          <span>
            <i className={online ? "status-dot online" : "status-dot warning"} />
            {online ? "INFERENCE READY" : health.isPending ? "CONNECTING" : "BACKEND OFFLINE"}
          </span>
          <span>
            3 <small>SIMULATED SATELLITES</small>
          </span>
          <span>
            {percent(evaluation.data?.metrics.accuracy)} <small>STORED ACCURACY</small>
          </span>
        </div>
      </div>
      <div className="mission-upper">
        <div className="mission-left-column">
          <Panel
            title="Satellite overview"
            tag={<span className="demo-pill">DEMO</span>}
            className="satellite-info"
          >
            <label className="satellite-picker">
              <Satellite size={15} />
              <select
                aria-label="Selected satellite"
                value={selectedId}
                onChange={(e) => setSelectedId(e.target.value)}
              >
                {DEMO_SATELLITES.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.id} / {s.name}
                  </option>
                ))}
              </select>
            </label>

            <div className="panel-data">
              <DataRow label="Orbit" value={selected.orbitType} />
              <DataRow label="Altitude" value={`${selected.altitudeKm.toLocaleString()} km`} />
              <DataRow label="Inclination" value={`${selected.inclinationDeg.toFixed(1)}°`} />
              <DataRow
                label="Orbital period"
                value={`${(orbitalPeriod(selected) / 60).toFixed(1)} min`}
              />
              <DataRow
                label="Scenario status"
                value={
                  clock.enabled && performance.now() >= clock.pauseUntil ? "SIMULATING" : "PAUSED"
                }
              />
              <DataRow label="Signal link" value="DEMO / NO RF" />
              <DataRow
                label="Ground visibility"
                value={elevation > 10 ? "IN VIEW" : "BELOW MASK"}
                kind={elevation > 10 ? "good" : "muted"}
              />
            </div>
            <p className="mission-note">Simulated geometry · no live spacecraft or RF link.</p>
          </Panel>
          <Panel title="Classification system" tag={<Cpu size={12} />}>
            <div className="panel-data">
              <DataRow label="Architecture" value={info.data ? "1D-CNN" : "—"} />
              <DataRow label="Input window" value={info.data?.inputShape.join(" × ") ?? "—"} />
              <DataRow
                label="Backend"
                value={online ? "CONNECTED" : "OFFLINE"}
                kind={online ? "good" : "muted"}
              />
            </div>
            <div className="class-tags">
              {(info.data?.classes ?? SUPPORTED_MODULATIONS).map((c) => (
                <span key={c}>{c}</span>
              ))}
            </div>
          </Panel>
          <div className="mission-actions">
            <Link to="/classifier">
              <Crosshair size={13} />
              Launch Classifier
              <ArrowRight size={12} />
            </Link>
            <Link to="/studio">
              <FlaskConical size={13} />
              Satellite Studio
              <ArrowRight size={12} />
            </Link>
            <Link to="/evaluation">
              <Activity size={13} />
              Model Evaluation
              <ArrowRight size={12} />
            </Link>
          </div>
        </div>
        <section className="mission-globe-panel">
          <Suspense
            fallback={
              <div className="viewport-loading" role="status">
                <GlobeLoader />
                Initializing orbital viewport…
              </div>
            }
          >
            <EarthGlobe
              clock={clock}
              selected={selected}
              onSelect={setSelectedId}
              autoRotate={autoRotate}
              onAutoRotate={() => setMotionChoice(!autoRotate)}
              motionEnabled={autoRotate}
            />
          </Suspense>
        </section>
        <div className="mission-right-column">
          <Panel title="Signal telemetry" tag={<Radio size={13} />}>
            <div className="telemetry-source">
              <span className="signal-live-icon">
                <Signal size={15} />
              </span>
              <div>
                <strong>{input?.source ?? "No capture selected"}</strong>
                <small>{capture ? "ACTIVE INPUT" : "SAVED DATASET FRAME · NOT A DOWNLINK"}</small>
              </div>
            </div>
            <div className="panel-data">
              <DataRow
                label="Signal power"
                value={input ? `${numeric(input.analysis.powerDb)} dB rel.` : "—"}
              />
              <DataRow label="SNR" value={actualSnr == null ? "—" : `${numeric(actualSnr)} dB`} />
              <DataRow label="Carrier frequency" value="Not provided" />
              <DataRow label="RF link strength" value="Not measured" />
            </div>
            <p className="mission-note">
              SNR: {snrKind.toLowerCase()}. Satellite scenario and RF capture are independent.
            </p>
          </Panel>
          <Panel title="Modulation analysis" tag={<span className="panel-eyebrow">3 CLASSES</span>}>
            <div className="probability-rows">
              {SUPPORTED_MODULATIONS.map((name, i) => (
                <div key={name}>
                  <div>
                    <span>
                      <i style={{ background: ["#00D9FF", "#248BFF", "#00E0A4"][i] }} />
                      {name}
                    </span>
                    <strong>{result ? percent(result.probabilities[name]) : "—"}</strong>
                  </div>
                  <div className="probability-track">
                    {result && (
                      <i
                        style={{
                          width: `${result.probabilities[name] * 100}%`,
                          background: ["#00D9FF", "#248BFF", "#00E0A4"][i],
                        }}
                      />
                    )}
                  </div>
                </div>
              ))}
            </div>
            <p className="mission-note">
              {result
                ? `Actual CNN output · ${new Date(result.createdAt).toLocaleTimeString("en-GB", { timeZone: "UTC" })} UTC`
                : "Awaiting a complete inference result in this browser."}
            </p>
          </Panel>
          <Panel
            title="AI inference status"
            tag={<i className={online ? "status-dot online" : "status-dot warning"} />}
          >
            <div className="panel-data">
              <DataRow
                label="CNN model"
                value={modelReady ? "LOADED" : "UNAVAILABLE"}
                kind={modelReady ? "good" : "muted"}
              />
              <DataRow
                label="Engine"
                value={
                  online
                    ? `${health.data?.backend?.replace(" Native Engine", "") ?? "Engine"} · ${processing ? "RUNNING" : "READY"}`
                    : "Offline"
                }
              />
              <DataRow
                label="Latest result"
                value={latest?.predictedModulation ?? "Awaiting inference"}
              />
              <DataRow
                label="Last latency"
                value={latest ? `${numeric(latest.processingMs)} ms` : "—"}
              />
              <DataRow
                label="Last inference"
                value={
                  latest
                    ? `${new Date(latest.createdAt).toLocaleTimeString("en-GB", { timeZone: "UTC" })} UTC`
                    : "—"
                }
              />
            </div>
          </Panel>
        </div>
      </div>
      <div className="mission-lower">
        <Panel
          title="Ground track"
          tag={
            <>
              <span className="map-key">
                <i /> {selected.id}
              </span>
              <span className="demo-pill">SIMULATED</span>
            </>
          }
          className="ground-track-panel"
        >
          <div className="map-container">
            <div className="map-canvas-frame">
              <Suspense fallback={<div className="map-empty">Loading geographical map…</div>}>
                <GroundTrackMap satellite={selected} seconds={clock.seconds} />
              </Suspense>
            </div>
            <aside className="map-position" aria-label="Simulated ground-track coordinates">
              <span className="panel-eyebrow">TRACK POSITION</span>
              <DataRow label="Latitude" value={`${position[0].toFixed(2)}°`} />
              <DataRow label="Longitude" value={`${position[1].toFixed(2)}°`} />
              <DataRow label="Elevation" value={`${elevation.toFixed(1)}°`} />
              <div className="map-station-ref">
                {GROUND_STATION.id}
                <small>Simulated reference station</small>
              </div>
            </aside>
          </div>
          <div className="map-footer">
            <span>
              <i /> {GROUND_STATION.id} · Simulated ground station
            </span>
            <span>Natural Earth · 1 orbit / ECEF</span>
          </div>
        </Panel>
        <Panel
          title="Signal processing pipeline"
          tag={
            <span className={processing ? "pipeline-status busy" : "pipeline-status"}>
              {processing ? "PROCESSING" : online ? "SERVICE READY" : "OFFLINE"}
            </span>
          }
          className={`pipeline-panel ${processing ? "is-processing" : ""}`}
        >
          <div className="pipeline-nodes">
            {nodes.map(({ label, icon: Icon, ready, note }, i) => (
              <div className="pipeline-step" key={label}>
                <div className={`pipeline-symbol ${ready ? "ready" : ""}`}>
                  <Icon size={18} />
                  {ready && <Check size={8} className="pipeline-check" />}
                </div>
                <strong>{label}</strong>
                <small>{note}</small>
                {i < 4 && <ChevronRight className="pipeline-arrow" size={15} />}
              </div>
            ))}
          </div>
          <div className="pipeline-footer">
            <span>
              <i className={online ? "status-dot online" : "status-dot warning"} />
              {online
                ? "Inference API available"
                : (health.error?.message ?? "Connecting to inference service…")}
            </span>
            <span>
              {workspace.data
                ? `${workspace.data.captures} captures · ${workspace.data.runningJobs} jobs`
                : "Workspace status unavailable"}{" "}
              · T+ {elapsedLabel(clock.seconds)} <small>SIM / 30×</small>
            </span>
          </div>
        </Panel>
      </div>
      <Panel
        title="Recent signal activity"
        tag={
          <>
            <span className="activity-count">{history.data?.length ?? 0} RECORDS</span>
            <Link to="/classifier">
              OPEN WORKSTATION <ChevronRight size={10} />
            </Link>
          </>
        }
        className="activity-panel"
      >
        <div className="activity-table-wrap">
          <table className="activity-table">
            <thead>
              <tr>
                <th>TIMESTAMP / UTC</th>
                <th>SIGNAL SOURCE</th>
                <th>MODULATION</th>
                <th>CONFIDENCE</th>
                <th>STATUS</th>
              </tr>
            </thead>
            <tbody>
              {history.data?.length ? (
                history.data.slice(0, 3).map((h) => (
                  <tr key={h.id}>
                    <td>
                      {new Date(h.createdAt).toLocaleTimeString("en-GB", { timeZone: "UTC" })}
                    </td>
                    <td title={h.source}>{h.source}</td>
                    <td>
                      <span className="modulation-cell">{h.predictedModulation}</span>
                    </td>
                    <td>
                      <div className="confidence-cell">
                        <span>{percent(h.confidence)}</span>
                        <i>
                          <b style={{ width: `${h.confidence * 100}%` }} />
                        </i>
                      </div>
                    </td>
                    <td>
                      <span className="completed-status">
                        <Check size={10} />
                        COMPLETED
                      </span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="activity-empty">
                    {history.error
                      ? "Activity unavailable while the backend is disconnected."
                      : "No classifications yet. Launch the classifier to analyze your first capture."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
function GlobeLoader() {
  return <CircleDot size={28} />;
}
