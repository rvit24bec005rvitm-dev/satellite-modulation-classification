import { useState } from "react";
import { Link } from "@tanstack/react-router";
import { toast } from "sonner";
import {
  GENERATOR_MODULATIONS,
  downloadCapture,
  downloadBlob,
  apiBase,
  type SignalGenerationRequest,
  type GeneratedSignal,
} from "@/lib/satmod-api";
import { useSignal, useGeneration } from "@/hooks/use-satmod";
import { Button } from "@/components/ui/button";
import { CapturePlots } from "@/components/scientific-chart";
import {
  Field,
  NumberField,
  Message,
  SectionHeading,
  Surface,
  Stats,
  numeric,
} from "@/components/workspace-ui";
export function StudioPage() {
  const { preferences, setCapture } = useSignal();
  const initial: SignalGenerationRequest = {
    modulation: preferences.defaultModulation,
    snrDb: preferences.defaultSnr,
    numSamples: 1024,
    sps: preferences.defaultSps,
    sampleRate: 1,
    gaussianBt: 0.3,
    bfskDeviationHz: null,
    continuousPhase: true,
    apskRadii: null,
    apskOffsets: null,
    phaseNoiseMode: "iid",
    fadingBlockSamples: 1,
    normalizeOutput: true,
    rrcAlpha: 0.35,
    rrcEnabled: true,
    dopplerHz: 0,
    cfoHz: 0,
    phaseNoiseStd: 0,
    ricianEnabled: false,
    ricianK: 5,
    twtaDrive: 0,
    amplitudeRms: 0.008584335,
    seed: 42,
  };
  const [params, setParams] = useState(initial);
  const [signal, setSignal] = useState<GeneratedSignal | null>(null);
  const [audioError, setAudioError] = useState<Error | null>(null);
  const generation = useGeneration();
  const update = <K extends keyof SignalGenerationRequest>(k: K, v: SignalGenerationRequest[K]) =>
    setParams((p) => ({ ...p, [k]: v }));
  const generate = async () => {
    try {
      const value = await generation.mutateAsync(params);
      setSignal(value);
      setCapture(value);
      toast.success("Signal generated");
    } catch {
      /* Mutation error is shown inline. */
    }
  };
  const dirty =
    signal !== null &&
    Object.entries(params).some(
      ([key, value]) =>
        JSON.stringify(signal.parameters[key as keyof SignalGenerationRequest]) !==
        JSON.stringify(value),
    );
  const send = () => {
    if (signal) {
      setCapture(signal);
      toast.success("Full-resolution capture ready in classifier");
    }
  };
  const audio = async () => {
    if (!signal) return;
    setAudioError(null);
    try {
      const r = await fetch(`${apiBase()}/api/signal/audio`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          samples: signal.samples,
          sampleRate: signal.sampleRate,
          source: signal.source,
        }),
      });
      if (!r.ok) throw new Error("Audio generation failed.");
      downloadBlob(await r.blob(), "satmod-audible-heterodyne.wav");
    } catch (e) {
      setAudioError(e as Error);
    }
  };
  return (
    <div className="space-y-4">
      <Message error={generation.error ?? audioError} />
      <div className="grid items-start gap-4 xl:grid-cols-[300px_minmax(0,1fr)]">
        <Surface className="p-4">
          <SectionHeading title="Simulation controls" note="REPRODUCIBLE DSP" />
          <Field label="Modulation scheme">
            <select
              className="control"
              value={params.modulation}
              onChange={(e) =>
                setParams((p) => ({
                  ...p,
                  modulation: e.target.value as SignalGenerationRequest["modulation"],
                  apskRadii: null,
                  apskOffsets: null,
                  bfskDeviationHz: null,
                }))
              }
            >
              {GENERATOR_MODULATIONS.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </Field>
          {params.modulation === "GMSK" && (
            <NumberField
              label="Gaussian BT product"
              value={params.gaussianBt ?? 0.3}
              onChange={(n) => update("gaussianBt", n)}
              min={0.05}
              max={1}
              step={0.05}
            />
          )}
          {params.modulation === "BFSK" && (
            <>
              <NumberField
                label="BFSK deviation (Hz; default Rs/4)"
                value={params.bfskDeviationHz ?? params.sampleRate / (4 * params.sps)}
                onChange={(n) => update("bfskDeviationHz", n)}
                min={0.000001}
                max={params.sampleRate / 2}
                step={0.001}
              />
              <Field label="Continuous-phase BFSK">
                <input
                  type="checkbox"
                  checked={params.continuousPhase ?? true}
                  onChange={(e) => update("continuousPhase", e.target.checked)}
                />
              </Field>
            </>
          )}
          {["16-APSK", "32-APSK"].includes(params.modulation) && (
            <>
              <NumberField
                label="APSK middle/inner radius ratio"
                value={params.apskRadii?.[1] ?? 2.53}
                min={1.01}
                max={10}
                step={0.1}
                onChange={(n) =>
                  update(
                    "apskRadii",
                    params.modulation === "32-APSK" ? [1, n, params.apskRadii?.[2] ?? 4.3] : [1, n],
                  )
                }
              />
              {params.modulation === "32-APSK" && (
                <NumberField
                  label="APSK outer/inner radius ratio"
                  value={params.apskRadii?.[2] ?? 4.3}
                  min={1.02}
                  max={20}
                  step={0.1}
                  onChange={(n) => update("apskRadii", [1, params.apskRadii?.[1] ?? 2.53, n])}
                />
              )}
              <p className="mb-4 text-xs text-muted-foreground">
                Educational ring mapping; no DVB-S2 compliance claim.
              </p>
            </>
          )}
          <NumberField
            label="AWGN SNR (dB)"
            value={params.snrDb}
            onChange={(n) => update("snrDb", n)}
            min={-30}
            max={60}
          />
          <NumberField
            label="Capture length (samples)"
            value={params.numSamples}
            onChange={(n) => update("numSamples", n)}
            min={128}
            max={65536}
            step={128}
          />
          <NumberField
            label="Samples per symbol"
            value={params.sps}
            onChange={(n) => update("sps", n)}
            min={1}
            max={32}
          />
          <NumberField
            label="Sample rate (Hz; 1 = normalized)"
            value={params.sampleRate}
            onChange={(n) => update("sampleRate", n)}
            min={1}
            max={1e9}
          />
          <NumberField
            label="RMS amplitude (explicit output scale)"
            value={params.amplitudeRms}
            onChange={(n) => update("amplitudeRms", n)}
            min={0.000001}
            max={100}
            step={0.001}
          />
          <p className="mb-4 text-[11px] leading-5 text-muted-foreground">
            The default RMS matches the training statistics scale. This is a simulator calibration,
            not a guarantee of classifier accuracy. Upload amplitudes are preserved.
          </p>
          <Field label="Output amplitude policy">
            <select
              className="control"
              value={params.normalizeOutput === false ? "preserve" : "unit"}
              onChange={(e) => update("normalizeOutput", e.target.value === "unit")}
            >
              <option value="unit">Calibrate final capture to selected RMS</option>
              <option value="preserve">Preserve channel/amplifier gain</option>
            </select>
          </Field>
          <details className="mb-5 border-y border-border py-3" open>
            <summary className="cursor-pointer text-sm text-primary">Channel impairments</summary>
            <div className="mt-4">
              <Field label="RRC pulse shaping (linear modulations)">
                <input
                  type="checkbox"
                  disabled={["GMSK", "BFSK"].includes(params.modulation)}
                  checked={params.rrcEnabled}
                  onChange={(e) => update("rrcEnabled", e.target.checked)}
                />
              </Field>
              {["GMSK", "BFSK"].includes(params.modulation) ? (
                <p className="mb-4 text-xs text-muted-foreground">
                  This continuous-phase modulation uses its own frequency pulse; RRC controls do not
                  apply.
                </p>
              ) : (
                <NumberField
                  label="RRC roll-off α"
                  value={params.rrcAlpha}
                  onChange={(n) => update("rrcAlpha", n)}
                  min={0.01}
                  max={1}
                  step={0.05}
                />
              )}
              <NumberField
                label="Doppler offset (Hz)"
                value={params.dopplerHz}
                onChange={(n) => update("dopplerHz", n)}
                step={0.001}
              />
              <NumberField
                label="Carrier frequency offset (Hz)"
                value={params.cfoHz}
                onChange={(n) => update("cfoHz", n)}
                step={0.001}
              />
              <Field label="Phase-noise model">
                <select
                  className="control"
                  value={params.phaseNoiseMode ?? "iid"}
                  onChange={(e) => update("phaseNoiseMode", e.target.value as "iid" | "wiener")}
                >
                  <option value="iid">Independent phase jitter (σ radians)</option>
                  <option value="wiener">Wiener phase walk (σ radians/sample increment)</option>
                </select>
              </Field>
              <NumberField
                label={
                  params.phaseNoiseMode === "wiener"
                    ? "Phase increment σ (rad/sample)"
                    : "Phase jitter σ (rad)"
                }
                value={params.phaseNoiseStd}
                onChange={(n) => update("phaseNoiseStd", n)}
                min={0}
                max={2}
                step={0.01}
              />
              <Field label="Enable Rician fading (K=0 gives Rayleigh)">
                <input
                  type="checkbox"
                  checked={params.ricianEnabled}
                  onChange={(e) => update("ricianEnabled", e.target.checked)}
                />
              </Field>
              {params.ricianEnabled && (
                <NumberField
                  label="Fading coherence block (samples)"
                  value={params.fadingBlockSamples ?? 1}
                  onChange={(n) => update("fadingBlockSamples", n)}
                  min={1}
                  max={65536}
                />
              )}
              <NumberField
                label="Rician K (linear ratio)"
                value={params.ricianK}
                onChange={(n) => update("ricianK", n)}
                min={0}
                max={100}
              />
              <NumberField
                label="TWTA input drive (0 bypasses amplifier)"
                value={params.twtaDrive}
                onChange={(n) => update("twtaDrive", n)}
                min={0}
                max={4}
                step={0.1}
              />
              <NumberField
                label="Random seed"
                value={params.seed}
                onChange={(n) => update("seed", n)}
                min={0}
                max={4294967295}
              />
            </div>
          </details>
          <Button
            className="w-full"
            disabled={
              generation.isPending ||
              Object.values(params).some((v) => typeof v === "number" && !Number.isFinite(v))
            }
            onClick={() => void generate()}
          >
            {generation.isPending ? "Generating…" : "Generate signal"}
          </Button>
          <Button
            className="mt-2 w-full"
            variant="outline"
            onClick={() => {
              setParams(initial);
              setSignal(null);
              generation.reset();
            }}
          >
            Reset controls
          </Button>
        </Surface>
        <div className="min-w-0 space-y-4">
          {dirty && (
            <p
              role="status"
              className="border border-signal-warning/40 p-3 text-xs text-signal-warning"
            >
              Controls changed. Generate again to apply them; the plots show the last generated
              capture.
            </p>
          )}
          <CapturePlots capture={signal} />
          {signal && (
            <Surface className="p-4">
              <SectionHeading
                title="Generated capture"
                note={signal.classifierSupported ? "CNN CLASS SUPPORTED" : "SIMULATION ONLY"}
              />
              <Stats
                items={[
                  ["Modulation", signal.modulation],
                  ["Configured SNR", `${signal.snrDb} dB`],
                  ["RMS amplitude", numeric(signal.analysis.rmsAmplitude)],
                  ["PAPR", `${numeric(signal.analysis.features["PAPR_dB"])} dB`],
                ]}
              />
              <p className="my-4 text-xs leading-6 text-muted-foreground">
                AWGN SNR is the requested clean-signal/noise power ratio; a finite random
                realization may differ. The selected output policy determines whether channel gain
                is retained or final RMS is calibrated.{" "}
                {signal.classifierSupported
                  ? ""
                  : "This modulation is outside the CNN training classes; classification is unavailable."}
              </p>
              {signal.generation && (
                <>
                  <Stats
                    items={[
                      ["Realized simulator SNR", `${numeric(signal.generation.realizedSnrDb)} dB`],
                      ["Amplitude policy", signal.generation.outputPowerPolicy],
                      ["Modulation index", numeric(signal.generation.modulationIndex)],
                    ]}
                  />
                  <p className="my-3 text-xs leading-5 text-muted-foreground">
                    SNR here is known from separate clean/noise components inside the simulator, not
                    estimated from the received capture. Channel order:{" "}
                    {signal.generation.channelOrder.join(" → ")}. {signal.generation.referenceNote}
                  </p>
                  <details className="mb-4 text-xs">
                    <summary className="cursor-pointer text-primary">
                      Pre-calibration channel power
                    </summary>
                    <pre className="mt-3 overflow-auto">
                      {JSON.stringify(signal.generation.powerStages, null, 2)}
                    </pre>
                  </details>
                </>
              )}
              <div className="flex flex-wrap gap-2">
                <Button variant="outline" onClick={() => downloadCapture(signal)}>
                  Download I/Q CSV
                </Button>
                <Button variant="outline" onClick={() => void audio()}>
                  Download audible heterodyne
                </Button>
                {signal.classifierSupported && (
                  <Link to="/classifier" onClick={send}>
                    <Button>Send to classifier →</Button>
                  </Link>
                )}
                <Link to="/intelligence" onClick={send}>
                  <Button variant="outline">Analyze capture →</Button>
                </Link>
              </div>
            </Surface>
          )}
        </div>
      </div>
    </div>
  );
}
