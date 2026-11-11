import { ExternalWorkspace } from "@/components/external-workspace";
import { PersistentHistory } from "@/components/persistent-history";
import { useState } from "react";
import { Link } from "@tanstack/react-router";
import { useQueryClient } from "@tanstack/react-query";
import { Play, Upload } from "lucide-react";
import { toast } from "sonner";
import {
  satmodApi,
  downloadJson,
  SUPPORTED_MODULATIONS,
  type ClassificationResult,
  type Capture,
} from "@/lib/satmod-api";
import { useSignal, useSamples, useClassification } from "@/hooks/use-satmod";
import { Button } from "@/components/ui/button";
import {
  Message,
  Surface,
  SectionHeading,
  Field,
  NumberField,
  Empty,
  percent,
  numeric,
  Stats,
  ExportButton,
} from "@/components/workspace-ui";
import { ClassificationInsights } from "@/components/classification-insights";
import { CapturePlots, ScientificChart } from "@/components/scientific-chart";

export function ClassifierPage() {
  const { capture, setCapture, setClassification } = useSignal();
  const samples = useSamples();
  const classify = useClassification();
  const queryClient = useQueryClient();
  const [sampleId, setSampleId] = useState(42);
  const [classFilter, setClassFilter] = useState("All");
  const [snrFilter, setSnrFilter] = useState("All");
  const [windowStart, setWindowStart] = useState(0);
  const [sampleRate, setSampleRate] = useState(capture?.sampleRate ?? 1);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [busy, setBusy] = useState(false);
  const accept = (next: Capture) => {
    setCapture(next);
    setWindowStart(0);
    setSampleRate(next.sampleRate);
    setResult(null);
    setError(null);
  };
  const load = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      accept(await satmodApi.sample(sampleId));
    } catch (e) {
      setError(e as Error);
    } finally {
      setBusy(false);
    }
  };
  const previousCaptureFor = (value: ClassificationResult): Capture => ({
    ...value,
    ...(capture?.source === value.source
      ? { modulation: capture.modulation, snrDb: capture.snrDb }
      : {}),
  });
  const complete = (value: ClassificationResult) => {
    setResult(value);
    setClassification(value);
    setCapture(previousCaptureFor(value));
    void queryClient.invalidateQueries({ queryKey: ["history"] });
    toast.success("CNN classification complete");
  };
  const run = async () => {
    if (!capture) return;
    setError(null);
    setResult(null);
    try {
      complete(
        await classify.mutateAsync({
          samples: capture.samples,
          source: capture.source,
          sampleRate,
          windowStart,
        }),
      );
    } catch (e) {
      setError(e as Error);
    }
  };
  const upload = async (file?: File) => {
    if (!file) return;
    setError(null);
    setResult(null);
    classify.reset();
    if (file.size > 4194304) {
      setError(new Error("Capture exceeds the 4 MiB upload limit."));
      return;
    }
    setBusy(true);
    try {
      const value = await satmodApi.upload(file);
      setSampleRate(value.sampleRate);
      setCapture(value);
      setWindowStart(0);
      complete(value);
    } catch (e) {
      setError(e as Error);
    } finally {
      setBusy(false);
    }
  };
  // Reset on external capture changes without discarding a just-uploaded result.
  const unsupported =
    !!capture?.modulation && !SUPPORTED_MODULATIONS.some((m) => m === capture.modulation);
  const displayedResult =
    result &&
    result.source === capture?.source &&
    result.samples === capture.samples &&
    result.sampleRate === sampleRate &&
    Number(result.preprocessing["windowStart"]) === windowStart
      ? result
      : null;
  const filtered =
    samples.data?.filter(
      (s) =>
        (classFilter === "All" || s.modulation === classFilter) &&
        (snrFilter === "All" || s.snrDb === Number(snrFilter)),
    ) ?? [];
  return (
    <div className="space-y-4">
      <ExternalWorkspace />
      <PersistentHistory />
      <Message error={error ?? classify.error} />
      {unsupported && (
        <p
          role="status"
          className="border border-signal-warning/40 p-3 text-sm text-signal-warning"
        >
          This generated modulation is outside the three CNN training classes. Use Signal
          Intelligence to inspect it.
        </p>
      )}
      <div className="grid items-start gap-4 xl:grid-cols-[280px_minmax(0,1fr)]">
        <Surface className="p-4">
          <SectionHeading title="Signal input" note="I / Q DATA" />
          <Message error={samples.error} loading={samples.isPending} />
          <Field label="Demo modulation">
            <select
              className="control"
              value={classFilter}
              onChange={(e) => setClassFilter(e.target.value)}
            >
              {["All", "BPSK", "QPSK", "16-QAM"].map((v) => (
                <option key={v}>{v}</option>
              ))}
            </select>
          </Field>
          <Field label="Demo SNR (dB)">
            <select
              className="control"
              value={snrFilter}
              onChange={(e) => setSnrFilter(e.target.value)}
            >
              <option>All</option>
              {[...new Set(samples.data?.map((s) => s.snrDb))]
                .sort((a, b) => a - b)
                .map((n) => (
                  <option key={n}>{n}</option>
                ))}
            </select>
          </Field>
          <Field label="Saved demo sample">
            <select
              className="control"
              value={filtered.some((s) => s.id === sampleId) ? sampleId : ""}
              onChange={(e) => setSampleId(Number(e.target.value))}
            >
              <option value="" disabled>
                Select capture
              </option>
              {filtered.map((s) => (
                <option key={s.id} value={s.id}>
                  #{s.id} · {s.modulation} · {s.snrDb} dB
                </option>
              ))}
            </select>
          </Field>
          <Button
            className="w-full"
            variant="outline"
            onClick={() => void load()}
            disabled={busy || classify.isPending || !filtered.some((s) => s.id === sampleId)}
          >
            Load demo sample
          </Button>
          <div className="my-5 border-t border-border" />
          <Field label="Upload and classify I/Q capture">
            <input
              aria-label="Upload I/Q file"
              className="control"
              type="file"
              accept=".npy,.npz,.csv,.mat,.wav,.json"
              disabled={busy || classify.isPending}
              onChange={(e) => {
                void upload(e.target.files?.[0]);
                e.currentTarget.value = "";
              }}
            />
          </Field>
          <p className="text-[11px] leading-5 text-muted-foreground">
            NPY, NPZ, CSV, MATLAB, WAV, JSON · max 4 MiB. At least 128 samples; the selected
            128-sample window is classified.
          </p>
          <div className="my-5 border-t border-border" />
          <NumberField
            label="Window start (sample offset)"
            value={windowStart}
            onChange={(n) => {
              setWindowStart(n);
              setResult(null);
            }}
            min={0}
            max={Math.max(0, (capture?.samples.length ?? 128) - 128)}
          />
          <NumberField
            label="Capture sample rate (Hz; 1 = normalized)"
            value={sampleRate}
            onChange={setSampleRate}
            min={1}
            max={1e9}
          />
          <p className="mb-3 text-xs text-muted-foreground">
            Reclassify to apply a changed sample clock to the analytical plots. I/Q amplitudes are
            preserved.
          </p>
          <Button
            className="w-full"
            disabled={
              !capture ||
              unsupported ||
              busy ||
              classify.isPending ||
              !Number.isInteger(windowStart)
            }
            onClick={() => void run()}
          >
            <Play size={15} />
            {classify.isPending || busy ? "Processing…" : "Run CNN classification"}
          </Button>
          <Link to="/studio" className="mt-4 block text-center text-xs text-primary">
            Generate a synthetic signal →
          </Link>
        </Surface>
        <div className="min-w-0 space-y-4">
          <CapturePlots capture={capture} />
          <Surface className="p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <SectionHeading title="Classification result" note="REAL MODEL RESPONSE" />
              <ExportButton
                disabled={!displayedResult}
                onClick={() =>
                  displayedResult && downloadJson(displayedResult, "satmod-classification.json")
                }
              />
            </div>
            {displayedResult ? (
              <>
                <Stats
                  items={[
                    ["Predicted modulation", displayedResult.predictedModulation],
                    ["Confidence", percent(displayedResult.confidence)],
                    ["Shannon entropy", `${numeric(displayedResult.entropyBits)} bits`],
                    ["Inference latency", `${numeric(displayedResult.processingMs)} ms`],
                  ]}
                />
                <ScientificChart
                  data={[
                    {
                      x: Object.keys(displayedResult.probabilities),
                      y: Object.values(displayedResult.probabilities).map((p) => p * 100),
                      type: "bar",
                      text: Object.values(displayedResult.probabilities).map(percent),
                      textposition: "auto",
                    },
                  ]}
                  xTitle="Modulation"
                  yTitle="Probability (%)"
                  height={230}
                />
                {displayedResult.warnings.map((w) => (
                  <p
                    key={w}
                    className="mt-2 border border-signal-warning/30 p-3 text-xs text-signal-warning"
                  >
                    {w}
                  </p>
                ))}
                <p className="mt-3 text-xs leading-6 text-muted-foreground">
                  Uncertainty range: 0–1.585 bits. Entropy measures ambiguity among the three
                  trained classes; it does not reliably detect an unknown modulation. Preprocessing:
                  training z-score only; samples retain their amplitude.{" "}
                  {displayedResult.preprocessing["windowStart"]}–
                  {Number(displayedResult.preprocessing["windowStart"]) + 127} is the classified
                  window.
                </p>
              </>
            ) : (
              <Empty
                text={
                  busy || classify.isPending
                    ? "Processing signal…"
                    : "No CNN result for the current capture. Select a sample, generate a signal, or upload a capture."
                }
              />
            )}
          </Surface>
          {displayedResult && (
            <ClassificationInsights
              key={displayedResult.id}
              result={displayedResult}
              onExplanation={(explanation) =>
                setResult((previous) => (previous ? { ...previous, explanation } : previous))
              }
            />
          )}
          {capture && (
            <Surface className="p-4">
              <SectionHeading title="Signal metrics" />
              <Stats
                items={[
                  ["Power (relative)", numeric(capture.analysis.signalPower)],
                  ["RMS amplitude", numeric(capture.analysis.rmsAmplitude)],
                  [
                    "Occupied bandwidth (99%)",
                    `${numeric(capture.analysis.bandwidthHz)} ${capture.sampleRate === 1 ? "cycles/sample" : "Hz"}`,
                  ],
                  ["Source", capture.modulation ?? "Uploaded capture"],
                ]}
              />
              <p className="mt-3 text-xs text-muted-foreground">{capture.analysis.qualityNote}</p>
            </Surface>
          )}
        </div>
      </div>
    </div>
  );
}
