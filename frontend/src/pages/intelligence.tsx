import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useSignal, useAnalysis, useSamples } from "@/hooks/use-satmod";
import { satmodApi, downloadJson, type Capture } from "@/lib/satmod-api";
import { Button } from "@/components/ui/button";
import {
  Surface,
  SectionHeading,
  Message,
  Stats,
  numeric,
  Field,
  Empty,
  ExportButton,
} from "@/components/workspace-ui";
import { CapturePlots, DistributionPlots, ScientificChart } from "@/components/scientific-chart";
export function IntelligencePage() {
  const { capture, setCapture } = useSignal();
  const analyzer = useAnalysis(),
    samples = useSamples();
  const [second, setSecond] = useState<Capture | null>(null);
  const [sampleId, setSampleId] = useState(142);
  const loader = useMutation({ mutationFn: () => satmodApi.sample(sampleId) });
  const comparison = useMutation({
    mutationFn: () => {
      if (!capture || !second) throw new Error("Select two captures first.");
      return satmodApi.compare(capture, second);
    },
  });
  const load = async (target: "first" | "second") => {
    try {
      const c = await loader.mutateAsync();
      if (target === "first") setCapture(c);
      else setSecond(c);
      comparison.reset();
    } catch {
      /* Render mutation error. */
    }
  };
  const reanalyze = async () => {
    if (capture)
      try {
        const analysis = await analyzer.mutateAsync(capture);
        setCapture({ ...capture, analysis });
      } catch {
        /* Render mutation error. */
      }
  };
  return (
    <div className="space-y-5">
      <Surface className="p-4">
        <SectionHeading title="Capture selection" note="FULL RESOLUTION" />
        <div className="grid gap-4 sm:grid-cols-[1fr_auto_auto]">
          <Field label="Demo capture">
            <select
              className="control"
              value={sampleId}
              onChange={(e) => setSampleId(Number(e.target.value))}
            >
              {samples.data?.map((s) => (
                <option key={s.id} value={s.id}>
                  #{s.id} · {s.modulation} · {s.snrDb} dB
                </option>
              ))}
            </select>
          </Field>
          <Button
            variant="outline"
            className="sm:mt-6"
            disabled={loader.isPending}
            onClick={() => void load("first")}
          >
            Load as A
          </Button>
          <Button
            variant="outline"
            className="sm:mt-6"
            disabled={loader.isPending}
            onClick={() => void load("second")}
          >
            Load as B
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">
          Capture A also receives signals from Satellite Studio and Live Classifier.
        </p>
        <Message error={loader.error ?? samples.error} />
      </Surface>
      {capture ? (
        <>
          <Surface className="p-4">
            <div className="flex flex-wrap justify-between gap-3">
              <SectionHeading title="Signal quality" note={capture.source} />
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  disabled={analyzer.isPending}
                  onClick={() => void reanalyze()}
                >
                  Refresh analysis
                </Button>
                <ExportButton
                  onClick={() => downloadJson(capture.analysis, "satmod-analysis.json")}
                />
              </div>
            </div>
            <Message error={analyzer.error} />
            <Stats
              items={[
                ["Power (relative units)", numeric(capture.analysis.signalPower)],
                ["Power (dB relative)", numeric(capture.analysis.powerDb)],
                ["RMS amplitude", numeric(capture.analysis.rmsAmplitude)],
                [
                  "99% occupied bandwidth",
                  `${numeric(capture.analysis.bandwidthHz)} ${capture.sampleRate === 1 ? "cycles/sample" : "Hz"}`,
                ],
              ]}
            />
            <p className="mt-4 text-xs leading-6 text-muted-foreground">
              {capture.analysis.qualityNote} Configured SNR, if known:{" "}
              {capture.snrDb === undefined ? "Unknown" : `${capture.snrDb} dB`}.
            </p>
          </Surface>
          <CapturePlots capture={capture} />
          <DistributionPlots analysis={capture.analysis} />
          <Surface className="p-4">
            <SectionHeading title="I/Q statistical properties" note="NUMPY / SCIPY" />
            <div className="grid grid-cols-2 gap-x-6 md:grid-cols-3">
              {Object.entries(capture.analysis.features).map(([key, value]) => (
                <div
                  key={key}
                  className="flex justify-between gap-2 border-b border-border py-3 text-xs"
                >
                  <span className="break-words text-muted-foreground">
                    {key.replaceAll("_", " ")}
                  </span>
                  <span className="font-mono">{numeric(value)}</span>
                </div>
              ))}
            </div>
          </Surface>
        </>
      ) : (
        <Empty />
      )}
      <Surface className="p-4">
        <SectionHeading title="Two-signal comparison" note="A / B CAPTURES" />
        <div className="grid gap-4 md:grid-cols-2">
          <p className="break-words text-sm">A: {capture?.source ?? "No capture"}</p>
          <p className="break-words text-sm">B: {second?.source ?? "Load a capture as B above"}</p>
        </div>
        <Button
          className="mt-4"
          disabled={!capture || !second || comparison.isPending}
          onClick={() => comparison.mutate()}
        >
          {comparison.isPending ? "Comparing…" : "Compare signals"}
        </Button>
        <Message error={comparison.error} />
        {comparison.data && (
          <>
            <Stats
              items={[
                ["A − B power", `${numeric(comparison.data.powerDifferenceDb)} dB`],
                ["Sample MSE", numeric(comparison.data.sampleMse)],
                ["A samples", comparison.data.first.sampleCount],
                ["B samples", comparison.data.second.sampleCount],
              ]}
            />
            <div className="mt-4 grid gap-4 lg:grid-cols-2">
              <ScientificChart
                data={[comparison.data.first, comparison.data.second].map((a, i) => ({
                  x: a.waveform.map((p) => p.time),
                  y: a.waveform.map((p) => p.i),
                  mode: "lines",
                  type: "scattergl",
                  name: i === 0 ? "Capture A" : "Capture B",
                }))}
                xTitle={comparison.data.first.sampleRate === 1 ? "Sample index" : "Time (s)"}
                yTitle="In-phase amplitude (relative)"
              />
              <ScientificChart
                data={[comparison.data.first, comparison.data.second].map((a, i) => ({
                  x: a.spectrum.map((p) => p.frequency),
                  y: a.spectrum.map((p) => p.psd),
                  mode: "lines",
                  type: "scattergl",
                  name: i === 0 ? "Capture A" : "Capture B",
                }))}
                xTitle="Frequency (Hz; cycles/sample when Fs=1)"
                yTitle="PSD (dB relative per frequency unit)"
              />
            </div>
            <p className="text-xs text-muted-foreground">{comparison.data.note}</p>
          </>
        )}
      </Surface>
    </div>
  );
}
