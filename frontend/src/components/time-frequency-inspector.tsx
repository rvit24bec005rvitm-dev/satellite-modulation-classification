import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { researchApi, type TimeFrequencyReport } from "@/lib/research-api";
import { downloadJson, type Capture, type IQSample } from "@/lib/satmod-api";
import { ScientificChart } from "@/components/scientific-chart";
import { Button } from "@/components/ui/button";
import { Field, Message, numeric } from "@/components/workspace-ui";

const FFT_SIZES = [16, 32, 64, 128, 256, 512, 1024];

export function TimeFrequencyInspector({ capture }: { capture: Capture }) {
  const availableSizes = FFT_SIZES.filter((size) => size <= capture.samples.length);
  const maximumSize = availableSizes.at(-1) ?? 16;
  const initialSize = Math.min(64, maximumSize);
  const [fftSize, setFftSize] = useState(initialSize);
  const [overlap, setOverlap] = useState(.5);
  const [windowName, setWindowName] = useState("hann");
  const [mode, setMode] = useState("heatmap");
  const [marker, setMarker] = useState("");
  const [report, setReport] = useState<TimeFrequencyReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const generation = useRef(0);
  const markerHelp = useId();
  const known = capture.metadata?.sampleRateKnown;
  const run = useCallback(async (samples: IQSample[], fs: number, fft: number, fraction: number, window: string, sampleRateKnown?: boolean) => {
    const request = ++generation.current;
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const result = await researchApi.timeFrequency(samples, fs, fft, fraction, window, sampleRateKnown);
      if (generation.current === request) setReport(result);
    } catch (failure) {
      if (generation.current === request) setError(failure instanceof Error ? failure : new Error("Time-frequency analysis failed."));
    } finally {
      if (generation.current === request) setLoading(false);
    }
  }, []);
  useEffect(() => {
    setFftSize(initialSize);
    setOverlap(.5);
    setWindowName("hann");
    setMarker("");
    void run(capture.samples, capture.sampleRate, initialSize, .5, "hann", known);
    return () => { generation.current += 1; };
  }, [capture.samples, capture.sampleRate, initialSize, known, run]);

  const fft = Math.min(fftSize, maximumSize);
  const validControls = Number.isFinite(overlap) && overlap >= 0 && overlap < 1;
  const markerFrequency = marker === "" ? null : Number(marker);
  const nyquist = report?.frequencyUnit === "Hz" ? capture.sampleRate / 2 : .5;
  const markerValid = markerFrequency !== null && Number.isFinite(markerFrequency) && Math.abs(markerFrequency) <= nyquist;
  const markerError = markerFrequency !== null && !markerValid
    ? new Error(`Frequency marker must lie between ${-nyquist} and ${nyquist} ${report?.frequencyUnit ?? "cycles/sample"}.`)
    : null;
  const shapes = useMemo(() => report && markerValid && mode === "heatmap" ? [{
    type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: markerFrequency, y1: markerFrequency,
    line: { color: "#f3b04f", dash: "dash", width: 1.5 },
  }] : [], [report, markerValid, mode, markerFrequency]);
  const data = useMemo(() => {
    if (!report) return [];
    const traces = [{
      x: report.times, y: report.frequencies, z: report.db,
      type: mode === "waterfall" ? "surface" : "heatmap",
      colorscale: "Viridis", colorbar: { title: { text: "PSD density (dB)" } },
      name: "Computed time-frequency PSD",
      hovertemplate: `Time: %{x} ${report.timeUnit}<br>Frequency: %{y} ${report.frequencyUnit}<br>PSD: %{z} ${report.densityUnit}<extra></extra>`,
    }];
    if (mode === "waterfall" && markerValid) {
      const index = report.frequencies.reduce((best, frequency, at) =>
        Math.abs(frequency - markerFrequency!) < Math.abs(report.frequencies[best]! - markerFrequency!) ? at : best, 0);
      return [...traces, {
        type: "scatter3d", mode: "lines", x: report.times,
        y: report.times.map(() => report.frequencies[index]!), z: report.db[index]!,
        line: { color: "#f3b04f", width: 4 }, name: "Computed PSD at nearest marker bin",
        hovertemplate: `Time: %{x} ${report.timeUnit}<br>Nearest computed frequency: %{y} ${report.frequencyUnit}<br>PSD: %{z} ${report.densityUnit}<extra></extra>`,
      }];
    }
    return traces;
  }, [report, mode, markerValid, markerFrequency]);

  return <div className="space-y-3" aria-busy={loading}>
    <fieldset className="grid min-w-0 gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <legend className="mb-3 text-sm font-semibold">Time-frequency analysis controls</legend>
      <Field label="STFT FFT size">
        <select aria-label="STFT FFT size" className="control" value={fft} onChange={(event) => setFftSize(Number(event.target.value))}>
          {availableSizes.map((size) => <option key={size} value={size}>{size} samples</option>)}
        </select>
      </Field>
      <Field label="STFT overlap fraction">
        <input aria-label="STFT overlap fraction" className="control" type="number" min={0} max={.999} step={.05} value={Number.isFinite(overlap) ? overlap : ""} onChange={(event) => setOverlap(event.target.valueAsNumber)} />
      </Field>
      <Field label="STFT window">
        <select aria-label="STFT window" className="control" value={windowName} onChange={(event) => setWindowName(event.target.value)}>
          <option value="hann">Periodic Hann</option><option value="boxcar">Boxcar</option>
        </select>
      </Field>
      <Field label="Time-frequency view">
        <select aria-label="Time-frequency view" className="control" value={mode} onChange={(event) => setMode(event.target.value)}>
          <option value="heatmap">Spectrogram heatmap</option><option value="waterfall">3D waterfall (WebGL)</option>
        </select>
      </Field>
    </fieldset>
    <div className="flex flex-wrap gap-2">
      <Button disabled={loading || !validControls} onClick={() => void run(capture.samples, capture.sampleRate, fft, overlap, windowName, known)}>
        {loading ? "Computing time-frequency density…" : "Recompute time-frequency analysis"}
      </Button>
      <Button variant="outline" disabled={!report} onClick={() => report && downloadJson({
        reportVersion: "satmod-time-frequency-export-v1", source: capture.source, analysis: report,
        view: { mode, operatorFrequencyMarker: markerValid ? markerFrequency : null },
      }, "satmod-time-frequency.json")}>Export time-frequency JSON</Button>
    </div>
    {loading && <p role="status" className="text-xs text-muted-foreground">Computing actual complete-window PSD from the selected I/Q samples.</p>}
    <Message error={error ?? markerError} />
    {!validControls && <p role="alert" className="text-xs text-destructive">Overlap must be a finite fraction from 0 up to, but not including, 1.</p>}
    {report && <>
      <p className="text-xs text-muted-foreground">Computed settings: FFT {report.parameters.fftSize}, overlap {report.parameters.overlap}, window {report.parameters.window}. Density unit: {report.densityUnit}. Recompute after changing settings.</p>
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label={`Baseband frequency marker (${report.frequencyUnit})`}>
          <input aria-label="Baseband frequency marker" aria-describedby={markerHelp} className="control" type="number" min={-nyquist} max={nyquist} step="any" value={marker} onChange={(event) => setMarker(event.target.value)} placeholder="Optional operator annotation" />
        </Field>
        <div className="flex items-center"><Button variant="outline" disabled={marker === ""} onClick={() => setMarker("")}>Clear frequency marker</Button></div>
      </div>
      <p id={markerHelp} className="text-xs text-muted-foreground">The marker is an operator annotation. The waterfall traces the nearest plotted frequency bin; computed peak bins use the full FFT. Peaks are spectral diagnostics, not verified RF carriers. Use heatmap when WebGL is unavailable.</p>
      <ScientificChart data={data} shapes={shapes} xTitle={report.timeUnit === "s" ? "Window time (s)" : "Window sample index"} yTitle={`Baseband frequency (${report.frequencyUnit})`} zTitle={report.densityUnit} title={mode === "waterfall" ? "Computed PSD waterfall" : "Computed time-frequency PSD"} height={360} />
      <p className="text-xs text-muted-foreground">{report.note}</p>
      <div className="flex flex-wrap gap-2" aria-label="Computed spectral peak diagnostics">
        {report.peaks.map((peak, index) => <Button key={`${index}:${peak.frequency}`} variant="outline" onClick={() => setMarker(String(peak.frequency))}>
          Peak {numeric(peak.frequency, 5)} {report.frequencyUnit} · {numeric(peak.densityDb, 4)} dB
        </Button>)}
        {!report.peaks.length && <p className="text-xs text-muted-foreground">No local spectral peaks above the display floor.</p>}
      </div>
    </>}
  </div>;
}
