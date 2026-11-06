import { useEffect, useRef, useState } from "react";
import { useSignal } from "@/hooks/use-satmod";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { Capture, SignalAnalysis } from "@/lib/satmod-api";
import { Empty, SectionHeading, Surface } from "./workspace-ui";
interface PlotTrace {
  type?: string;
  mode?: string;
  name?: string;
  x?: number[] | string[];
  y?: number[] | string[];
  z?: number[][];
  marker?: Record<string, unknown>;
  line?: Record<string, unknown>;
  [key: string]: unknown;
}
interface ChartProps {
  data: PlotTrace[];
  xTitle?: string;
  yTitle?: string;
  title?: string;
  height?: number;
  equal?: boolean;
}
export function ScientificChart({
  data,
  xTitle = "",
  yTitle = "",
  title = "",
  height = 280,
  equal = false,
}: ChartProps) {
  const element = useRef<HTMLDivElement>(null);
  const [error, setError] = useState("");
  const { preferences } = useSignal();
  useEffect(() => {
    let disposed = false;
    const target = element.current;
    if (!target) return;
    let cleanup: (() => void) | undefined;
    void import("plotly.js-dist-min")
      .then(async ({ default: Plotly }) => {
        if (disposed) return;
        const style = getComputedStyle(document.documentElement);
        const canvas = document.createElement("canvas");
        canvas.width = canvas.height = 1;
        const context = canvas.getContext("2d");
        const color = (name: string) => {
          const value = style.getPropertyValue(name).trim();
          if (!context) return "#9aaec6";
          context.clearRect(0, 0, 1, 1);
          context.fillStyle = value;
          context.fillRect(0, 0, 1, 1);
          const rgba = context.getImageData(0, 0, 1, 1).data;
          return `rgb(${rgba[0]},${rgba[1]},${rgba[2]})`;
        };
        await Plotly.react(
          target,
          data,
          {
            title: { text: title, font: { size: 12 } },
            height,
            autosize: true,
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            font: { color: color("--muted-foreground"), family: "Inter, sans-serif", size: 10 },
            colorway: [
              color("--signal-i"),
              color("--signal-q"),
              color("--signal-violet"),
              color("--signal-warning"),
            ],
            margin: { l: 65, r: 20, b: 55, t: title ? 35 : 15 },
            xaxis: {
              title: { text: xTitle },
              gridcolor: color("--border"),
              zerolinecolor: color("--border"),
            },
            yaxis: {
              title: { text: yTitle },
              gridcolor: color("--border"),
              zerolinecolor: color("--border"),
              ...(equal ? { scaleanchor: "x", scaleratio: 1 } : {}),
            },
            legend: { orientation: "h", y: 1.15 },
            showlegend: data.length > 1,
          },
          {
            responsive: true,
            displaylogo: false,
            displayModeBar: preferences.charts === "interactive",
            scrollZoom: preferences.charts === "interactive",
          },
        );
        const observer = new ResizeObserver(() => Plotly.Plots.resize(target));
        observer.observe(target);
        cleanup = () => {
          observer.disconnect();
          Plotly.purge(target);
        };
        if (disposed) cleanup();
      })
      .catch(() => {
        if (!disposed)
          setError("Scientific chart could not load. Refresh after dependencies are installed.");
      });
    return () => {
      disposed = true;
      cleanup?.();
    };
  }, [data, xTitle, yTitle, title, height, equal, preferences.charts, preferences.theme]);
  return error ? (
    <div role="alert">{error}</div>
  ) : (
    <div
      ref={element}
      role="img"
      aria-label={title || `${yTitle} versus ${xTitle}`}
      className="w-full min-w-0"
      style={{ height }}
    />
  );
}
export function CapturePlots({ capture }: { capture: Capture | null }) {
  if (!capture) return <Empty />;
  const a = capture.analysis;
  const traces = ["i", "q"] as const;
  return (
    <Surface className="p-4">
      <SectionHeading
        title="Capture inspector"
        note={`${a.sampleCount.toLocaleString()} SAMPLES`}
      />
      <p className="mb-3 break-words text-xs text-muted-foreground">
        {capture.source} · Fs {a.sampleRate} {a.sampleRate === 1 ? "(normalized)" : "Hz"} · previews
        ≤1,024 points
      </p>
      <Tabs defaultValue="waveform">
        <TabsList className="flex h-auto flex-wrap justify-start bg-muted">
          {[
            "waveform",
            "constellation",
            "spectrum",
            "psd",
            "spectrogram",
            ...(a.eye ? ["eye"] : []),
          ].map((v) => (
            <TabsTrigger key={v} value={v} className="text-xs capitalize">
              {v === "psd" ? "PSD" : v}
            </TabsTrigger>
          ))}
        </TabsList>
        <TabsContent value="waveform">
          <ScientificChart
            data={traces.map((c) => ({
              x: a.waveform.map((p) => p.time),
              y: a.waveform.map((p) => p[c]),
              type: "scattergl",
              mode: "lines",
              name: c.toUpperCase(),
            }))}
            xTitle={a.sampleRate === 1 ? "Sample index" : "Time (s)"}
            yTitle="Amplitude (relative)"
          />
        </TabsContent>
        <TabsContent value="constellation">
          <ScientificChart
            data={[
              {
                x: a.constellation.map((p) => p.i),
                y: a.constellation.map((p) => p.q),
                type: "scattergl",
                mode: "markers",
                marker: { size: 5, opacity: 0.65 },
              },
            ]}
            xTitle="In-phase I"
            yTitle="Quadrature Q"
            equal
          />
        </TabsContent>
        <TabsContent value="spectrum">
          <ScientificChart
            data={[
              {
                x: a.spectrum.map((p) => p.frequency),
                y: a.spectrum.map((p) => p.magnitude),
                type: "scattergl",
                mode: "lines",
              },
            ]}
            xTitle={a.sampleRate === 1 ? "Frequency (cycles/sample)" : "Frequency (Hz)"}
            yTitle="FFT magnitude (dB relative)"
          />
        </TabsContent>
        <TabsContent value="psd">
          <ScientificChart
            data={[
              {
                x: a.spectrum.map((p) => p.frequency),
                y: a.spectrum.map((p) => p.psd),
                type: "scattergl",
                mode: "lines",
              },
            ]}
            xTitle={a.sampleRate === 1 ? "Frequency (cycles/sample)" : "Frequency (Hz)"}
            yTitle={a.sampleRate === 1 ? "PSD (dB / cycles per sample)" : "PSD (dB relative / Hz)"}
          />
        </TabsContent>
        <TabsContent value="spectrogram">
          <ScientificChart
            data={[
              {
                x: a.spectrogram.times,
                y: a.spectrogram.frequencies,
                z: a.spectrogram.db,
                type: "heatmap",
                colorscale: "Viridis",
              },
            ]}
            xTitle={a.sampleRate === 1 ? "Sample index" : "Time (s)"}
            yTitle={a.sampleRate === 1 ? "Frequency (cycles/sample)" : "Frequency (Hz)"}
          />
        </TabsContent>
        {a.eye && (
          <TabsContent value="eye">
            <ScientificChart
              data={a.eye.i.map((y) => ({
                x: a.eye!.time,
                y,
                type: "scatter",
                mode: "lines",
                line: { color: "#36c9dd", width: 1 },
                showlegend: false,
              }))}
              xTitle="Symbol periods (known transmitter timing)"
              yTitle="In-phase I"
            />
          </TabsContent>
        )}
      </Tabs>
    </Surface>
  );
}
export function DistributionPlots({ analysis: a }: { analysis: SignalAnalysis }) {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {[
        ["Amplitude distribution", a.amplitudeDistribution, "Amplitude (relative)"],
        ["Phase distribution", a.phaseDistribution, "Phase (radians)"],
      ].map(([title, values, axis]) => {
        const points = values as SignalAnalysis["phaseDistribution"];
        return (
          <Surface key={String(title)} className="p-4">
            <SectionHeading title={String(title)} />
            <ScientificChart
              data={[{ x: points.map((p) => p.value), y: points.map((p) => p.count), type: "bar" }]}
              xTitle={String(axis)}
              yTitle="Sample count"
            />
          </Surface>
        );
      })}
    </div>
  );
}
export function SnrChart({ points }: { points: { snrDb: number; accuracy: number }[] }) {
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer>
        <LineChart data={points} margin={{ left: 0, right: 20, bottom: 20, top: 10 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 4" />
          <XAxis
            dataKey="snrDb"
            stroke="var(--muted-foreground)"
            label={{ value: "SNR (dB)", position: "bottom" }}
          />
          <YAxis
            domain={[0, 1]}
            stroke="var(--muted-foreground)"
            tickFormatter={(v) => `${Math.round(Number(v) * 100)}%`}
          />
          <Tooltip
            contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)" }}
          />
          <Line dataKey="accuracy" stroke="var(--signal-i)" dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
