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
import { TimeFrequencyInspector } from "./time-frequency-inspector";
interface PlotTrace {
  type?: string;
  mode?: string;
  name?: string;
  x?: (number | string | null)[];
  y?: (number | string | null)[];
  z?: (number | null)[] | (number | null)[][];
  marker?: Record<string, unknown>;
  line?: Record<string, unknown>;
  [key: string]: unknown;
}
interface ChartProps {
  data: PlotTrace[];
  xTitle?: string;
  yTitle?: string;
  zTitle?: string;
  title?: string;
  height?: number;
  equal?: boolean;
  shapes?: Record<string, unknown>[];
  description?: string;
}
export function ScientificChart({
  data,
  xTitle = "",
  yTitle = "",
  zTitle = "",
  title = "",
  height = 280,
  equal = false,
  shapes = [],
  description,
}: ChartProps) {
  const element = useRef<HTMLDivElement>(null);
  const renderQueue = useRef<Promise<void>>(Promise.resolve());
  const disposePlot = useRef<(() => void) | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const { preferences } = useSignal();
  // Job polling recreates equivalent trace objects. Compare their bounded
  // numeric content so a progress update does not purge a user's zoomed plot.
  const serialized = JSON.stringify(data);
  const serializedShapes = JSON.stringify(shapes);
  useEffect(
    () => () => {
      // A render already in flight must finish before releasing its resources.
      // Prop changes reuse the same graph; only unmount purges it.
      void renderQueue.current.finally(() => {
        disposePlot.current?.();
        disposePlot.current = null;
      });
    },
    [],
  );
  useEffect(() => {
    let disposed = false;
    const target = element.current;
    if (!target) return;
    setLoading(true);
    setError("");
    renderQueue.current = renderQueue.current
      .then(async () => {
        if (disposed) return;
        const { default: Plotly } = await import("plotly.js-dist-min");
        if (disposed) return;
        const traces = JSON.parse(serialized) as PlotTrace[];
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
          traces,
          {
            title: { text: title, font: { size: 12 } },
            height,
            autosize: true,
            uirevision: `${title}:${xTitle}:${yTitle}`,
            shapes: JSON.parse(serializedShapes),
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
            scene: {
              xaxis: { title: { text: xTitle } },
              yaxis: { title: { text: yTitle } },
              zaxis: { title: { text: zTitle } },
            },
            legend: { orientation: "h", y: 1.15 },
            showlegend: traces.length > 1,
          },
          {
            responsive: true,
            displaylogo: false,
            modeBarButtonsToRemove: ["sendChartToCloud"],
            displayModeBar: preferences.charts === "interactive",
            scrollZoom: preferences.charts === "interactive",
          },
        );
        if (!disposePlot.current) {
          const observer = new ResizeObserver(() => Plotly.Plots.resize(target));
          observer.observe(target);
          disposePlot.current = () => {
            observer.disconnect();
            Plotly.purge(target);
          };
        }
        if (!disposed) setLoading(false);
      })
      .catch(() => {
        if (!disposed) {
          setLoading(false);
          setError("Scientific chart could not load. Refresh after dependencies are installed.");
        }
      });
    return () => {
      disposed = true;
    };
  }, [
    serialized,
    serializedShapes,
    xTitle,
    yTitle,
    zTitle,
    title,
    height,
    equal,
    preferences.charts,
    preferences.theme,
  ]);
  const tableTrace = data.find((trace) => trace.x && trace.y);
  return (
    <div className="min-w-0" role="figure" aria-label={title || `${yTitle} versus ${xTitle}`}>
      {error && <div role="alert">{error}</div>}
      {loading && (
        <span className="sr-only" role="status">
          Rendering computed signal plot…
        </span>
      )}
      <div
        ref={element}
        role="group"
        aria-label={title || `${yTitle} versus ${xTitle}`}
        aria-busy={loading}
        tabIndex={0}
        className="w-full min-w-0"
        style={{ height }}
      />
      {description && <p className="text-xs text-muted-foreground">{description}</p>}
      {tableTrace?.x && tableTrace.y && (
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer focus-visible:outline-2 focus-visible:outline-primary">
            Accessible plot data preview
          </summary>
          <p>
            First {Math.min(20, tableTrace.x.length)} points of{" "}
            {tableTrace.name ?? "the first trace"}. Full values are available in the analysis
            export.
          </p>
          <table className="w-full">
            <caption className="sr-only">
              {title || `${yTitle} versus ${xTitle}`} data preview
            </caption>
            <thead>
              <tr>
                <th>{xTitle}</th>
                <th>{yTitle}</th>
              </tr>
            </thead>
            <tbody>
              {tableTrace.x.slice(0, 20).map((x, index) => (
                <tr key={index}>
                  <td>{x ?? "Unavailable"}</td>
                  <td>{tableTrace.y?.[index] ?? "Unavailable"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </div>
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
            ...(a.phaseTrace ? ["phase"] : []),
            ...(a.instantaneousFrequency ? ["instantaneous frequency"] : []),
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
          {["GMSK", "BFSK"].includes(capture.modulation ?? "") && (
            <p className="mb-3 text-xs text-muted-foreground">
              Continuous-phase trajectory samples; no static symbol constellation. Use the phase and
              instantaneous-frequency views to inspect modulation.
            </p>
          )}
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
          <TimeFrequencyInspector capture={capture} />
        </TabsContent>
        {a.phaseTrace && (
          <TabsContent value="phase">
            <ScientificChart
              data={[
                {
                  x: a.phaseTrace.map((p) => p.time),
                  y: a.phaseTrace.map((p) => p.phaseRad ?? NaN),
                  type: "scattergl",
                  mode: "lines",
                  name: "Unwrapped phase",
                },
              ]}
              xTitle={a.sampleRate === 1 ? "Sample index" : "Time (s)"}
              yTitle="Unwrapped phase (radians)"
            />
          </TabsContent>
        )}
        {a.instantaneousFrequency && (
          <TabsContent value="instantaneous frequency">
            <ScientificChart
              data={[
                {
                  x: a.instantaneousFrequency.map((p) => p.time),
                  y: a.instantaneousFrequency.map((p) => p.frequency ?? NaN),
                  type: "scattergl",
                  mode: "lines",
                  name: "Instantaneous frequency",
                },
              ]}
              xTitle={a.sampleRate === 1 ? "Sample index" : "Time (s)"}
              yTitle={a.sampleRate === 1 ? "Frequency (cycles/sample)" : "Frequency (Hz)"}
            />
            <p className="text-xs text-muted-foreground">{a.instantaneousFrequencyNote}</p>
          </TabsContent>
        )}
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
