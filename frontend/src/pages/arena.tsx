import { useState } from "react";
import { useModels, useBenchmark } from "@/hooks/use-satmod";
import { downloadJson } from "@/lib/satmod-api";
import { Button } from "@/components/ui/button";
import {
  Field,
  NumberField,
  Message,
  Surface,
  SectionHeading,
  Stats,
  percent,
  numeric,
  ExportButton,
} from "@/components/workspace-ui";
import { ScientificChart } from "@/components/scientific-chart";
export function ArenaPage() {
  const models = useModels(),
    benchmark = useBenchmark();
  const [modelId, setModelId] = useState("1d-cnn");
  const [sampleId, setSampleId] = useState(42);
  const [iterations, setIterations] = useState(30);
  const [compareId, setCompareId] = useState("resnet");
  const other = models.data?.find((m) => m.id === compareId);
  return (
    <div className="space-y-5">
      <Message loading={models.isPending} error={models.error} />
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {models.data?.map((m) => (
          <Surface key={m.id} className="p-4">
            <div
              className={`mb-6 font-mono text-[10px] uppercase ${m.status === "available" ? "text-signal-q" : "text-muted-foreground"}`}
            >
              {m.status.replace("-", " ")}
            </div>
            <h2 className="font-display text-xl">{m.name}</h2>
            <dl className="mt-5 space-y-3 text-xs">
              <div className="flex justify-between gap-2">
                <dt>Stored accuracy</dt>
                <dd className="text-primary">{percent(m.accuracy)}</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt>Parameters</dt>
                <dd>{m.parameters?.toLocaleString() ?? "—"}</dd>
              </div>
            </dl>
            <p className="mt-4 text-xs leading-5 text-muted-foreground">
              {m.backend ?? m.reason ?? m.error}
            </p>
          </Surface>
        ))}
      </div>
      <Surface className="p-5">
        <SectionHeading title="Measured inference benchmark" note="WARM CPU INFERENCE" />
        <p className="mb-5 text-xs leading-6 text-muted-foreground">
          Three warm-up runs precede the timed iterations. Measurements use a saved demo frame with
          the same preprocessing as Live Classifier. Results describe this host.
        </p>
        <div className="grid gap-4 sm:grid-cols-3">
          <Field label="Model">
            <select
              className="control"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
            >
              {models.data?.map((m) => (
                <option key={m.id} value={m.id} disabled={m.status !== "available"}>
                  {m.name} {m.status !== "available" ? "— Not Trained" : ""}
                </option>
              ))}
            </select>
          </Field>
          <NumberField
            label="Demo sample ID"
            value={sampleId}
            onChange={setSampleId}
            min={0}
            max={299}
          />
          <NumberField
            label="Timed iterations"
            value={iterations}
            onChange={setIterations}
            min={5}
            max={100}
          />
        </div>
        <Button
          disabled={
            benchmark.isPending ||
            models.data?.find((m) => m.id === modelId)?.status !== "available"
          }
          onClick={() => benchmark.mutate({ modelId, sampleId, iterations })}
        >
          {benchmark.isPending ? "Benchmarking…" : "Run CNN benchmark"}
        </Button>
        <Message error={benchmark.error} />
        {benchmark.data && (
          <>
            <Stats
              items={[
                ["Mean latency", `${numeric(benchmark.data.meanMs)} ms`],
                ["Median latency", `${numeric(benchmark.data.medianMs)} ms`],
                ["P95 latency", `${numeric(benchmark.data.p95Ms)} ms`],
                ["Throughput", `${numeric(benchmark.data.throughputFps)} frames/s`],
              ]}
            />
            <ScientificChart
              data={[
                {
                  x: benchmark.data.latenciesMs.map((_, i) => i + 1),
                  y: benchmark.data.latenciesMs,
                  type: "scatter",
                  mode: "lines+markers",
                  name: "CPU inference",
                },
              ]}
              xTitle="Iteration"
              yTitle="Latency (ms)"
            />
            <p className="mb-4 text-xs text-muted-foreground">{benchmark.data.scope}</p>
            <ExportButton onClick={() => downloadJson(benchmark.data, "satmod-benchmark.json")} />
          </>
        )}
      </Surface>
      <Surface className="p-5">
        <SectionHeading title="Model comparison" note="ARTIFACT AVAILABILITY" />
        <Field label="Compare CNN with">
          <select
            className="control"
            value={compareId}
            onChange={(e) => setCompareId(e.target.value)}
          >
            {models.data
              ?.filter((m) => m.id !== "1d-cnn")
              .map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
          </select>
        </Field>
        <p className="text-sm leading-6 text-muted-foreground">
          {other?.name ?? "Alternative model"}:{" "}
          {other?.reason ?? "No trained artifact is available."} Predictions, accuracy, and latency
          remain unavailable. To add a real model, register its artifact loader and input contract
          in the backend model registry and supply an independently measured evaluation.
        </p>
      </Surface>
    </div>
  );
}
