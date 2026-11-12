import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { operationsApi } from "@/lib/operations-api";
import { researchApi, researchDownload, type ManagedComparisonReport } from "@/lib/research-api";
import { Button } from "@/components/ui/button";
import {
  Field,
  Message,
  NumberField,
  Stats,
  Surface,
  numeric,
  percent,
} from "@/components/workspace-ui";
import { ScientificChart } from "@/components/scientific-chart";

export function ManagedComparison() {
  const cache = useQueryClient();
  const [first, setFirst] = useState("");
  const [second, setSecond] = useState("");
  const [startA, setStartA] = useState(0);
  const [startB, setStartB] = useState(0);
  const [length, setLength] = useState(128);
  const [analysisA, setAnalysisA] = useState("");
  const [analysisB, setAnalysisB] = useState("");
  const [search, setSearch] = useState("");
  const [offset, setOffset] = useState(0);
  const [report, setReport] = useState<ManagedComparisonReport | null>(null);
  const captures = useQuery({
    queryKey: ["comparison-captures", search, offset],
    queryFn: () => operationsApi.captures(search, offset),
  });
  const saved = useQuery({
    queryKey: ["managed-comparisons"],
    queryFn: () => researchApi.comparisons(),
  });
  const compare = useMutation({
    mutationFn: () =>
      researchApi.compareCaptures({
        first: { captureId: first, start: startA, length },
        second: { captureId: second, start: startB, length },
        ...(analysisA ? { firstAnalysisId: analysisA } : {}),
        ...(analysisB ? { secondAnalysisId: analysisB } : {}),
      }),
    onSuccess: (value) => {
      setReport(value);
      void cache.invalidateQueries({ queryKey: ["managed-comparisons"] });
    },
  });
  const action = useMutation({ mutationFn: (fn: () => Promise<unknown>) => fn() });
  return (
    <Surface className="p-4 space-y-4">
      <details>
        <summary className="cursor-pointer font-semibold text-primary">
          Managed recording comparison workstation
        </summary>
        <div className="space-y-4 mt-4">
          <p className="text-xs text-muted-foreground">
            Compare exact stored windows. No sample alignment, amplitude normalization or resampling
            is inferred. Reports retain the capture/window hashes and actual model outputs.
          </p>
          <Field label="Search comparison recordings">
            <input
              className="control"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setOffset(0);
              }}
            />
          </Field>
          <div className="grid gap-3 md:grid-cols-2">
            <Field label="Comparison recording A">
              <select
                className="control"
                value={first}
                onChange={(e) => {
                  setFirst(e.target.value);
                  setStartA(0);
                }}
              >
                <option value="">Select recording A</option>
                {captures.data?.items.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.source} · {c.sampleCount.toLocaleString()} samples
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Comparison recording B">
              <select
                className="control"
                value={second}
                onChange={(e) => {
                  setSecond(e.target.value);
                  setStartB(0);
                }}
              >
                <option value="">Select recording B</option>
                {captures.data?.items.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.source} · {c.sampleCount.toLocaleString()} samples
                  </option>
                ))}
              </select>
            </Field>
            <NumberField
              label="Recording A sample offset"
              value={startA}
              onChange={setStartA}
              min={0}
            />
            <NumberField
              label="Recording B sample offset"
              value={startB}
              onChange={setStartB}
              min={0}
            />
            <NumberField
              label="Comparison window length"
              value={length}
              onChange={setLength}
              min={16}
              max={65536}
            />
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              disabled={!offset}
              onClick={() => setOffset(Math.max(0, offset - 50))}
            >
              Previous recording page
            </Button>
            <Button
              variant="outline"
              disabled={offset + 50 >= (captures.data?.total ?? 0)}
              onClick={() => setOffset(offset + 50)}
            >
              Next recording page
            </Button>
          </div>
          <details>
            <summary className="cursor-pointer text-xs">
              Reuse saved predictions / attribution (optional)
            </summary>
            <div className="grid gap-3 md:grid-cols-2 mt-3">
              <Field label="Recording A analysis ID">
                <input
                  className="control"
                  value={analysisA}
                  onChange={(e) => setAnalysisA(e.target.value)}
                  placeholder="Exact matching historical analysis ID"
                />
              </Field>
              <Field label="Recording B analysis ID">
                <input
                  className="control"
                  value={analysisB}
                  onChange={(e) => setAnalysisB(e.target.value)}
                  placeholder="Exact matching historical analysis ID"
                />
              </Field>
            </div>
            <p className="text-xs">
              Historical input, model and preprocessing hashes must match. Attribution differences
              require the same method/target and compatible sample clocks.
            </p>
          </details>
          <Button
            disabled={
              !first ||
              !second ||
              compare.isPending ||
              ![startA, startB, length].every(Number.isInteger)
            }
            onClick={() => compare.mutate()}
          >
            {compare.isPending ? "Computing stored-window comparison…" : "Compare managed windows"}
          </Button>
          <Message
            error={captures.error ?? compare.error ?? saved.error ?? action.error}
            loading={compare.isPending}
          />
          {report && (
            <div className="space-y-4" role="region" aria-label="Stored comparison result">
              <p className="text-xs break-all">
                Comparison {report.id} · {report.createdAt}
              </p>
              <Stats
                items={[
                  [
                    "A sample clock",
                    report.sampleRates.first === null
                      ? "Unknown"
                      : `${report.sampleRates.first} Hz`,
                  ],
                  [
                    "B sample clock",
                    report.sampleRates.second === null
                      ? "Unknown"
                      : `${report.sampleRates.second} Hz`,
                  ],
                  [
                    "Axis policy",
                    report.normalizedAxes
                      ? "Normalized sample/frequency axes"
                      : "Physical time/frequency axes",
                  ],
                  [
                    "Sample-rate comparison",
                    report.sampleRates.first === null || report.sampleRates.second === null
                      ? "At least one rate unknown"
                      : report.ratesEqual ? "Equal supplied rates" : "Different supplied rates",
                  ],
                  ["A − B relative power", `${numeric(report.powerDifferenceDb)} dB`],
                  ["A − B raw confidence", numeric(report.confidenceDifference)],
                  ["Unaligned sample MSE", numeric(report.sampleMse)],
                ]}
              />
              <p className="text-xs text-muted-foreground">
                Sample-rate metadata describes sampling intervals; clock synchronization and sample
                alignment are unverified.
              </p>
              <div className="grid gap-4 lg:grid-cols-2">
                {(["first", "second"] as const).map((key, index) => (
                  <div className="border border-border p-3 space-y-3" key={key}>
                    <h3 className="font-semibold">
                      Recording {index === 0 ? "A" : "B"}: {report[key].metadata.source}
                    </h3>
                    <p className="text-xs break-all">
                      Samples {report[key].start}–{report[key].start + report[key].length - 1} ·
                      input SHA-256 {report[key].inputHash}
                    </p>
                    <Stats
                      items={[
                        [
                          "Predicted class",
                          report[key].prediction.predictedModulation ?? "Unavailable",
                        ],
                        ["Raw softmax", percent(report[key].prediction.confidence)],
                        ["Power", numeric(report[key].analysis.signalPower)],
                        ["RMS", numeric(report[key].analysis.rmsAmplitude)],
                      ]}
                    />
                    {report[key].prediction.reason && <p>{report[key].prediction.reason}</p>}
                    <ScientificChart
                      title={`Recording ${index === 0 ? "A" : "B"} I/Q waveform`}
                      data={(["i", "q"] as const).map((channel) => ({
                        type: "scatter",
                        mode: "lines",
                        name: channel.toUpperCase(),
                        x: report.plots[key].waveform.map((p) => p.x),
                        y: report.plots[key].waveform.map((p) => p[channel]),
                      }))}
                      xTitle={report.plots.timeUnit}
                      yTitle="Relative sample amplitude"
                    />
                    <ScientificChart
                      title={`Recording ${index === 0 ? "A" : "B"} constellation`}
                      data={[
                        {
                          type: "scatter",
                          mode: "markers",
                          x: report.plots[key].constellation.map((p) => p.i),
                          y: report.plots[key].constellation.map((p) => p.q),
                          marker: { size: 4 },
                        },
                      ]}
                      xTitle="In-phase I"
                      yTitle="Quadrature Q"
                      equal
                    />
                  </div>
                ))}
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                <ScientificChart
                  title="Matched-axis spectral comparison"
                  data={(["first", "second"] as const).map((key, i) => ({
                    type: "scatter",
                    mode: "lines",
                    name: i === 0 ? "A" : "B",
                    x: report.plots[key].spectrum.map((p) => p.frequency),
                    y: report.plots[key].spectrum.map((p) => p.psd),
                  }))}
                  xTitle={`Frequency (${report.plots.frequencyUnit})`}
                  yTitle={report.plots.psdUnit}
                />
                <ScientificChart
                  title="Independent phase trajectories"
                  data={(["first", "second"] as const).map((key, i) => ({
                    type: "scatter",
                    mode: "lines",
                    name: i === 0 ? "A" : "B",
                    x: report.plots[key].phase.map((p) => p.x),
                    y: report.plots[key].phase.map((p) => p.phaseRad),
                  }))}
                  xTitle={report.plots.timeUnit}
                  yTitle="Unwrapped phase (rad)"
                />
                <ScientificChart
                  title="Common-bin amplitude distributions"
                  data={(["first", "second"] as const).map((key, i) => ({
                    type: "scatter",
                    mode: "lines",
                    name: i === 0 ? "A" : "B",
                    x: report.plots.amplitudeDistribution[key].map((p) => p.value),
                    y: report.plots.amplitudeDistribution[key].map((p) => p.fraction),
                  }))}
                  xTitle="Relative amplitude"
                  yTitle="Fraction of samples"
                  description={report.plots.amplitudeDistribution.note}
                />
              </div>
              {report.attributionDifferences.status === "available" ? (
                report.attributionDifferences.comparisons.map((value) => (
                  <ScientificChart
                    key={`${value.method}:${value.targetClass}`}
                    title={`A − B ${value.method}: ${value.targetClass}`}
                    data={(["i", "q"] as const).map((channel) => ({
                      type: "scatter",
                      mode: "lines",
                      name: channel.toUpperCase(),
                      x: value.samples.map((p) => p.sample),
                      y: value.samples.map((p) => p[channel]),
                    }))}
                    xTitle="Relative model-window sample index (not physical alignment)"
                    yTitle={value.quantityUnit}
                  />
                ))
              ) : (
                <p className="text-xs">
                  Attribution difference unavailable: {report.attributionDifferences.reason}
                </p>
              )}
              <p className="text-xs">{report.attributionDifferences.note}</p>
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="outline"
                  onClick={() =>
                    action.mutate(() =>
                      researchDownload(
                        `/comparisons/${report.id}/export?format=json`,
                        "satmod-managed-comparison.json",
                      ),
                    )
                  }
                >
                  Export comparison JSON
                </Button>
                <Button
                  variant="outline"
                  onClick={() =>
                    action.mutate(() =>
                      researchDownload(
                        `/comparisons/${report.id}/export?format=pdf`,
                        "satmod-managed-comparison.pdf",
                      ),
                    )
                  }
                >
                  Export comparison PDF
                </Button>
              </div>
              {report.limitations.map((note) => (
                <p className="text-xs text-muted-foreground" key={note}>
                  {note}
                </p>
              ))}
            </div>
          )}
          <Field label="Inspect saved comparison">
            <select
              className="control"
              value={report?.id ?? ""}
              onChange={(e) => {
                const comparisonId = e.currentTarget.value;
                if (comparisonId)
                  action.mutate(async () =>
                    setReport(await researchApi.comparison(comparisonId)),
                  );
              }}
            >
              <option value="">Select a stored comparison</option>
              {saved.data?.items.map((value) => (
                <option key={value.id} value={value.id}>
                  {value.createdAt} · {value.source ?? value.id}
                </option>
              ))}
            </select>
          </Field>
          {report && (
            <Button
              variant="outline"
              onClick={() =>
                action.mutate(async () => {
                  await researchApi.deleteComparison(report.id);
                  setReport(null);
                  await cache.invalidateQueries({ queryKey: ["managed-comparisons"] });
                })
              }
            >
              Delete comparison report
            </Button>
          )}
        </div>
      </details>
    </Surface>
  );
}
