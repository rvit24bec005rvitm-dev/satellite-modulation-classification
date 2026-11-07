import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { satmodApi, downloadJson } from "@/lib/satmod-api";
import { Button } from "@/components/ui/button";
import {
  Surface,
  SectionHeading,
  Message,
  Stats,
  percent,
  numeric,
  ExportButton,
} from "@/components/workspace-ui";
import { ScientificChart } from "@/components/scientific-chart";
export function ModelDiagnostics() {
  const [enabled, setEnabled] = useState(false);
  const query = useQuery({
    queryKey: ["model-diagnostics"],
    queryFn: ({ signal }) => satmodApi.diagnostics(signal),
    enabled,
    retry: 1,
  });
  const data = query.data;
  const canonical = data?.comparisons["training-zscore-only"];
  const bins = canonical?.calibration.reliability.filter((b) => b.count > 0) ?? [];
  return (
    <Surface className="p-4 space-y-4">
      <SectionHeading
        title="Recomputed demo diagnostics"
        note="DESCRIPTIVE · NOT AN INDEPENDENT TEST"
      />
      <p className="text-xs leading-6 text-muted-foreground">
        Run the real CNN on the 300 supplied demo frames to inspect preprocessing sensitivity, SNR
        degradation and uncalibrated confidence. Their train/test provenance is unknown. These
        frames are never used to tune a calibration temperature or OOD threshold.
      </p>
      <Button disabled={enabled && query.isPending} onClick={() => setEnabled(true)}>
        {enabled && query.isPending ? "Computing diagnostics…" : "Load real demo diagnostics"}
      </Button>
      <Message error={query.error} />
      {data && canonical && (
        <>
          <p className="text-xs text-signal-warning">
            {data.scope} {data.warning}
          </p>
          <Stats
            items={[
              ["Demo accuracy", percent(canonical.accuracy)],
              ["Expected calibration error", numeric(canonical.calibration.ece)],
              ["Negative log likelihood", numeric(canonical.calibration.nll)],
              ["Multiclass Brier score", numeric(canonical.calibration.brier)],
            ]}
          />
          <div className="grid gap-4 lg:grid-cols-2">
            <ScientificChart
              data={[
                {
                  x: bins.map((b) => b.confidence!),
                  y: bins.map((b) => b.accuracy!),
                  type: "scatter",
                  mode: "lines+markers",
                  name: "Observed demo bins",
                },
                {
                  x: [0, 1],
                  y: [0, 1],
                  type: "scatter",
                  mode: "lines",
                  name: "Perfect calibration reference",
                },
              ]}
              xTitle="Mean uncalibrated confidence"
              yTitle="Observed bin accuracy"
            />
            <ScientificChart
              data={[
                {
                  x: Object.keys(data.comparisons),
                  y: Object.values(data.comparisons).map((m) => 100 * m.accuracy),
                  type: "bar",
                },
              ]}
              xTitle="Controlled preprocessing path"
              yTitle="Demo accuracy (%)"
            />
          </div>
          <ScientificChart
            data={["BPSK", "QPSK", "16-QAM"].map((label) => ({
              x: canonical.snrAccuracy.map((p) => p.snrDb),
              y: canonical.snrAccuracy.map((p) => 100 * (p.perClass[label] ?? NaN)),
              type: "scatter",
              mode: "lines+markers",
              name: label,
            }))}
            xTitle="Demo SNR metadata (dB)"
            yTitle="Recomputed per-class recall (%)"
          />
          <ExportButton onClick={() => downloadJson(data, "satmod-demo-diagnostics.json")} />
        </>
      )}
    </Surface>
  );
}
