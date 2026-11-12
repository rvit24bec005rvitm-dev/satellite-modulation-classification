import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { researchApi } from "@/lib/research-api";
import { operationDownload } from "@/lib/operations-api";
import {
  downloadJson,
  SUPPORTED_MODULATIONS,
  type IQSample,
  type Modulation,
} from "@/lib/satmod-api";
import { Button } from "@/components/ui/button";
import { Field, Message, NumberField, Stats, numeric, percent } from "@/components/workspace-ui";
import { ScientificChart } from "@/components/scientific-chart";

export function SavedExplanations({
  analysisId,
  originalSamples,
  predictedClass,
}: {
  analysisId: string;
  originalSamples?: IQSample[];
  predictedClass?: Modulation;
}) {
  const cache = useQueryClient();
  const [target, setTarget] = useState<Modulation | null>(predictedClass ?? null);
  const [steps, setSteps] = useState(64);
  const [selected, setSelected] = useState("");
  const report = useQuery({
    queryKey: ["saved-explanations", analysisId],
    queryFn: () => researchApi.explanations(analysisId),
    retry: false,
  });
  const targetClass = target ?? report.data?.prediction.predictedModulation ?? "BPSK";
  const save = useMutation({
    mutationFn: () =>
      researchApi.saveExplanation(analysisId, {
        method: "both",
        targetClass,
        steps,
        ...(originalSamples ? { samples: originalSamples } : {}),
      }),
    onSuccess: (result) => {
      cache.setQueryData(["saved-explanations", analysisId], result);
      setSelected("");
    },
  });
  const exportPdf = useMutation({
    mutationFn: () =>
      operationDownload(
        `/history/${analysisId}/report?format=pdf`,
        "satmod-explanation-report.pdf",
      ),
  });
  const explanations = report.data?.explanations ?? [];
  const explanation =
    explanations.find((e) => `${e.method}:${e.targetClass}:${e.createdAt}` === selected) ??
    explanations.at(-1);
  const methodComparison = report.data?.methodComparisons?.find(
    (value) => value.targetClass === explanation?.targetClass,
  ) ?? (report.data?.methodComparison.targetClass === explanation?.targetClass ? report.data?.methodComparison : undefined);
  return (
    <div
      className="space-y-4 border-t border-border pt-4"
      role="region"
      aria-label="Persistent explanation workspace"
    >
      <h3 className="font-semibold text-primary">Persistent explanation and method comparison</h3>
      <p className="text-xs text-muted-foreground">
        Saved with the analysis, model hash and original input hash. Managed recordings can be
        reconstructed; legacy analyses require their original samples. No signal or attribution
        values are inferred from a class label.
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Saved explanation target class">
          <select
            className="control"
            value={targetClass}
            onChange={(e) => setTarget(e.target.value as Modulation)}
            disabled={save.isPending}
          >
            {SUPPORTED_MODULATIONS.map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </Field>
        <NumberField
          label="Saved explanation integration steps"
          min={16}
          max={256}
          value={steps}
          onChange={setSteps}
        />
      </div>
      <Button
        disabled={save.isPending || !Number.isInteger(steps) || steps < 16 || steps > 256}
        onClick={() => save.mutate()}
      >
        {save.isPending
          ? "Computing and saving both methods…"
          : "Compute and save both explanation methods"}
      </Button>
      <Message
        error={save.error ?? exportPdf.error ?? report.error}
        loading={report.isPending || save.isPending}
      />
      {report.data && !explanations.length && (
        <p className="text-xs">No explanations saved for this analysis yet.</p>
      )}
      {explanation && report.data && (
        <>
          <Field label="Saved explanation view">
            <select
              className="control"
              value={`${explanation.method}:${explanation.targetClass}:${explanation.createdAt}`}
              onChange={(e) => setSelected(e.target.value)}
            >
              {explanations.map((e) => (
                <option
                  key={`${e.method}:${e.targetClass}:${e.createdAt}`}
                  value={`${e.method}:${e.targetClass}:${e.createdAt}`}
                >
                  {e.targetClass} · {e.method} · {e.createdAt}
                </option>
              ))}
            </select>
          </Field>
          <Stats
            items={[
              ["Predicted modulation", report.data.prediction.predictedModulation],
              ["Raw softmax", percent(report.data.prediction.confidence)],
              ["Target class", explanation.targetClass],
              ["Method", explanation.method],
              ["Completeness residual", numeric(explanation.completenessError)],
              ["Logit change from baseline", numeric(explanation.logitDifference)],
            ]}
          />
          <p className="text-xs break-all">
            Model {report.data.modelVersion} · SHA-256 {report.data.modelHash}
          </p>
          <p className="text-xs">
            {explanation.diagnostics.quantityDescription} Baseline:{" "}
            {explanation.baseline.description}. {explanation.integrationSteps} integration steps
            used.
          </p>
          <ScientificChart
            title="Signed I/Q class-logit attribution"
            data={(["i", "q"] as const).map((channel) => ({
              type: "scatter",
              mode: "lines",
              name: channel.toUpperCase(),
              x: explanation.samples.map((p) => p.sample),
              y: explanation.samples.map((p) => p[channel]),
            }))}
            xTitle="Original analysis sample index"
            yTitle={explanation.quantityUnit}
          />
          <div className="overflow-auto">
            <table className="w-full text-xs">
              <caption className="text-left py-2">
                Signed channel summaries · {explanation.quantityUnit}
              </caption>
              <thead>
                <tr>
                  <th>Channel</th>
                  <th>Positive</th>
                  <th>Negative</th>
                  <th>Absolute sum</th>
                </tr>
              </thead>
              <tbody>
                {(["i", "q"] as const).map((channel) => (
                  <tr key={channel}>
                    <th scope="row">{channel.toUpperCase()}</th>
                    <td>{numeric(explanation.diagnostics.channels[channel].positiveSum)}</td>
                    <td>{numeric(explanation.diagnostics.channels[channel].negativeSum)}</td>
                    <td>{numeric(explanation.diagnostics.channels[channel].absoluteSum)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs">{explanation.diagnostics.regionRule}</p>
          <ul className="text-xs space-y-1">
            {explanation.diagnostics.importantRegions.map((region) => (
              <li key={region.startSample}>
                Samples {region.startSample}–{region.endSample - 1}:{" "}
                {percent(region.shareOfAbsoluteImportance)} of absolute model importance
              </li>
            ))}
          </ul>
          <p className="text-xs">
            Method rank correlation for {explanation.targetClass}: {numeric(methodComparison?.spearmanRho)}.{" "}
            {methodComparison?.note ?? "Store both methods for this target class to compare sample rankings."}
          </p>
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              onClick={() => downloadJson(report.data, "satmod-full-explanations.json")}
            >
              Export full-resolution explanations JSON
            </Button>
            <Button
              variant="outline"
              disabled={exportPdf.isPending}
              onClick={() => exportPdf.mutate()}
            >
              Download saved explanation PDF
            </Button>
          </div>
          {explanation.limitations.map((note) => (
            <p className="text-xs text-muted-foreground" key={note}>
              {note}
            </p>
          ))}
        </>
      )}
    </div>
  );
}
