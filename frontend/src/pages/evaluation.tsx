import { useState } from "react";
import { useEvaluation } from "@/hooks/use-satmod";
import { downloadJson, downloadBlob } from "@/lib/satmod-api";
import {
  Surface,
  SectionHeading,
  Message,
  Stats,
  percent,
  ExportButton,
} from "@/components/workspace-ui";
import { ScientificChart } from "@/components/scientific-chart";
export function EvaluationPage() {
  const query = useEvaluation();
  const [normalized, setNormalized] = useState(false);
  const data = query.data;
  return (
    <div className="space-y-5">
      <Message loading={query.isPending} error={query.error} />
      {data && (
        <>
          <div className="flex flex-wrap gap-3">
            <ExportButton onClick={() => downloadJson(data, "satmod-evaluation.json")} />
            <ExportButton
              onClick={() =>
                downloadBlob(
                  new Blob(
                    [
                      "SNR_dB,Accuracy\n" +
                        data.snrAccuracy.map((r) => `${r.snrDb},${r.accuracy}`).join("\n"),
                    ],
                    { type: "text/csv" },
                  ),
                  "satmod-snr.csv",
                )
              }
            >
              Export SNR CSV
            </ExportButton>
          </div>
          <Stats
            items={[
              ["Overall accuracy", percent(data.metrics.accuracy)],
              ["Macro precision", percent(data.metrics.precision)],
              ["Macro recall", percent(data.metrics.recall)],
              ["Macro F1-score", percent(data.metrics.f1)],
            ]}
          />
          <div className="grid gap-4 xl:grid-cols-2">
            <Surface className="p-4">
              <div className="flex flex-wrap justify-between gap-3">
                <SectionHeading
                  title="Confusion matrix"
                  note={`${data.metrics.samples} OBSERVATIONS`}
                />
                <label className="text-xs">
                  <input
                    type="checkbox"
                    checked={normalized}
                    onChange={(e) => setNormalized(e.target.checked)}
                  />{" "}
                  Row percentages
                </label>
              </div>
              <ScientificChart
                data={[
                  {
                    type: "heatmap",
                    x: data.classes,
                    y: data.classes,
                    z: normalized
                      ? data.confusionMatrix.map((row) =>
                          row.map((v) => (100 * v) / row.reduce((a, b) => a + b, 0)),
                        )
                      : data.confusionMatrix,
                    colorscale: "Viridis",
                    texttemplate: normalized ? "%{z:.1f}%" : "%{z}",
                    hovertemplate: "Actual %{y}<br>Predicted %{x}<br>%{z}<extra></extra>",
                  },
                ]}
                xTitle="Predicted modulation"
                yTitle="Actual modulation"
                height={330}
              />
            </Surface>
            <Surface className="p-4">
              <SectionHeading title="Accuracy versus SNR" note="STORED CSV" />
              <ScientificChart
                data={["accuracy", ...data.classes].map((key) => ({
                  x: data.snrAccuracy.map((p) => p.snrDb),
                  y: data.snrAccuracy.map((p) => 100 * Number(p[key as keyof typeof p])),
                  type: "scatter",
                  mode: "lines+markers",
                  name: key === "accuracy" ? "Overall" : key,
                }))}
                xTitle="SNR (dB)"
                yTitle="Accuracy (%)"
                height={330}
              />
            </Surface>
            <Surface className="p-4">
              <SectionHeading title="Per-class performance" note="RECOMPUTED FROM COUNTS" />
              <ScientificChart
                data={(["precision", "recall", "f1"] as const).map((key) => ({
                  x: data.perClass.map((p) => p.className),
                  y: data.perClass.map((p) => p[key] * 100),
                  type: "bar",
                  name: key,
                }))}
                xTitle="Class"
                yTitle="Score (%)"
              />
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr>
                      {["Class", "Precision", "Recall", "F1", "Support"].map((v) => (
                        <th key={v} className="p-2">
                          {v}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.perClass.map((p) => (
                      <tr key={p.className} className="border-t border-border">
                        <td className="p-2">{p.className}</td>
                        <td className="p-2">{percent(p.precision)}</td>
                        <td className="p-2">{percent(p.recall)}</td>
                        <td className="p-2">{percent(p.f1)}</td>
                        <td className="p-2">{p.support}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Surface>
            <Surface className="p-4">
              <SectionHeading title="Misclassification analysis" note="ACTUAL → PREDICTED" />
              {data.misclassifications.map((r) => (
                <div
                  key={r.actual + r.predicted}
                  className="flex justify-between border-b border-border py-3 text-xs"
                >
                  <span>
                    {r.actual} → {r.predicted}
                  </span>
                  <span className="font-mono text-primary">{r.count}</span>
                </div>
              ))}
              <p className="mt-4 text-xs leading-6 text-muted-foreground">
                The largest error group is {data.misclassifications[0]?.actual} classified as{" "}
                {data.misclassifications[0]?.predicted}. These are stored aggregate counts;
                individual error frames are not provided.
              </p>
            </Surface>
          </div>
          <Surface className="p-4">
            <SectionHeading title="Evaluation provenance" note={data.source} />
            {data.notes.map((n) => (
              <p key={n} className="mt-2 text-xs leading-6 text-muted-foreground">
                {n}
              </p>
            ))}
          </Surface>
        </>
      )}
    </div>
  );
}
