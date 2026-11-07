import { useEffect, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  satmodApi,
  SUPPORTED_MODULATIONS,
  type ClassificationResult,
  type AttributionResult,
  type Modulation,
} from "@/lib/satmod-api";
import { Button } from "@/components/ui/button";
import {
  Surface,
  SectionHeading,
  Field,
  NumberField,
  Stats,
  numeric,
  Message,
} from "@/components/workspace-ui";
import { ScientificChart } from "@/components/scientific-chart";

export function ClassificationInsights({
  result,
  onExplanation,
}: {
  result: ClassificationResult;
  onExplanation: (data: AttributionResult) => void;
}) {
  const [target, setTarget] = useState<Modulation>(result.predictedModulation);
  const [method, setMethod] = useState<"integrated-gradients" | "saliency">("integrated-gradients");
  const [steps, setSteps] = useState(64);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  const explanation = useMutation({
    mutationFn: async () => {
      controller.current = new AbortController();
      return satmodApi.explain(
        {
          samples: result.samples,
          sampleRate: result.sampleRate,
          source: result.source,
          windowStart: Number(result.preprocessing["windowStart"]),
          targetClass: target,
          method,
          steps,
        },
        controller.current.signal,
      );
    },
    onSuccess: onExplanation,
  });
  const trust = result.trust;
  return (
    <Surface className="p-4 space-y-4">
      <SectionHeading title="Trust and model explanation" note="ACTUAL CNN GRADIENTS" />
      <Stats
        items={[
          ["Input validity", trust?.inputValidity.status ?? "Unavailable"],
          ["Calibration", trust?.calibration.status ?? "Unavailable"],
          ["OOD detection", trust?.ood.status ?? "Unavailable"],
        ]}
      />
      <p className="text-xs leading-6 text-muted-foreground">
        {trust?.interpretation} {trust?.inputValidity.channelOrder}
      </p>
      <p className="text-xs leading-6 text-muted-foreground">
        {trust?.calibration.reason} {trust?.ood.reason}
      </p>
      <div className="grid gap-4 md:grid-cols-3">
        <Field label="Attribution method">
          <select
            className="control"
            value={method}
            disabled={explanation.isPending}
            onChange={(e) => {
              setMethod(e.target.value as typeof method);
              explanation.reset();
            }}
          >
            <option value="integrated-gradients">Integrated Gradients</option>
            <option value="saliency">Logit saliency</option>
          </select>
        </Field>
        <Field label="Explain target class">
          <select
            className="control"
            value={target}
            disabled={explanation.isPending}
            onChange={(e) => {
              setTarget(e.target.value as Modulation);
              explanation.reset();
            }}
          >
            {SUPPORTED_MODULATIONS.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </Field>
        {method === "integrated-gradients" && (
          <NumberField
            label="Integration steps (adaptive up to 256)"
            min={16}
            max={256}
            value={steps}
            onChange={setSteps}
          />
        )}
      </div>
      <div className="flex gap-3">
        <Button
          disabled={explanation.isPending || !Number.isInteger(steps) || steps < 16 || steps > 256}
          onClick={() => explanation.mutate()}
        >
          {explanation.isPending ? "Computing model attributions…" : "Explain classification"}
        </Button>
        {explanation.isPending && (
          <Button variant="outline" onClick={() => controller.current?.abort()}>
            Cancel explanation
          </Button>
        )}
      </div>
      <Message error={explanation.error} />
      {explanation.data && (
        <>
          <Stats
            items={[
              ["Target class", explanation.data.targetClass],
              ["Method", explanation.data.method],
              ["Explanation time", `${numeric(explanation.data.processingMs)} ms`],
              ["Completeness residual", numeric(explanation.data.completenessError)],
            ]}
          />
          <ScientificChart
            data={(["i", "q"] as const).map((channel) => ({
              x: explanation.data.samples.map((p) => p.sample),
              y: explanation.data.samples.map((p) => p[channel]),
              type: "scatter",
              mode: "lines",
              name: `${channel.toUpperCase()} contribution`,
            }))}
            xTitle="Capture sample index"
            yTitle={
              explanation.data.method === "integrated-gradients"
                ? "Signed class-logit contribution"
                : "Logit derivative / raw sample unit"
            }
          />
          <ScientificChart
            data={[
              {
                x: explanation.data.samples.map((p) => p.sample),
                y: explanation.data.samples.map((p) => p.importance),
                type: "bar",
                name: "Absolute I+Q attribution",
              },
            ]}
            xTitle="Capture sample index"
            yTitle="Absolute model importance"
            height={210}
          />
          <p className="text-xs leading-6 text-muted-foreground">
            {explanation.data.baseline.description}. Target: {explanation.data.targetQuantity}.{" "}
            {explanation.data.integrationSteps} integration steps used. Export results includes the
            latest computed explanation and method metadata.
          </p>
          {explanation.data.limitations.map((note) => (
            <p key={note} className="text-xs leading-5 text-muted-foreground">
              {note}
            </p>
          ))}
        </>
      )}
    </Surface>
  );
}
