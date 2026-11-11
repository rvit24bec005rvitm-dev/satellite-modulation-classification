import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { operationsApi, operationDownload } from "@/lib/operations-api";
import { Surface, Field, Message } from "@/components/workspace-ui";
import { Button } from "@/components/ui/button";
export function ExternalEvaluation() {
  const cache = useQueryClient();
  const [text, setText] = useState("");
  const [jobId, setJobId] = useState("");
  const datasets = useQuery({
    queryKey: ["external-datasets"],
    queryFn: () => operationsApi.datasets(),
  });
  const jobs = useQuery({
    queryKey: ["workspace-jobs"],
    queryFn: () => operationsApi.jobs(),
    refetchInterval: 1500,
  });
  const job = jobs.data?.items.find((j) => j.id === jobId);
  const result = useQuery({
    queryKey: ["external-eval-result", jobId],
    queryFn: () => operationsApi.result(jobId),
    enabled: job?.state === "completed",
  });
  const register = useMutation({
    mutationFn: () => operationsApi.register(JSON.parse(text)),
    onSuccess: () => void cache.invalidateQueries({ queryKey: ["external-datasets"] }),
  });
  const action = useMutation({ mutationFn: (fn: () => Promise<unknown>) => fn() });
  return (
    <Surface className="p-4 space-y-3">
      <details>
        <summary className="cursor-pointer font-semibold text-primary">
          External labeled evaluation and dataset provenance
        </summary>
        <div className="space-y-3 mt-4">
          <p className="text-xs text-muted-foreground">
            Import recordings in Classifier/Intelligence first. Labels and test membership must come
            from independent acquisition records, never CNN guesses. Session, exact-duplicate and
            recording-overlap checks reject split contamination. No calibration or model promotion
            occurs.
          </p>
          <Field label="Labeled dataset manifest JSON">
            <textarea
              className="control min-h-40 font-mono text-xs"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder={
                '{"name":"External test","source":"Acquisition record","license":"Usage rights","entries":[{"captureId":"UUID from capture browser","start":0,"label":"BPSK","split":"test"}]}'
              }
            />
          </Field>
          <Field label="Load labeled manifest file">
            <input
              className="control"
              type="file"
              accept=".json"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file)
                  action.mutate(async () => {
                    if (file.size > 1048576) throw Error("Manifest exceeds 1 MiB");
                    setText(await file.text());
                  });
              }}
            />
          </Field>
          <Button disabled={!text || register.isPending} onClick={() => register.mutate()}>
            Register labeled dataset
          </Button>
          <Message
            error={register.error ?? datasets.error ?? action.error ?? jobs.error ?? result.error}
          />
          <div className="space-y-3">
            {datasets.data?.items.map((d) => (
              <div key={d.id} className="border border-border p-3 text-xs space-y-2">
                <p>
                  {d.name} · {d.entries.length} labeled windows · {d.source}
                </p>
                <p className="break-all">SHA-256 {d.hash}</p>
                <p>{d.note}</p>
                <Button
                  onClick={() =>
                    action.mutate(async () => {
                      const j = await operationsApi.evaluate(d.id);
                      setJobId(j.id);
                      await cache.invalidateQueries({ queryKey: ["workspace-jobs"] });
                    })
                  }
                >
                  Evaluate held-out windows
                </Button>
                <Button
                  variant="outline"
                  onClick={() =>
                    action.mutate(async () => {
                      await operationsApi.deleteDataset(d.id);
                      await cache.invalidateQueries({ queryKey: ["external-datasets"] });
                    })
                  }
                >
                  Delete dataset registration
                </Button>
              </div>
            ))}
          </div>
          {job && (
            <p role="status">
              External evaluation: {job.state} · {Math.round(job.progress * 100)}% {job.error}
            </p>
          )}
          {result.data && (
            <>
              <Button
                onClick={() =>
                  action.mutate(() =>
                    operationDownload(
                      `/jobs/${jobId}/export?format=json`,
                      "satmod-external-evaluation.json",
                    ),
                  )
                }
              >
                Export external evaluation JSON
              </Button>
              <pre className="max-h-96 overflow-auto text-xs">
                {JSON.stringify(result.data.metrics, null, 2)}
              </pre>
              <p className="text-xs">
                Raw softmax calibration diagnostics apply only to these labeled test observations.
                SNR bins exist only where importers supply independently measured SNR metadata.
              </p>
            </>
          )}
        </div>
      </details>
    </Surface>
  );
}
