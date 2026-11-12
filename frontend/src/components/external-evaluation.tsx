import { useEffect, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { operationsApi, operationDownload } from "@/lib/operations-api";
import { Surface, Field, Message } from "@/components/workspace-ui";
import { Button } from "@/components/ui/button";
import { researchApi } from "@/lib/research-api";
import { EvaluationMetricsPanel } from "@/components/matched-evaluations";
import { TrainingDatasetExport } from "@/components/training-dataset-export";
import { numeric } from "@/components/workspace-ui";
export function ExternalEvaluation() {
  const cache = useQueryClient();
  const [text, setText] = useState("");
  const [jobId, setJobId] = useState("");
  const [offset, setOffset] = useState(0);
  const [search, setSearch] = useState("");
  const [modelId, setModelId] = useState("1d-cnn");
  const models = useQuery({ queryKey: ["research-models"], queryFn: () => researchApi.models() });
  const model = models.data?.find((item) => item.id === modelId);
  const datasets = useQuery({
    queryKey: ["external-datasets", offset, search],
    queryFn: () => operationsApi.datasets(offset, search),
  });
  const jobStatus = useQuery({
    queryKey: ["workspace-job", jobId],
    queryFn: () => operationsApi.job(jobId),
    enabled: !!jobId,
    refetchInterval: (query) =>
      ["completed", "failed", "cancelled"].includes(query.state.data?.state ?? "") ? false : 1000,
  });
  const job = jobStatus.data;
  useEffect(() => {
    if (job?.kind === "dataset-import" && job.state === "completed") {
      setOffset(0);
      setSearch("");
      void cache.invalidateQueries({ queryKey: ["external-datasets"] });
    }
  }, [job?.id, job?.kind, job?.state, cache]);
  useEffect(() => {
    if (job?.kind === "evaluation" && job.state === "completed")
      void cache.invalidateQueries({ queryKey: ["external-evaluations"] });
  }, [job?.id, job?.kind, job?.state, cache]);
  const result = useQuery({
    queryKey: ["external-eval-result", jobId],
    queryFn: () => researchApi.evaluationResult(jobId),
    enabled: job?.state === "completed" && job.kind === "evaluation",
  });
  const register = useMutation({
    mutationFn: () => operationsApi.registerJob(JSON.parse(text)),
    onSuccess: (j) => {
      setJobId(j.id);
      void cache.invalidateQueries({ queryKey: ["workspace-jobs"] });
    },
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
            error={
              register.error ??
              datasets.error ??
              action.error ??
              jobStatus.error ??
              result.error ??
              models.error
            }
          />
          <Field label="External evaluation model">
            <select
              className="control"
              aria-label="External evaluation model"
              value={modelId}
              onChange={(event) => setModelId(event.target.value)}
            >
              {!models.data && (
                <option value="1d-cnn">Loading production and experimental models…</option>
              )}
              {models.data?.map((item) => (
                <option key={item.id} value={item.id} disabled={!item.available}>
                  {item.name} · {item.version} · {item.status}
                </option>
              ))}
            </select>
          </Field>
          {model && (
            <div className="space-y-2 text-xs">
              <p>
                Supported classes: {model.classLabels.join(", ")} · {model.lifecycle}
              </p>
              {model.available ? (
                <>
                  <p className="break-all font-mono">
                    Selected model SHA-256 {model.artifactSha256}
                  </p>
                  <details>
                    <summary className="cursor-pointer text-primary">
                      Selected model preprocessing and runtime
                    </summary>
                    <pre className="max-h-56 overflow-auto whitespace-pre-wrap break-words">
                      {JSON.stringify(
                        {
                          version: model.version,
                          backend: model.backend,
                          preprocessing: model.preprocessingConfiguration,
                        },
                        null,
                        2,
                      )}
                    </pre>
                  </details>
                </>
              ) : (
                <p>{model.reason}</p>
              )}
            </div>
          )}
          {models.data?.some((item) => !item.available) && (
            <details>
              <summary className="cursor-pointer text-xs text-primary">
                Unavailable model requirements
              </summary>
              {models.data
                .filter((item) => !item.available)
                .map((item) => (
                  <p key={item.id} className="mt-2 text-xs">
                    {item.name}: {item.reason}
                  </p>
                ))}
            </details>
          )}
          <Field label="Search registered datasets">
            <input
              className="control"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setOffset(0);
              }}
            />
          </Field>
          <div className="space-y-3">
            {datasets.data?.items.map((d) => (
              <div
                key={d.id}
                data-testid={`registered-dataset-${d.id}`}
                role="group"
                aria-label={`Registered dataset ${d.name}`}
                className="border border-border p-3 text-xs space-y-2"
              >
                <p>
                  {d.name} · {d.entries.length} labeled windows · {d.source}
                </p>
                <p className="break-all">SHA-256 {d.hash}</p>
                <p>{d.note}</p>
                <Button
                  disabled={action.isPending || !model?.available}
                  onClick={() =>
                    action.mutate(async () => {
                      if (!model?.available)
                        throw new Error("Select an available registered model.");
                      const j = await researchApi.evaluate(d.id, model.id, model.version);
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
                <TrainingDatasetExport dataset={d} />
              </div>
            ))}
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              disabled={!offset}
              onClick={() => setOffset(Math.max(0, offset - 50))}
            >
              Previous dataset page
            </Button>
            <Button
              variant="outline"
              disabled={offset + 50 >= (datasets.data?.total ?? 0)}
              onClick={() => setOffset(offset + 50)}
            >
              Next dataset page
            </Button>
          </div>
          {job && (
            <p role="status">
              {job.kind === "dataset-import" ? "Dataset import" : "External evaluation"}:{" "}
              {job.state} · {Math.round(job.progress * 100)}% {job.phase} {job.error}
            </p>
          )}
          {job && ["queued", "running"].includes(job.state) && (
            <Button
              variant="outline"
              onClick={() => action.mutate(() => operationsApi.cancel(job.id))}
            >
              Cancel dataset operation
            </Button>
          )}
          {result.data && (
            <>
              <p className="text-xs">
                Actual evaluation: {result.data.modelId} · {result.data.modelVersion} ·{" "}
                {result.data.dataset.name}
              </p>
              <p className="break-all font-mono text-xs">
                Model SHA-256 {result.data.modelHash}
                <br />
                Dataset SHA-256 {result.data.dataset.hash}
              </p>
              <p className="text-xs">
                Median inference {numeric(result.data.latency.inferenceMedianMs)} ms; p95{" "}
                {numeric(result.data.latency.inferenceP95Ms)} ms. Preprocessing, initialization and
                request time are separate.
              </p>
              <EvaluationMetricsPanel
                metrics={result.data.metrics}
                classOrder={result.data.classOrder}
              />
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
              <Button
                variant="outline"
                onClick={() =>
                  action.mutate(() =>
                    operationDownload(
                      `/jobs/${jobId}/export?format=pdf`,
                      "satmod-external-evaluation.pdf",
                    ),
                  )
                }
              >
                Export external evaluation PDF
              </Button>
              <details>
                <summary className="cursor-pointer text-xs text-primary">
                  Full evaluation metrics JSON
                </summary>
                <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words text-xs">
                  {JSON.stringify(result.data.metrics, null, 2)}
                </pre>
              </details>
              <p className="text-xs">
                Raw model-score calibration diagnostics apply only to these labeled test observations.
                SNR bins exist only where importers supply independently measured SNR metadata.
              </p>
            </>
          )}
        </div>
      </details>
    </Surface>
  );
}
