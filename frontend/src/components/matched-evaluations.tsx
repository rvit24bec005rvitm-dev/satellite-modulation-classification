import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { operationsApi, operationDownload } from "@/lib/operations-api";
import { researchApi, type EvaluationMetrics } from "@/lib/research-api";
import { downloadJson } from "@/lib/satmod-api";
import { Button } from "@/components/ui/button";
import { Field, Message, Stats, Surface, numeric, percent } from "@/components/workspace-ui";

export function EvaluationMetricsPanel({
  metrics,
  classOrder,
}: {
  metrics: EvaluationMetrics;
  classOrder: string[];
}) {
  return (
    <div className="space-y-3">
      <Stats
        items={[
          ["Labeled windows", metrics.samples],
          ["Observed accuracy", percent(metrics.accuracy)],
          ["Macro precision", percent(metrics.precision)],
          ["Macro recall", percent(metrics.recall)],
          ["Macro F1", percent(metrics.macroF1)],
        ]}
      />
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <caption className="py-2 text-left text-muted-foreground">
            Class metrics; support is the number of supplied labels.
          </caption>
          <thead>
            <tr>
              <th scope="col">Class</th>
              <th scope="col">Support</th>
              <th scope="col">Precision</th>
              <th scope="col">Recall / class accuracy</th>
              <th scope="col">F1</th>
            </tr>
          </thead>
          <tbody>
            {metrics.perClass.map((item) => (
              <tr key={item.className} className="border-t border-border">
                <th scope="row" className="py-2">
                  {item.className}
                </th>
                <td>{item.support}</td>
                <td>{percent(item.precision)}</td>
                <td>{item.support ? percent(item.recall) : "Unavailable"}</td>
                <td>{percent(item.f1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <caption className="py-2 text-left text-muted-foreground">
            Confusion matrix: rows are supplied ground truth; columns are model predictions.
          </caption>
          <thead>
            <tr>
              <th scope="col">True / predicted</th>
              {classOrder.map((label) => (
                <th key={label} scope="col">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metrics.confusionMatrix.map((row, index) => (
              <tr key={classOrder[index]} className="border-t border-border">
                <th scope="row" className="py-2">
                  {classOrder[index]}
                </th>
                {row.map((count, column) => (
                  <td key={classOrder[column]}>{count}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {metrics.snrAccuracy.length ? (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <caption className="py-2 text-left text-muted-foreground">
              Accuracy by importer-supplied SNR metadata (measurement provenance is not inferred).
            </caption>
            <thead>
              <tr>
                <th scope="col">SNR dB</th>
                <th scope="col">Labeled windows</th>
                <th scope="col">Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {metrics.snrAccuracy.map((point) => (
                <tr key={point.snrDb} className="border-t border-border">
                  <th scope="row">{point.snrDb}</th>
                  <td>{point.samples}</td>
                  <td>{percent(point.accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-xs text-muted-foreground">
          SNR accuracy unavailable: no SNR metadata was supplied for these test windows.
        </p>
      )}
      <details>
        <summary className="cursor-pointer text-xs text-primary">
          Raw confidence calibration diagnostics
        </summary>
        <p className="my-2 text-xs">
          Diagnostic only. Test labels do not fit calibration or change production.
        </p>
        <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words text-xs">
          {JSON.stringify(metrics.calibration, null, 2)}
        </pre>
      </details>
    </div>
  );
}

export function MatchedEvaluations() {
  const [expanded, setExpanded] = useState(false);
  const [datasetId, setDatasetId] = useState("");
  const [search, setSearch] = useState("");
  const [offset, setOffset] = useState(0);
  const [protocolId, setProtocolId] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const datasets = useQuery({
    queryKey: ["comparison-datasets", search],
    queryFn: () => operationsApi.datasets(0, search),
    enabled: expanded,
  });
  const query = useQuery({
    queryKey: ["external-evaluations", datasetId, offset],
    queryFn: () => researchApi.evaluations(datasetId, offset),
    enabled: expanded,
    refetchInterval: expanded ? 5000 : false,
  });
  const groups = useMemo(() => {
    const data = query.data?.items ?? [];
    return [...new Set(data.map((item) => item.evaluationProtocol.protocolId))].map((id) => ({
      id,
      items: data.filter((item) => item.evaluationProtocol.protocolId === id),
    }));
  }, [query.data]);
  const active = groups.find((group) => group.id === protocolId) ?? groups[0];
  const comparison = useMutation({ mutationFn: () => researchApi.compareEvaluations(selected) });
  const resetComparison = comparison.reset;
  useEffect(() => {
    setSelected([]);
    resetComparison();
  }, [active?.id, resetComparison]);
  const exported = useMutation({
    mutationFn: (jobId: string) =>
      operationDownload(`/jobs/${jobId}/export?format=json`, `${jobId}-evaluation.json`),
  });
  const clear = () => {
    setSelected([]);
    setProtocolId("");
    comparison.reset();
    exported.reset();
  };
  return (
    <Surface className="p-4 space-y-3">
      <details onToggle={(event) => setExpanded(event.currentTarget.open)}>
        <summary className="cursor-pointer font-semibold text-primary">
          Matched external model evaluations
        </summary>
        <div className="mt-4 space-y-4">
          <p className="text-xs leading-5 text-muted-foreground">
            Compare completed models on one exact dataset ID, hash and held-out test-window
            protocol. Stored synthetic benchmarks above remain separate. Labels and acquisition
            independence are importer-declared; no result promotes a model.
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Find datasets for model comparison">
              <input
                className="control"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </Field>
            <Field label="Comparison evaluation dataset">
              <select
                className="control"
                aria-label="Comparison evaluation dataset"
                value={datasetId}
                onChange={(event) => {
                  setDatasetId(event.target.value);
                  setOffset(0);
                  clear();
                }}
              >
                <option value="">All recorded evaluation protocols</option>
                {datasetId && !datasets.data?.items.some((dataset) => dataset.id === datasetId) && (
                  <option value={datasetId}>Selected dataset {datasetId.slice(0, 8)}</option>
                )}
                {datasets.data?.items.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name} · {dataset.id.slice(0, 8)}
                  </option>
                ))}
              </select>
            </Field>
          </div>
          <Message
            loading={query.isPending && expanded}
            error={query.error ?? datasets.error ?? comparison.error ?? exported.error}
          />
          <Field label="Matched test-window protocol">
            <select
              className="control"
              aria-label="Matched test-window protocol"
              value={active?.id ?? ""}
              onChange={(event) => {
                setProtocolId(event.target.value);
                setSelected([]);
                comparison.reset();
              }}
            >
              {!groups.length && <option value="">No completed external evaluations</option>}
              {groups.map((group) => (
                <option key={group.id} value={group.id}>
                  {group.items[0]!.datasetName} · test · {group.id.slice(0, 10)} ·{" "}
                  {group.items.length} evaluations
                </option>
              ))}
            </select>
          </Field>
          {active && (
            <>
              <p className="break-all font-mono text-xs">
                Dataset SHA-256 {active.items[0]!.datasetHash}
                <br />
                Test-window SHA-256 {active.items[0]!.evaluationProtocol.testWindowsHash}
              </p>
              <p className="text-xs">Select two to eight completed jobs from this protocol.</p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <caption className="py-2 text-left text-muted-foreground">
                    Observed metrics from matching held-out windows.
                  </caption>
                  <thead>
                    <tr>
                      {[
                        "Select",
                        "Model / version",
                        "Status",
                        "Accuracy",
                        "Macro F1",
                        "Median inference ms",
                        "Completed evidence",
                      ].map((label) => (
                        <th key={label} scope="col">
                          {label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {active.items.map((item) => (
                      <tr key={item.jobId} className="border-t border-border">
                        <td>
                          <input
                            type="checkbox"
                            aria-label={`Compare evaluation ${item.jobId}`}
                            checked={selected.includes(item.jobId)}
                            disabled={!selected.includes(item.jobId) && selected.length >= 8}
                            onChange={(event) => {
                              setSelected(
                                event.target.checked
                                  ? [...selected, item.jobId]
                                  : selected.filter((id) => id !== item.jobId),
                              );
                              comparison.reset();
                            }}
                          />
                        </td>
                        <th scope="row" className="p-2 text-left">
                          {item.modelName ?? item.modelId}
                          <br />
                          <span className="font-normal">{item.modelVersion}</span>
                        </th>
                        <td>{item.modelStatus}</td>
                        <td>{percent(item.metrics.accuracy)}</td>
                        <td>{percent(item.metrics.macroF1)}</td>
                        <td>{numeric(item.latency.inferenceMedianMs)}</td>
                        <td>
                          <Button
                            variant="outline"
                            disabled={exported.isPending}
                            onClick={() => exported.mutate(item.jobId)}
                          >
                            Export evaluation JSON
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Button
                disabled={selected.length < 2 || comparison.isPending}
                onClick={() => comparison.mutate()}
              >
                {comparison.isPending
                  ? "Comparing held-out results…"
                  : "Compare selected evaluations"}
              </Button>
            </>
          )}
          {comparison.data && (
            <div className="space-y-3">
              <p className="text-xs">{comparison.data.note}</p>
              <Button
                variant="outline"
                onClick={() =>
                  downloadJson(comparison.data, "satmod-matched-model-comparison.json")
                }
              >
                Export matched comparison JSON
              </Button>
              {comparison.data.evaluations.map((item) => (
                <details key={item.jobId} className="border border-border p-3">
                  <summary className="cursor-pointer text-sm text-primary">
                    {item.modelName ?? item.modelId} · {item.modelVersion} · measured test results
                  </summary>
                  <div className="mt-3 space-y-3">
                    <p className="text-xs">
                      {item.acquisitionType} · {item.datasetSource}
                    </p>
                    <p className="break-all font-mono text-xs">Model SHA-256 {item.modelHash}</p>
                    <EvaluationMetricsPanel metrics={item.metrics} classOrder={item.classOrder} />
                    <details>
                      <summary className="cursor-pointer text-xs text-primary">
                        Preprocessing and hardware
                      </summary>
                      <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words text-xs">
                        {JSON.stringify(
                          {
                            preprocessing: item.preprocessingConfiguration,
                            hardware: item.hardware,
                            timing: item.latency,
                          },
                          null,
                          2,
                        )}
                      </pre>
                    </details>
                  </div>
                </details>
              ))}
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              disabled={!offset}
              onClick={() => {
                setOffset(Math.max(0, offset - 50));
                clear();
              }}
            >
              Previous evaluation page
            </Button>
            <Button
              variant="outline"
              disabled={offset + 50 >= (query.data?.total ?? 0)}
              onClick={() => {
                setOffset(offset + 50);
                clear();
              }}
            >
              Next evaluation page
            </Button>
          </div>
        </div>
      </details>
    </Surface>
  );
}
