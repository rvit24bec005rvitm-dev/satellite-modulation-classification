import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { operationsApi, operationDownload } from "@/lib/operations-api";
import { Surface, Message, Field, percent, numeric } from "@/components/workspace-ui";
import { Button } from "@/components/ui/button";
export function PersistentHistory() {
  const cache = useQueryClient();
  const [search, setSearch] = useState("");
  const [offset, setOffset] = useState(0);
  const [modulation, setModulation] = useState("");
  const query = useQuery({
    queryKey: ["analysis-history", search, offset, modulation],
    queryFn: () => operationsApi.history(search, offset, modulation),
    refetchInterval: 5000,
  });
  const action = useMutation({
    mutationFn: (fn: () => Promise<unknown>) => fn(),
    onSuccess: () => {
      void cache.invalidateQueries({ queryKey: ["analysis-history"] });
      void cache.invalidateQueries({ queryKey: ["history"] });
    },
  });
  return (
    <Surface className="p-4 space-y-3">
      <details>
        <summary className="cursor-pointer font-semibold text-primary">
          Persistent analysis history and technical reports
        </summary>
        <div className="space-y-3 mt-4">
          <p className="text-xs text-muted-foreground">
            Stored locally across backend restarts. Shared workspace: access controls do not provide
            per-user isolation.
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Search analysis history">
              <input
                className="control"
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value);
                  setOffset(0);
                }}
              />
            </Field>
            <Field label="History class filter">
              <select
                className="control"
                value={modulation}
                onChange={(e) => {
                  setModulation(e.target.value);
                  setOffset(0);
                }}
              >
                <option value="">All classifications</option>
                {["BPSK", "QPSK", "16-QAM"].map((v) => (
                  <option key={v}>{v}</option>
                ))}
              </select>
            </Field>
          </div>
          <Message error={query.error ?? action.error} />
          <p className="text-xs">{query.data?.total ?? 0} saved analyses</p>
          <div className="overflow-auto">
            <table className="w-full text-xs">
              <thead>
                <tr>
                  {[
                    "Timestamp",
                    "Source",
                    "Class",
                    "Raw confidence",
                    "Entropy",
                    "Inference ms",
                    "Export / delete",
                  ].map((v) => (
                    <th key={v}>{v}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {query.data?.items.map((r) => (
                  <tr key={r.id} className="border-t border-border">
                    <td>{r.createdAt}</td>
                    <td>{r.source}</td>
                    <td>{r.predictedModulation}</td>
                    <td>{percent(r.confidence)}</td>
                    <td>{numeric(r.entropyBits)}</td>
                    <td>{numeric(r.processingMs)}</td>
                    <td>
                      <div className="flex gap-1">
                        {["json", "csv", "pdf"].map((format) => (
                          <Button
                            key={format}
                            variant="outline"
                            onClick={() =>
                              action.mutate(() =>
                                operationDownload(
                                  `/history/${r.id}/report?format=${format}`,
                                  `satmod-report.${format}`,
                                ),
                              )
                            }
                          >
                            {format.toUpperCase()}
                          </Button>
                        ))}
                        <Button
                          variant="outline"
                          onClick={() => action.mutate(() => operationsApi.deleteHistory(r.id))}
                        >
                          Delete analysis
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex gap-2">
            <Button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 50))}>
              Previous history page
            </Button>
            <Button
              disabled={offset + 50 >= (query.data?.total ?? 0)}
              onClick={() => setOffset(offset + 50)}
            >
              Next history page
            </Button>
            <Button
              variant="outline"
              onClick={() => {
                if (
                  window.confirm(
                    "Delete all saved analysis reports? Capture recordings remain stored.",
                  )
                )
                  action.mutate(() => operationsApi.clearHistory());
              }}
            >
              Clear saved history
            </Button>
          </div>
        </div>
      </details>
    </Surface>
  );
}
