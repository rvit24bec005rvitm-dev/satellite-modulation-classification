import { useQuery } from "@tanstack/react-query";
import { satmodApi, downloadJson } from "@/lib/satmod-api";
import { Surface, SectionHeading, Message, percent, ExportButton } from "@/components/workspace-ui";
export function ExperimentResults() {
  const query = useQuery({
    queryKey: ["experiments"],
    queryFn: ({ signal }) => satmodApi.experiments(signal),
    retry: 1,
  });
  return (
    <Surface className="p-4 space-y-4">
      <SectionHeading
        title="Controlled synthetic experiments"
        note="MATCHED HELD-OUT TEST · EXPERIMENTAL"
      />
      <p className="text-xs leading-6 text-muted-foreground">
        These genuinely trained candidate models and the unchanged production CNN were evaluated on
        identical generated test frames. Source sessions are disjoint across train, validation and
        test. These results apply to the recorded synthetic domain and do not reproduce the
        historical 900-observation benchmark. No candidate was promoted.
      </p>
      <Message loading={query.isPending} error={query.error} />
      {query.data?.length === 0 && (
        <p className="text-xs text-muted-foreground">
          No completed experiment reports are installed.
        </p>
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr>
              {[
                "Experiment",
                "Test frames",
                "Candidate accuracy",
                "Candidate macro F1",
                "CNN on same test",
                "Protocol",
              ].map((t) => (
                <th key={t} className="p-2">
                  {t}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {query.data?.map((run) => (
              <tr key={run.id} className="border-t border-border">
                <td className="p-2">{run.id}</td>
                <td className="p-2">{run.test.samples}</td>
                <td className="p-2">{percent(run.test.accuracy)}</td>
                <td className="p-2">{percent(run.test.macroF1)}</td>
                <td className="p-2">{percent(run.productionOnSameTest.accuracy)}</td>
                <td className="p-2 font-mono" title={run.protocolId}>
                  {run.protocolId.slice(0, 10)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {query.data && (
        <ExportButton
          onClick={() => downloadJson(query.data, "satmod-controlled-experiments.json")}
        />
      )}
    </Surface>
  );
}
