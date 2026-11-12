import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import type { DatasetRow } from "@/lib/operations-api";
import { researchApi, researchDownload } from "@/lib/research-api";
import { downloadJson } from "@/lib/satmod-api";
import { Button } from "@/components/ui/button";
import { Field, Message, NumberField, Stats } from "@/components/workspace-ui";

export function TrainingDatasetExport({ dataset }: { dataset: DatasetRow }) {
  const [model, setModel] = useState("random-forest");
  const [seed, setSeed] = useState(42);
  const [allowSmall, setAllowSmall] = useState(false);
  const validation = useMutation({
    mutationFn: () => researchApi.trainingManifest(dataset.id, model, seed, allowSmall),
  });
  const download = useMutation({
    mutationFn: () =>
      researchDownload(
        `/datasets/${dataset.id}/training.npz?model=${model}&seed=${seed}&allowSmall=${allowSmall}`,
        validation.data!.artifact.filename,
      ),
  });
  const reset = () => {
    validation.reset();
    download.reset();
  };
  return (
    <details className="mt-3 border-t border-border pt-3">
      <summary className="cursor-pointer font-semibold text-primary">
        Offline training dataset export
      </summary>
      <div className="mt-3 space-y-3">
        <p className="text-xs leading-5 text-muted-foreground">
          Requires explicit train, validation and test splits, every supported class in each split,
          and at least nine declared independent acquisition sessions. Test-only datasets are never
          repartitioned. Files preserve raw windows and provenance; fitting remains in the offline
          training script. Production weights are not changed.
        </p>
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Offline training architecture">
            <select
              className="control"
              aria-label={`Offline training architecture for ${dataset.id}`}
              value={model}
              onChange={(event) => {
                setModel(event.target.value);
                reset();
              }}
            >
              <option value="random-forest">Random Forest</option>
              <option value="cnn">1D-CNN</option>
              <option value="resnet">Residual CNN</option>
              <option value="cldnn">CLDNN</option>
            </select>
          </Field>
          <NumberField
            label="Offline training seed"
            min={0}
            max={4294967295}
            value={seed}
            onChange={(value) => {
              setSeed(value);
              reset();
            }}
          />
        </div>
        <label className="flex items-start gap-2 text-xs leading-5">
          <input
            type="checkbox"
            className="mt-1"
            checked={allowSmall}
            onChange={(event) => {
              setAllowSmall(event.target.checked);
              reset();
            }}
          />
          Allow fewer than 1,000 windows for an explicitly experimental smoke dataset. Session,
          class and leakage checks still apply.
        </label>
        <Button
          variant="outline"
          disabled={
            validation.isPending || !Number.isInteger(seed) || seed < 0 || seed > 4294967295
          }
          onClick={() => validation.mutate()}
        >
          {validation.isPending
            ? "Validating registered splits…"
            : "Validate offline training export"}
        </Button>
        <Message error={validation.error ?? download.error} />
        {validation.data && (
          <>
            <Stats
              items={[
                ["Exported windows", validation.data.sampleCount],
                ["Known SNR metadata", validation.data.snr.knownSamples],
                ["Unavailable SNR", validation.data.snr.unavailableSamples],
                ["Artifact bytes", validation.data.artifact.bytes],
              ]}
            />
            <p className="break-all font-mono text-xs">
              NPZ SHA-256 {validation.data.artifact.sha256}
            </p>
            <p className="text-xs">
              Unavailable SNR remains explicitly masked; no measured value is invented.
            </p>
            <div className="flex flex-wrap gap-2">
              <Button disabled={download.isPending} onClick={() => download.mutate()}>
                {download.isPending ? "Exporting training NPZ…" : "Export validated training NPZ"}
              </Button>
              <Button
                variant="outline"
                onClick={() =>
                  downloadJson(validation.data, `${dataset.id}-training-manifest.json`)
                }
              >
                Export training manifest JSON
              </Button>
              <Button
                variant="outline"
                onClick={() =>
                  downloadJson(validation.data!.experimentConfig, "experiment-config.json")
                }
              >
                Export offline experiment config
              </Button>
            </div>
            <p className="text-xs">
              Run locally in a compatible training environment after saving both files:
            </p>
            <pre className="overflow-x-auto whitespace-pre-wrap break-words rounded border border-border p-3 text-xs">
              {validation.data.offlineCommand}
            </pre>
          </>
        )}
      </div>
    </details>
  );
}
