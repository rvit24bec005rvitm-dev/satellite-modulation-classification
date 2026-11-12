import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { useSignal, useHealth, useInfo, useModels } from "@/hooks/use-satmod";
import { apiBase } from "@/lib/satmod-api";
import { Button } from "@/components/ui/button";
import { Surface, SectionHeading, Field, NumberField, Message } from "@/components/workspace-ui";
export function SettingsPage() {
  const health = useHealth(),
    info = useInfo(),
    models = useModels();
  const { preferences, setPreferences } = useSignal();
  const client = useQueryClient();
  const [url, setUrl] = useState(apiBase);
  const [error, setError] = useState<Error | null>(null);
  const save = () => {
    try {
      const trimmed = url.trim().replace(/\/$/, "");
      if (trimmed) {
        const parsed = new URL(trimmed);
        if (
          !["http:", "https:"].includes(parsed.protocol) ||
          parsed.username ||
          parsed.password ||
          parsed.search ||
          parsed.hash
        )
          throw new Error(
            "Use an HTTP(S) API URL without credentials, query parameters, or fragments.",
          );
      }
      localStorage.setItem("satmod-api-url", trimmed);
      client.clear();
      void client.invalidateQueries();
      setError(null);
      toast.success("API address saved");
      window.location.reload();
    } catch (e) {
      setError(e as Error);
    }
  };
  return (
    <div className="grid items-start gap-5 xl:grid-cols-2">
      <Surface className="p-5">
        <SectionHeading
          title="Backend connection"
          note={health.isSuccess && health.data.connected ? "CONNECTED" : "UNAVAILABLE"}
        />
        <Message error={health.error ?? error} loading={health.isPending} />
        <Field label="API base URL (blank uses the same-origin /api proxy)">
          <input
            className="control"
            type="url"
            placeholder="http://localhost:8000"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
        </Field>
        <div className="flex flex-wrap gap-2">
          <Button onClick={save}>Save API address</Button>
          <Button
            variant="outline"
            disabled={health.isFetching}
            onClick={() => void health.refetch()}
          >
            Check connection
          </Button>
        </div>
        <p className="mt-5 text-xs leading-6 text-muted-foreground">
          {health.data?.backend} {health.data ? `· API v${health.data.version}` : ""}
        </p>
        <p className="mt-2 text-xs leading-6 text-muted-foreground">
          The backend must allow this page's origin through SATMOD_CORS_ORIGINS when using a
          separate address. Preferences are stored in this browser.
        </p>
      </Surface>
      <Surface className="p-5">
        <Field label="Workspace access token (session only)">
          <input
            className="control"
            type="password"
            autoComplete="off"
            defaultValue={sessionStorage.getItem("satmod-access-token:" + apiBase()) ?? ""}
            onChange={(e) => {
              sessionStorage.setItem("satmod-access-token:" + apiBase(), e.target.value);
              void client.invalidateQueries();
            }}
          />
        </Field>
        <SectionHeading title="Workspace preferences" note="LOCAL TO THIS BROWSER" />
        <Field label="Theme">
          <select
            className="control"
            value={preferences.theme}
            onChange={(e) =>
              setPreferences({ ...preferences, theme: e.target.value as "dark" | "light" })
            }
          >
            <option value="dark">Mission dark</option>
            <option value="light">Daylight</option>
          </select>
        </Field>
        <Field label="Scientific chart controls">
          <select
            className="control"
            value={preferences.charts}
            onChange={(e) =>
              setPreferences({ ...preferences, charts: e.target.value as "interactive" | "simple" })
            }
          >
            <option value="interactive">Interactive: zoom, pan, export</option>
            <option value="simple">Simple: hide tools and scroll zoom</option>
          </select>
        </Field>
        <Field label="Default modulation">
          <select
            className="control"
            value={preferences.defaultModulation}
            onChange={(e) =>
              setPreferences({
                ...preferences,
                defaultModulation: e.target.value as typeof preferences.defaultModulation,
              })
            }
          >
            {["BPSK", "QPSK", "16-QAM"].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </Field>
        <NumberField
          label="Default SNR (dB)"
          value={preferences.defaultSnr}
          onChange={(n) =>
            Number.isFinite(n) &&
            n >= -30 &&
            n <= 60 &&
            setPreferences({ ...preferences, defaultSnr: n })
          }
          min={-30}
          max={60}
        />
        <NumberField
          label="Default samples per symbol"
          value={preferences.defaultSps}
          onChange={(n) =>
            Number.isInteger(n) &&
            n >= 1 &&
            n <= 32 &&
            setPreferences({ ...preferences, defaultSps: n })
          }
          min={1}
          max={32}
        />
      </Surface>
      <Surface className="p-5">
        <SectionHeading title="Model metadata" note="LOADED ARTIFACT" />
        <Message error={info.error ?? models.error} />
        <p className="text-xs leading-7 text-muted-foreground">{info.data?.architecture}</p>
        {info.data && (
          <dl className="mt-4 space-y-3 text-xs">
            <div>
              <dt className="text-muted-foreground">Training mean / standard deviation</dt>
              <dd className="mt-1 font-mono">
                {info.data.normalization.mean} / {info.data.normalization.std}
              </dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Input shape / class order</dt>
              <dd className="mt-1">
                {info.data.inputShape.join(" × ")} / {info.data.classes.join(", ")}
              </dd>
            </div>
          </dl>
        )}
        {models.data?.map((m) => (
          <div
            key={m.id}
            className="mt-3 flex justify-between gap-3 border-b border-border py-2 text-xs"
          >
            <span>{m.name}</span>
            <span className="font-mono text-primary">{m.status}</span>
          </div>
        ))}
      </Surface>
      <Surface className="p-5">
        <SectionHeading title="Application information" note="SATMODCLASS 1.0.0" />
        <p className="text-xs leading-7 text-muted-foreground">
          {info.data?.historyPolicy ??
            "Backend history policy is unavailable until the service connects."}{" "}
          Model artifacts and the legacy Streamlit interface remain in the Python repository. Saved
          browser preferences contain no credentials.
        </p>
        <a
          className="mt-4 inline-block text-sm text-primary"
          href={`${apiBase()}/docs`}
          target="_blank"
          rel="noreferrer"
        >
          Open FastAPI documentation →
        </a>
      </Surface>
    </div>
  );
}
