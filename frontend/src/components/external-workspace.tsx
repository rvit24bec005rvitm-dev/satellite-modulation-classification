import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { operationsApi, operationDownload, type Job } from "@/lib/operations-api";
import { useSignal } from "@/hooks/use-satmod";
import { downloadJson } from "@/lib/satmod-api";
import { Button } from "@/components/ui/button";
import {
  Surface,
  Field,
  NumberField,
  Message,
  Stats,
  numeric,
  percent,
} from "@/components/workspace-ui";
import { ScientificChart } from "@/components/scientific-chart";
import { ClassificationInsights } from "@/components/classification-insights";

export function ExternalWorkspace() {
  const cache = useQueryClient();
  const { setCapture, setClassification } = useSignal();
  const [file, setFile] = useState<File>();
  const [metadata, setMetadata] = useState<File>();
  const [format, setFormat] = useState("application");
  const [datatype, setDatatype] = useState("");
  const [layout, setLayout] = useState("interleaved");
  const [order, setOrder] = useState("IQ");
  const [fs, setFs] = useState("");
  const [frequency, setFrequency] = useState("");
  const [scale, setScale] = useState(1);
  const [session, setSession] = useState("");
  const [sourceOffset, setSourceOffset] = useState(0);
  const [sourceLength, setSourceLength] = useState("");
  const [license, setLicense] = useState("Unspecified; verify usage rights");
  const [selected, setSelected] = useState("");
  const [captureSearch, setCaptureSearch] = useState("");
  const [captureOffset, setCaptureOffset] = useState(0);
  const [start, setStart] = useState(0);
  const [windows, setWindows] = useState(10);
  const [stride, setStride] = useState(128);
  const [job, setJob] = useState<Job | null>(null);
  const [page, setPage] = useState(0);
  const [error, setError] = useState<Error | null>(null);
  const [cfoScheme, setCfoScheme] = useState("");
  const [fullScale, setFullScale] = useState("");
  const [threshold, setThreshold] = useState(0);
  const list = useQuery({
    queryKey: ["managed-captures", captureSearch, captureOffset],
    queryFn: () => operationsApi.captures(captureSearch, captureOffset),
  });
  const overview = useQuery({
    queryKey: ["capture-overview", selected],
    queryFn: () => operationsApi.overview(selected),
    enabled: !!selected,
  });
  const jobs = useQuery({
    queryKey: ["workspace-jobs"],
    queryFn: () => operationsApi.jobs(),
    refetchInterval: 1500,
  });
  const active = jobs.data?.items.find((j) => j.id === job?.id) ?? job;
  const timeline = useQuery({
    queryKey: ["batch-result", active?.id, page],
    queryFn: () => operationsApi.result(active!.id, page),
    enabled: active?.state === "completed",
  });
  const importer = useMutation({
    mutationFn: () => {
      if (!file) throw Error("Choose a recording.");
      return operationsApi.import(
        file,
        {
          format,
          datatype: datatype || null,
          layout,
          iqOrder: order,
          sampleRate: fs ? Number(fs) : null,
          centerFrequency: frequency ? Number(frequency) : null,
          scale,
          sessionId: session,
          source: file.name,
          sampleOffset: sourceOffset,
          sampleCount: sourceLength ? Number(sourceLength) : null,
          license,
        },
        metadata,
      );
    },
    onSuccess: (c) => {
      setSelected(c.id);
      setCaptureSearch("");
      setCaptureOffset(0);
      setStart(0);
      void cache.invalidateQueries({ queryKey: ["managed-captures"] });
    },
  });
  const inspect = useMutation({
    mutationFn: (offset: number) => operationsApi.window(selected, offset),
    onSuccess: (c) => setCapture(c),
  });
  const classify = useMutation({
    mutationFn: () => operationsApi.classify(selected, start),
    onSuccess: (r) => {
      setClassification(r);
      setCapture(r);
      void cache.invalidateQueries({ queryKey: ["history"] });
      void cache.invalidateQueries({ queryKey: ["analysis-history"] });
    },
  });
  const batch = useMutation({
    mutationFn: () => operationsApi.batch(selected, start, windows, stride),
    onSuccess: (j) => {
      setJob(j);
      setPage(0);
      void cache.invalidateQueries({ queryKey: ["workspace-jobs"] });
    },
  });
  const diagnostics = useMutation({
    mutationFn: () =>
      operationsApi.diagnostics(
        selected,
        start,
        Math.min(4096, (overview.data?.metadata.sampleCount ?? 128) - start),
        cfoScheme,
        fullScale,
      ),
  });
  const domain = useMutation({ mutationFn: () => operationsApi.domain(selected, start) });
  const activity = useMutation({ mutationFn: () => operationsApi.activity(selected, threshold) });
  const act = async (fn: () => Promise<unknown>) => {
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e as Error);
    }
  };
  const selectWindow = (offset: number, captureId = selected) => {
    setSelected(captureId);
    setStart(offset);
    classify.reset();
    void act(async () => setCapture(await operationsApi.window(captureId, offset)));
  };
  return (
    <Surface className="p-4 space-y-4">
      <details>
        <summary className="cursor-pointer font-semibold text-primary">
          External recordings · capture browser, jobs and prediction timeline
        </summary>
        <div className="mt-4 space-y-4">
          <p className="text-xs text-muted-foreground">
            Managed disk storage; raw samples preserve explicit scaling. Windows use the existing
            128 × 2 CNN. No model result establishes a real satellite label.
          </p>
          <div className="grid gap-3 md:grid-cols-3">
            <Field label="External recording file">
              <input
                className="control"
                type="file"
                onChange={(e) => setFile(e.target.files?.[0])}
              />
            </Field>
            <Field label="External import format">
              <select
                className="control"
                value={format}
                onChange={(e) => setFormat(e.target.value)}
              >
                <option value="application">Existing NPY/NPZ/CSV/MAT/WAV/JSON</option>
                <option value="raw">Explicit complex binary</option>
                <option value="sigmf">SigMF data + metadata</option>
              </select>
            </Field>
            <Field label="Recording session / group ID">
              <input
                className="control"
                value={session}
                onChange={(e) => setSession(e.target.value)}
                placeholder="Required: acquisition session"
              />
            </Field>
            {format === "sigmf" && (
              <Field label="SigMF metadata file">
                <input
                  className="control"
                  type="file"
                  accept=".sigmf-meta,.json"
                  onChange={(e) => setMetadata(e.target.files?.[0])}
                />
              </Field>
            )}
            {format === "raw" && (
              <>
                <Field label="Binary datatype and endianness">
                  <select
                    className="control"
                    value={datatype}
                    onChange={(e) => setDatatype(e.target.value)}
                  >
                    <option value="">Select explicitly</option>
                    {[
                      "ci8",
                      "ci16_le",
                      "ci16_be",
                      "ci32_le",
                      "ci32_be",
                      "cf32_le",
                      "cf32_be",
                      "cf64_le",
                      "cf64_be",
                    ].map((v) => (
                      <option key={v}>{v}</option>
                    ))}
                  </select>
                </Field>
                <Field label="Binary interleaving">
                  <select
                    className="control"
                    value={layout}
                    onChange={(e) => setLayout(e.target.value)}
                  >
                    <option value="interleaved">IQIQ interleaved</option>
                    <option value="planar">All I then all Q (planar)</option>
                  </select>
                </Field>
                <Field label="Binary channel order">
                  <select
                    className="control"
                    value={order}
                    onChange={(e) => setOrder(e.target.value)}
                  >
                    <option>IQ</option>
                    <option>QI</option>
                  </select>
                </Field>
              </>
            )}
            <Field label="Recording sample rate (Hz)">
              <input
                className="control"
                type="number"
                min={1}
                value={fs}
                onChange={(e) => setFs(e.target.value)}
                placeholder="Required for raw; SigMF reads metadata"
              />
            </Field>
            <Field label="Center frequency (Hz, optional)">
              <input
                className="control"
                type="number"
                min={0}
                value={frequency}
                onChange={(e) => setFrequency(e.target.value)}
              />
            </Field>
            <NumberField
              label="Explicit sample scaling"
              value={scale}
              onChange={setScale}
              min={0}
            />
            <NumberField
              label="Import sample offset"
              value={sourceOffset}
              onChange={setSourceOffset}
              min={0}
            />
            <Field label="Import window length (blank = remainder)">
              <input
                className="control"
                type="number"
                min={128}
                value={sourceLength}
                onChange={(e) => setSourceLength(e.target.value)}
              />
            </Field>
            <Field label="License / usage restrictions">
              <input
                className="control"
                value={license}
                onChange={(e) => setLicense(e.target.value)}
              />
            </Field>
          </div>
          <Button
            disabled={
              importer.isPending ||
              !file ||
              !session ||
              (format === "raw" && (!datatype || !fs)) ||
              (format === "sigmf" && !metadata)
            }
            onClick={() => importer.mutate()}
          >
            {importer.isPending ? "Importing bounded capture…" : "Import external recording"}
          </Button>
          <Message
            error={
              error ??
              importer.error ??
              list.error ??
              overview.error ??
              inspect.error ??
              classify.error ??
              batch.error ??
              timeline.error ??
              diagnostics.error ??
              domain.error ??
              activity.error ??
              jobs.error
            }
          />
          <Field label="Search stored captures">
            <input
              className="control"
              value={captureSearch}
              onChange={(e) => {
                setCaptureSearch(e.target.value);
                setCaptureOffset(0);
              }}
            />
          </Field>
          <Field label="Stored capture">
            <select
              className="control"
              value={selected}
              onChange={(e) => {
                setSelected(e.target.value);
                setStart(0);
                classify.reset();
                diagnostics.reset();
                domain.reset();
                activity.reset();
              }}
            >
              <option value="">Select a managed recording</option>
              {list.data?.items.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.source} · {c.sampleCount.toLocaleString()} samples
                </option>
              ))}
            </select>
          </Field>
          <div className="flex gap-2">
            <Button
              variant="outline"
              disabled={!captureOffset}
              onClick={() => setCaptureOffset(Math.max(0, captureOffset - 50))}
            >
              Previous capture page
            </Button>
            <Button
              variant="outline"
              disabled={captureOffset + 50 >= (list.data?.total ?? 0)}
              onClick={() => setCaptureOffset(captureOffset + 50)}
            >
              Next capture page
            </Button>
          </div>
          {overview.data && (
            <>
              <Stats
                items={[
                  ["Capture ID", selected],
                  ["Sample count", overview.data.metadata.sampleCount],
                  [
                    "Sample rate",
                    overview.data.metadata.sampleRateKnown
                      ? overview.data.metadata.sampleRate
                      : "Unknown / normalized",
                  ],
                  ["Session", overview.data.metadata.sessionId],
                ]}
              />
              <p className="text-xs break-all">
                SHA-256 {overview.data.metadata.hash} · {overview.data.metadata.license}
              </p>
              <ScientificChart
                data={[
                  {
                    x: overview.data.points.map((p) => p.sample),
                    y: overview.data.points.map((p) => p.power),
                    type: "scatter",
                    mode: "lines",
                    name: "Contiguous mean power",
                  },
                ]}
                xTitle="Capture sample index"
                yTitle="Relative sample power"
                height={200}
              />
              <div className="grid gap-3 sm:grid-cols-3">
                <NumberField
                  label="External window start"
                  min={0}
                  max={overview.data.metadata.sampleCount - 128}
                  value={start}
                  onChange={(n) => {
                    setStart(n);
                    classify.reset();
                  }}
                />
                <NumberField
                  label="Batch window count"
                  value={windows}
                  onChange={setWindows}
                  min={1}
                  max={10000}
                />
                <NumberField
                  label="Batch stride (samples)"
                  value={stride}
                  onChange={setStride}
                  min={1}
                />
              </div>
              <div className="flex flex-wrap gap-2">
                <Button disabled={inspect.isPending} onClick={() => selectWindow(start)}>
                  Inspect selected window
                </Button>
                <Button disabled={classify.isPending} onClick={() => classify.mutate()}>
                  Classify external window
                </Button>
                <Button disabled={batch.isPending} onClick={() => batch.mutate()}>
                  Start batch classification
                </Button>
                <Button
                  variant="outline"
                  onClick={() =>
                    void act(() =>
                      operationDownload(
                        `/captures/${selected}/segment?start=${start}&length=128`,
                        "satmod-segment.npy",
                      ),
                    )
                  }
                >
                  Export selected segment
                </Button>
                <Button
                  variant="outline"
                  onClick={() =>
                    void act(async () => {
                      await operationsApi.removeCapture(selected);
                      setSelected("");
                      await cache.invalidateQueries({ queryKey: ["managed-captures"] });
                    })
                  }
                >
                  Delete selected capture
                </Button>
              </div>
              <div className="grid gap-3 sm:grid-cols-3">
                <Field label="CFO assumption (optional)">
                  <select
                    className="control"
                    value={cfoScheme}
                    onChange={(e) => setCfoScheme(e.target.value)}
                  >
                    <option value="">Unknown / unavailable</option>
                    <option>BPSK</option>
                    <option>QPSK</option>
                  </select>
                </Field>
                <Field label="ADC full scale (optional)">
                  <input
                    className="control"
                    type="number"
                    value={fullScale}
                    onChange={(e) => setFullScale(e.target.value)}
                  />
                </Field>
                <NumberField
                  label="Activity power threshold"
                  value={threshold}
                  onChange={setThreshold}
                  min={0}
                />
              </div>
              <div className="flex gap-2 flex-wrap">
                <Button variant="outline" onClick={() => diagnostics.mutate()}>
                  Run quality diagnostics
                </Button>
                <Button variant="outline" onClick={() => domain.mutate()}>
                  Compare demo-domain statistics
                </Button>
                <Button variant="outline" onClick={() => activity.mutate()}>
                  Detect signal activity
                </Button>
              </div>
              {diagnostics.data && (
                <>
                  <Stats
                    items={[
                      ["DC I", numeric(diagnostics.data.dcOffset.i)],
                      ["DC Q", numeric(diagnostics.data.dcOffset.q)],
                      ["I/Q gain indicator dB", numeric(diagnostics.data.iqGainImbalanceDb)],
                      ["Clipped fraction", numeric(diagnostics.data.clippedFraction)],
                      [
                        "CFO diagnostic",
                        `${numeric(diagnostics.data.frequencyOffset?.estimate)} ${diagnostics.data.frequencyOffset?.unit ?? ""}`,
                      ],
                    ]}
                  />
                  <p className="text-xs">
                    {diagnostics.data.clippingNote} {diagnostics.data.imbalanceNote}{" "}
                    {diagnostics.data.frequencyOffset?.assumptions}
                  </p>
                  <ScientificChart
                    data={[
                      {
                        x: diagnostics.data.welch.map((p) => p.frequency),
                        y: diagnostics.data.welch.map((p) => p.density),
                        type: "scatter",
                        name: "Welch PSD",
                      },
                    ]}
                    xTitle={`Frequency (${diagnostics.data.frequencyUnit})`}
                    yTitle={diagnostics.data.welchUnits}
                    height={200}
                  />
                  <ScientificChart
                    data={[
                      {
                        x: diagnostics.data.powerOverTime.map((p) => p.sample),
                        y: diagnostics.data.powerOverTime.map((p) => p.power),
                        type: "scatter",
                        name: "Power over time",
                      },
                    ]}
                    xTitle="Window sample index"
                    yTitle="Relative power"
                    height={180}
                  />
                </>
              )}
              {domain.data && (
                <div className="overflow-auto text-xs">
                  <p>{String(domain.data["note"])}</p>
                  <pre>{JSON.stringify(domain.data, null, 2)}</pre>
                </div>
              )}
              {activity.data && (
                <>
                  <p className="text-xs">{activity.data.note}</p>
                  {activity.data.segments.map((s) => (
                    <Button
                      key={s.startSample}
                      variant="outline"
                      onClick={() =>
                        selectWindow(
                          Math.min(s.startSample, overview.data!.metadata.sampleCount - 128),
                        )
                      }
                    >
                      Activity {s.startSample}–{s.endSample}
                    </Button>
                  ))}
                </>
              )}
              {classify.data && (
                <>
                  <Stats
                    items={[
                      ["External prediction", classify.data.predictedModulation],
                      ["Raw softmax", percent(classify.data.confidence)],
                      ["Entropy bits", numeric(classify.data.entropyBits)],
                    ]}
                  />
                  <ClassificationInsights
                    key={classify.data.id}
                    result={classify.data}
                    onExplanation={(explanation) =>
                      downloadJson(
                        { analysis: classify.data, explanation },
                        "satmod-explained-window.json",
                      )
                    }
                  />
                </>
              )}
            </>
          )}
          <div className="space-y-2">
            <p className="text-sm">Persistent jobs · select to inspect results</p>
            {jobs.data?.items.slice(0, 20).map((j) => (
              <div key={j.id} className="flex flex-wrap items-center gap-3 text-xs">
                <button
                  className="text-primary"
                  onClick={() => {
                    setJob(j);
                    setPage(0);
                  }}
                >
                  {j.kind} · {j.state} · {percent(j.progress)}
                </button>
                {["queued", "running"].includes(j.state) && (
                  <Button
                    variant="outline"
                    onClick={() => void act(() => operationsApi.cancel(j.id))}
                  >
                    Cancel job
                  </Button>
                )}
                {j.error && <span role="alert">{j.error}</span>}
                {!["queued", "running"].includes(j.state) && (
                  <Button
                    variant="outline"
                    onClick={() =>
                      void act(async () => {
                        await operationsApi.removeJob(j.id);
                        if (job?.id === j.id) setJob(null);
                        await cache.invalidateQueries({ queryKey: ["workspace-jobs"] });
                      })
                    }
                  >
                    Delete job result
                  </Button>
                )}
              </div>
            ))}
          </div>
          {active && (
            <p role="status">
              Job {active.state} · {percent(active.progress)} complete
            </p>
          )}
          {timeline.data && (
            <>
              <p className="text-xs">{timeline.data.note}</p>
              <Stats items={Object.entries(timeline.data.classCounts).map(([k, v]) => [k, v])} />
              <div className="max-h-72 overflow-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr>
                      {[
                        "Window",
                        "Sample",
                        "Class",
                        "Softmax",
                        "Entropy",
                        "Inference ms",
                        "Timestamp",
                      ].map((t) => (
                        <th key={t}>{t}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {timeline.data.predictions.map((p) => (
                      <tr key={p.index} className="border-t border-border">
                        <td>
                          <button
                            className="text-primary"
                            onClick={() => {
                              selectWindow(p.start, p.captureId);
                            }}
                          >
                            {p.index}
                          </button>
                        </td>
                        <td>{p.start}</td>
                        <td>{p.predictedModulation}</td>
                        <td>{percent(p.confidence)}</td>
                        <td>{numeric(p.entropyBits)}</td>
                        <td>{numeric(p.processingMs)}</td>
                        <td>{p.timestamp ?? "Unavailable"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="flex gap-2">
                <Button disabled={page === 0} onClick={() => setPage(Math.max(0, page - 200))}>
                  Previous prediction page
                </Button>
                <Button
                  disabled={page + 200 >= timeline.data.total}
                  onClick={() => setPage(page + 200)}
                >
                  Next prediction page
                </Button>
                <Button
                  variant="outline"
                  onClick={() => downloadJson(timeline.data, "satmod-batch-page.json")}
                >
                  Export prediction page
                </Button>
                <Button
                  variant="outline"
                  onClick={() =>
                    void act(() =>
                      operationDownload(
                        `/jobs/${active!.id}/export?format=json`,
                        "satmod-complete-batch.json",
                      ),
                    )
                  }
                >
                  Export full batch JSON
                </Button>
                <Button
                  variant="outline"
                  onClick={() =>
                    void act(() =>
                      operationDownload(
                        `/jobs/${active!.id}/export?format=csv`,
                        "satmod-complete-batch.csv",
                      ),
                    )
                  }
                >
                  Export full batch CSV
                </Button>
              </div>
              <ScientificChart
                data={[
                  {
                    x: timeline.data.predictions.map((p) => p.start),
                    y: timeline.data.predictions.map((p) => p.confidence),
                    type: "scatter",
                    mode: "lines+markers",
                    name: "Raw softmax by window",
                  },
                ]}
                xTitle="Sample offset (current page)"
                yTitle="Uncalibrated confidence"
                height={210}
              />
              {timeline.data.metrics && (
                <pre className="max-h-72 overflow-auto text-xs">
                  {JSON.stringify(timeline.data.metrics, null, 2)}
                </pre>
              )}
            </>
          )}
        </div>
      </details>
    </Surface>
  );
}
