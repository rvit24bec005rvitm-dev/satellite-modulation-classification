import { request, apiBase, type Capture, type ClassificationResult } from "./satmod-api";
export interface ManagedCapture {
  id: string;
  source: string;
  sampleCount: number;
  sampleRate: number;
  sampleRateKnown: boolean;
  createdAt: string;
  sessionId: string;
  hash: string;
  sourceHash: string;
  split: string;
  license: string;
  centerFrequency: number | null;
  sourceOffset: number;
}
export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}
export interface Job {
  id: string;
  kind: string;
  state: string;
  progress: number;
  error?: string;
  cancelRequested: boolean;
  phase?: string;
  captureId?: string;
  datasetId?: string;
}
export interface WindowPrediction extends Pick<
  ClassificationResult,
  "predictedModulation" | "confidence" | "entropyBits" | "processingMs" | "probabilities"
> {
  index: number;
  start: number;
  captureId: string;
  timestamp: string | null;
}
export interface BatchResult {
  predictions: WindowPrediction[];
  total: number;
  classCounts: Record<string, number>;
  note: string;
  metrics?: Record<string, unknown>;
}
export interface AnalysisRow {
  id: string;
  createdAt: string;
  source: string;
  predictedModulation: string;
  confidence: number;
  entropyBits: number;
  processingMs: number;
  modelHash: string;
}
export interface DatasetRow {
  id: string;
  name: string;
  hash: string;
  source: string;
  entries: unknown[];
  note: string;
}
export interface Status {
  captures: number;
  datasets: number;
  analyses: number;
  runningJobs: number;
  inferenceMedianMs: number | null;
  storageScope: string;
}
export interface Diagnostics {
  dcOffset: { i: number; q: number };
  iqGainImbalanceDb: number | null;
  iqCorrelation: number | null;
  clippedFraction: number | null;
  powerCoefficientOfVariation: number | null;
  frequencyOffset: {
    estimate: number;
    estimateHz: number | null;
    unit: string;
    assumptions: string;
  } | null;
  frequencyUnit: string;
  welchUnits: string;
  welch: { frequency: number; density: number }[];
  powerOverTime: { sample: number; power: number; envelope: number }[];
  clippingNote: string;
  imbalanceNote: string;
}
const prefix = "/api/workspace";
export const operationsApi = {
  captures: (search = "", offset = 0) =>
    request<Page<ManagedCapture>>(
      `${prefix}/captures?search=${encodeURIComponent(search)}&offset=${offset}`,
    ),
  import: (file: File, options: Record<string, unknown>, metadata?: File) => {
    const data = new FormData();
    data.append("file", file);
    data.append("options", JSON.stringify(options));
    if (metadata) data.append("metadata", metadata);
    return request<ManagedCapture>(`${prefix}/captures`, { method: "POST", body: data });
  },
  importJob: (file: File, options: Record<string, unknown>, metadata?: File) => {
    const data = new FormData();
    data.append("file", file);
    data.append("options", JSON.stringify(options));
    if (metadata) data.append("metadata", metadata);
    return request<Job>(`${prefix}/captures?background=true`, { method: "POST", body: data });
  },
  overview: (id: string) =>
    request<{
      metadata: ManagedCapture;
      points: { sample: number; power: number }[];
      note: string;
    }>(`${prefix}/captures/${id}`),
  window: (id: string, start: number) =>
    request<Capture>(`${prefix}/captures/${id}/window?start=${start}`),
  classify: (id: string, start: number) =>
    request<ClassificationResult>(`${prefix}/captures/${id}/classify?start=${start}`, {
      method: "POST",
    }),
  removeCapture: (id: string) => request(`${prefix}/captures/${id}`, { method: "DELETE" }),
  batch: (captureId: string, start: number, windows: number, stride: number) =>
    request<Job>(`${prefix}/jobs/batch`, {
      method: "POST",
      body: JSON.stringify({ captureId, start, windows, stride }),
    }),
  jobs: (offset = 0) => request<Page<Job>>(`${prefix}/jobs?offset=${offset}`),
  job: (id: string) => request<Job>(`${prefix}/jobs/${id}`),
  cancel: (id: string) => request<Job>(`${prefix}/jobs/${id}/cancel`, { method: "POST" }),
  removeJob: (id: string) => request(`${prefix}/jobs/${id}`, { method: "DELETE" }),
  result: (id: string, offset = 0) =>
    request<BatchResult>(`${prefix}/jobs/${id}/result?offset=${offset}`),
  diagnostics: (id: string, start: number, length: number, modulation: string, fullScale: string) =>
    request<Diagnostics>(
      `${prefix}/captures/${id}/diagnostics?start=${start}&length=${length}${modulation ? `&modulation=${modulation}` : ""}${fullScale ? `&fullScale=${encodeURIComponent(fullScale)}` : ""}`,
    ),
  domain: (id: string, start: number) =>
    request<Record<string, unknown>>(`${prefix}/captures/${id}/domain-shift?start=${start}`),
  activity: (id: string, threshold: number) =>
    request<{ segments: { startSample: number; endSample: number }[]; note: string }>(
      `${prefix}/captures/${id}/activity?threshold=${threshold}`,
    ),
  history: (search = "", offset = 0, modulation = "") =>
    request<Page<AnalysisRow>>(
      `${prefix}/history?search=${encodeURIComponent(search)}&offset=${offset}${modulation ? `&modulation=${modulation}` : ""}`,
    ),
  deleteHistory: (id: string) => request(`${prefix}/history/${id}`, { method: "DELETE" }),
  clearHistory: () => request(`${prefix}/history`, { method: "DELETE" }),
  datasets: (offset = 0, search = "") =>
    request<Page<DatasetRow> & { references: unknown[] }>(
      `${prefix}/datasets?offset=${offset}&search=${encodeURIComponent(search)}`,
    ),
  register: (body: unknown) =>
    request<DatasetRow>(`${prefix}/datasets`, { method: "POST", body: JSON.stringify(body) }),
  registerJob: (body: unknown) =>
    request<Job>(`${prefix}/datasets?background=true`, { method: "POST", body: JSON.stringify(body) }),
  evaluate: (id: string) => request<Job>(`${prefix}/datasets/${id}/evaluate`, { method: "POST" }),
  deleteDataset: (id: string) => request(`${prefix}/datasets/${id}`, { method: "DELETE" }),
  status: () => request<Status>(`${prefix}/status`),
};
export async function operationDownload(path: string, name: string) {
  const response = await fetch(`${apiBase()}${prefix}${path}`, {
    headers: { "X-Satmod-Token": sessionStorage.getItem("satmod-access-token:" + apiBase()) ?? "" },
    signal: AbortSignal.timeout(60000),
  });
  if (!response.ok) throw new Error(`Export failed: HTTP ${response.status}`);
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
