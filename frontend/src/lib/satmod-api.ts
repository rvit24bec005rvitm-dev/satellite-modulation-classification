export type Modulation = "BPSK" | "QPSK" | "16-QAM";
export type GeneratorModulation = Modulation | "8-PSK" | "16-APSK" | "32-APSK" | "GMSK" | "BFSK";
export const SUPPORTED_MODULATIONS: Modulation[] = ["BPSK", "QPSK", "16-QAM"];
export const GENERATOR_MODULATIONS: GeneratorModulation[] = [
  ...SUPPORTED_MODULATIONS,
  "8-PSK",
  "16-APSK",
  "32-APSK",
  "GMSK",
  "BFSK",
];
export interface IQSample {
  i: number;
  q: number;
}
export interface WavePoint extends IQSample {
  sample: number;
  time: number;
}
export interface Distribution {
  value: number;
  count: number;
}
export interface SignalAnalysis {
  sampleCount: number;
  sampleRate: number;
  signalPower: number;
  powerDb: number;
  rmsAmplitude: number;
  bandwidthHz: number;
  noisePower: number | null;
  estimatedSnrDb: number | null;
  qualityNote: string;
  features: Record<string, number | null>;
  waveform: WavePoint[];
  constellation: IQSample[];
  spectrum: { frequency: number; psd: number; magnitude: number }[];
  amplitudeDistribution: Distribution[];
  phaseDistribution: Distribution[];
  spectrogram: { frequencies: number[]; times: number[]; db: number[][] };
  eye: { time: number[]; i: number[][]; q: number[][] } | null;
}
export interface Capture {
  samples: IQSample[];
  source: string;
  sampleRate: number;
  analysis: SignalAnalysis;
  modulation?: string;
  snrDb?: number;
}
export interface ClassifyRequest {
  samples: IQSample[];
  source?: string;
  sampleRate?: number;
  windowStart?: number;
}
export interface ClassificationResult extends Capture {
  id: string;
  createdAt: string;
  predictedModulation: Modulation;
  confidence: number;
  probabilities: Record<Modulation, number>;
  entropyBits: number;
  processingMs: number;
  modelId: string;
  preprocessing: Record<string, number | string>;
  warnings: string[];
}
export interface SignalGenerationRequest {
  modulation: GeneratorModulation;
  snrDb: number;
  numSamples: number;
  sps: number;
  sampleRate: number;
  rrcAlpha: number;
  rrcEnabled: boolean;
  dopplerHz: number;
  cfoHz: number;
  phaseNoiseStd: number;
  ricianEnabled: boolean;
  ricianK: number;
  twtaDrive: number;
  amplitudeRms: number;
  seed: number;
}
export interface GeneratedSignal extends Capture {
  modulation: GeneratorModulation;
  snrDb: number;
  parameters: SignalGenerationRequest;
  reference: IQSample[];
  classifierSupported: boolean;
}
export interface ApiModel {
  id: string;
  name: string;
  status: "available" | "not-trained" | "unavailable";
  accuracy?: number;
  parameters?: number;
  backend?: string;
  reason?: string;
  error?: string;
}
export interface SnrPoint {
  snrDb: number;
  samples: number;
  accuracy: number;
  BPSK: number;
  QPSK: number;
  "16-QAM": number;
}
export interface EvaluationResults {
  metrics: { accuracy: number; precision: number; recall: number; f1: number; samples: number };
  confusionMatrix: number[][];
  snrAccuracy: SnrPoint[];
  classes: string[];
  perClass: { className: string; precision: number; recall: number; f1: number; support: number }[];
  misclassifications: { actual: string; predicted: string; count: number }[];
  notes: string[];
  source: string;
}
export interface HealthStatus {
  connected: boolean;
  modelId: string;
  backend?: string;
  version: string;
  error?: string;
}
export interface HistoryItem {
  id: string;
  createdAt: string;
  predictedModulation: string;
  confidence: number;
  source: string;
  entropyBits: number;
  processingMs: number;
}
export interface DemoSample {
  id: number;
  modulation: string;
  snrDb: number;
  sampleCount: number;
}
export interface SystemInfo {
  name: string;
  version: string;
  classes: string[];
  inputShape: number[];
  architecture: string;
  normalization: { mean: number; std: number; policy: string };
  defaultAmplitudeRms: number;
  maxUploadBytes: number;
  maxSamples: number;
  historyPolicy: string;
}
export interface BenchmarkResult {
  modelId: string;
  iterations: number;
  sampleId: number;
  meanMs: number;
  medianMs: number;
  p95Ms: number;
  minMs: number;
  maxMs: number;
  throughputFps: number;
  latenciesMs: number[];
  scope: string;
}
export interface Comparison {
  first: SignalAnalysis;
  second: SignalAnalysis;
  powerDifferenceDb: number;
  sampleMse: number | null;
  note: string;
}
export interface OrbitResult {
  time_sec: number[];
  elevation_deg: number[];
  doppler_khz: number[];
  range_km: number[];
  v_orbit_kms: number;
  period_min: number;
  max_doppler_khz: number;
}

export function apiBase(): string {
  const stored = typeof window === "undefined" ? null : localStorage.getItem("satmod-api-url");
  return (stored ?? import.meta.env["VITE_API_BASE_URL"] ?? "").replace(/\/$/, "");
}
export async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${apiBase()}${path}`, {
      ...options,
      headers: {
        ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...options.headers,
      },
      signal: options.signal ?? AbortSignal.timeout(60000),
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error(
      "Backend connection failed. Start the Python service and check the API address in Settings.",
    );
  }
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    const detail =
      typeof body === "object" && body !== null
        ? "detail" in body
          ? body.detail
          : "error" in body
            ? body.error
            : null
        : null;
    throw new Error(
      typeof detail === "string" ? detail : `Signal service returned HTTP ${response.status}`,
    );
  }
  return response.json() as Promise<T>;
}
const post = <T>(path: string, body: unknown, signal?: AbortSignal) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body), ...(signal ? { signal } : {}) });
export const satmodApi = {
  health: (signal?: AbortSignal) => request<HealthStatus>("/api/health", signal ? { signal } : {}),
  info: (signal?: AbortSignal) => request<SystemInfo>("/api/system/info", signal ? { signal } : {}),
  classify: (body: ClassifyRequest) => post<ClassificationResult>("/api/classify", body),
  upload: (file: File, windowStart = 0) => {
    const form = new FormData();
    form.append("file", file);
    return request<ClassificationResult>(`/api/classify/upload?windowStart=${windowStart}`, {
      method: "POST",
      body: form,
    });
  },
  generate: (body: SignalGenerationRequest) => post<GeneratedSignal>("/api/signal/generate", body),
  analyze: (capture: Capture) =>
    post<SignalAnalysis>("/api/signal/analyze", {
      samples: capture.samples,
      sampleRate: capture.sampleRate,
      source: capture.source,
    }),
  compare: (first: Capture, second: Capture) =>
    post<Comparison>("/api/signal/compare", {
      first: { samples: first.samples, sampleRate: first.sampleRate, source: first.source },
      second: { samples: second.samples, sampleRate: second.sampleRate, source: second.source },
    }),
  models: (signal?: AbortSignal) => request<ApiModel[]>("/api/models", signal ? { signal } : {}),
  evaluation: (signal?: AbortSignal) =>
    request<EvaluationResults>("/api/evaluation", signal ? { signal } : {}),
  history: (signal?: AbortSignal) =>
    request<HistoryItem[]>("/api/history", signal ? { signal } : {}),
  samples: (signal?: AbortSignal) =>
    request<DemoSample[]>("/api/samples", signal ? { signal } : {}),
  sample: (id: number, signal?: AbortSignal) =>
    request<Capture>(`/api/samples/${id}`, signal ? { signal } : {}),
  benchmark: (modelId: string, sampleId: number, iterations: number) =>
    post<BenchmarkResult>("/api/benchmark", { modelId, sampleId, iterations }),
  orbit: (altitude: number, carrier: number) =>
    request<OrbitResult>(`/api/orbit?altitudeKm=${altitude}&carrierGhz=${carrier}`),
};
export function downloadJson(value: unknown, name: string) {
  downloadBlob(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }), name);
}
export function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function downloadCapture(capture: Capture) {
  downloadBlob(
    new Blob(["I,Q\n" + capture.samples.map((p) => `${p.i},${p.q}`).join("\n")], {
      type: "text/csv",
    }),
    "satmod-iq.csv",
  );
}
