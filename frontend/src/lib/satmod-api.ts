export type Modulation = "BPSK" | "QPSK" | "16-QAM";
export type GeneratorModulation =
  Modulation | "32-QAM" | "8-PSK" | "16-APSK" | "32-APSK" | "GMSK" | "BFSK";
export const SUPPORTED_MODULATIONS: Modulation[] = ["BPSK", "QPSK", "16-QAM"];
export const GENERATOR_MODULATIONS: GeneratorModulation[] = [
  ...SUPPORTED_MODULATIONS,
  "32-QAM",
  "8-PSK",
  "16-APSK",
  "32-APSK",
  "GMSK",
  "BFSK",
];
export interface TrustStatus {
  inputValidity: { status: string; checks: string[]; channelOrder: string };
  calibration: { status: string; temperature: number; reason: string };
  ood: { status: string; detected: boolean | null; reason: string };
  interpretation: string;
}
export interface AttributionResult {
  method: "integrated-gradients" | "saliency";
  targetClass: Modulation;
  targetQuantity: string;
  modelId: string;
  windowStart: number;
  windowLength: number;
  integrationSteps: number;
  baseline: { description: string; rawValue: number; normalizedValue: number };
  inputLogit: number;
  baselineLogit: number;
  logitDifference: number;
  attributionSum: number | null;
  completenessError: number | null;
  processingMs: number;
  samples: {
    sample: number;
    i: number;
    q: number;
    importance: number;
    rawI: number;
    rawQ: number;
  }[];
  limitations: string[];
}
export interface CalibrationMetrics {
  ece: number;
  nll: number;
  brier: number;
  reliability: {
    lower: number;
    upper: number;
    count: number;
    accuracy: number | null;
    confidence: number | null;
  }[];
}
export interface MeasuredMetrics {
  samples: number;
  accuracy: number;
  macroF1: number;
  precision: number;
  recall: number;
  calibration: CalibrationMetrics;
  snrAccuracy: {
    snrDb: number;
    samples: number;
    accuracy: number;
    perClass: Record<string, number | null>;
  }[];
}
export interface DemoDiagnostics {
  scope: string;
  independentTest: boolean;
  productionChanged: boolean;
  samples: number;
  comparisons: Record<string, MeasuredMetrics>;
  warning: string;
}
export interface ExperimentReport {
  id: string;
  model: string;
  status: string;
  protocolId: string;
  productionPromoted: boolean;
  dataset: { id: string; sha256: string; samples: number };
  test: MeasuredMetrics;
  productionOnSameTest: MeasuredMetrics;
  calibration: { temperature: number; fitSplit: string; testCalibrated: MeasuredMetrics };
  limitations: string[];
}

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
  powerDb: number | null;
  rmsAmplitude: number;
  bandwidthHz: number | null;
  noisePower: number | null;
  estimatedSnrDb: number | null;
  qualityNote: string;
  features: Record<string, number | null>;
  phaseTrace?: { sample: number; time: number; phaseRad: number }[];
  instantaneousFrequency?: { sample: number; time: number; frequency: number | null }[];
  instantaneousFrequencyNote?: string;
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
  trust: TrustStatus;
  preprocessingMs: number;
  explanation?: AttributionResult;
  preprocessing: Record<string, number | string>;
  warnings: string[];
}
export interface SignalGenerationRequest {
  modulation: GeneratorModulation;
  snrDb: number;
  numSamples: number;
  sps: number;
  sampleRate: number;
  gaussianBt?: number;
  bfskDeviationHz?: number | null;
  continuousPhase?: boolean;
  apskRadii?: number[] | null;
  apskOffsets?: number[] | null;
  phaseNoiseMode?: "iid" | "wiener";
  fadingBlockSamples?: number;
  normalizeOutput?: boolean;
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
  generation?: {
    realizedSnrDb: number;
    channelOrder: string[];
    powerStages: Record<string, number>;
    outputPowerPolicy: string;
    referenceNote: string;
    modulationIndex?: number;
    frequencyDeviationHz?: number;
  };
  reference: IQSample[];
  classifierSupported: boolean;
}
export interface ApiModel {
  id: string;
  name: string;
  status: "available" | "not-trained" | "unavailable";
  accuracy?: number;
  parameters?: number;
  trainableParameters?: number;
  nonTrainableParameters?: number;
  version?: string | null;
  artifactBytes?: number;
  evaluationMetadata?: {
    status: string;
    note: string;
    source: string;
    samples: number;
    datasetId?: string;
  };
  trainingMetadata?: { status: string; reason?: string };
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
  preprocessing: { medianMs: number; p95Ms: number };
  hardware: {
    os: string;
    machine: string;
    processor: string;
    python: string;
    numpy: string;
    batchSize: number;
  };
  scope: string;
}
export interface Comparison {
  first: SignalAnalysis;
  second: SignalAnalysis;
  powerDifferenceDb: number | null;
  sampleMse: number | null;
  processingMs: number;
  ratesEqual: boolean;
  lengthsEqual: boolean;
  overlapSamples: number;
  predictions: Record<
    "first" | "second",
    {
      status: string;
      reason?: string;
      predictedModulation?: string;
      confidence?: number;
      processingMs?: number;
    }
  >;
  plots: {
    frequencyUnit: string;
    timeUnit: string;
    first: {
      waveform: { x: number; i: number; q: number }[];
      spectrum: { frequency: number; psd: number }[];
    };
    second: {
      waveform: { x: number; i: number; q: number }[];
      spectrum: { frequency: number; psd: number }[];
    };
  };
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
        "X-Satmod-Token":
          typeof sessionStorage === "undefined"
            ? ""
            : (sessionStorage.getItem("satmod-access-token:" + apiBase()) ?? ""),
        ...options.headers,
      },
      signal: options.signal
        ? AbortSignal.any([options.signal, AbortSignal.timeout(60000)])
        : AbortSignal.timeout(60000),
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
  explain: (
    body: ClassifyRequest & {
      targetClass?: Modulation;
      method: "integrated-gradients" | "saliency";
      steps: number;
    },
    signal?: AbortSignal,
  ) => post<AttributionResult>("/api/explain", body, signal),
  constellations: (signal?: AbortSignal) =>
    request<{ mapping: string; energyPolicy: string; modulations: Record<string, IQSample[]> }>(
      "/api/constellations",
      signal ? { signal } : {},
    ),
  diagnostics: (signal?: AbortSignal) =>
    request<DemoDiagnostics>("/api/evaluation/diagnostics", signal ? { signal } : {}),
  experiments: (signal?: AbortSignal) =>
    request<ExperimentReport[]>("/api/experiments", signal ? { signal } : {}),
  uploadCapture: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Capture>("/api/signal/upload", { method: "POST", body: form });
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
  orbit: (altitude: number, carrier: number, signal?: AbortSignal) =>
    request<OrbitResult>(
      `/api/orbit?altitudeKm=${altitude}&carrierGhz=${carrier}`,
      signal ? { signal } : {},
    ),
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
