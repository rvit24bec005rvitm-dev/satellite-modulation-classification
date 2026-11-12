import {
  apiBase,
  request,
  downloadBlob,
  type AttributionResult,
  type IQSample,
  type Modulation,
  type SignalAnalysis,
} from "./satmod-api";
import type { Job, ManagedCapture } from "./operations-api";

export interface ResearchModel {
  id: string;
  name: string;
  version: string;
  lifecycle: string;
  status: string;
  available: boolean;
  classLabels: string[];
  artifactSha256?: string;
  reason?: string;
  backend?: string;
  preprocessingConfiguration?: Record<string, unknown>;
}
export interface EvaluationMetrics {
  samples: number;
  accuracy: number;
  precision: number;
  recall: number;
  macroF1: number;
  confusionMatrix: number[][];
  perClass: { className: string; precision: number; recall: number; f1: number; support: number }[];
  snrAccuracy: { snrDb: number; samples: number; accuracy: number }[];
  calibration: Record<string, unknown>;
}
export interface EvaluationSummary {
  jobId: string;
  datasetId: string;
  datasetHash: string;
  datasetName: string;
  datasetSource: string;
  acquisitionType: string;
  modelId: string;
  modelVersion: string;
  modelHash: string;
  modelStatus: string;
  createdAt: string;
  classOrder: string[];
  metrics: EvaluationMetrics;
  evaluationProtocol: {
    protocolId: string;
    datasetHash: string;
    split: string;
    testWindowsHash: string;
  };
  latency: { inferenceMedianMs: number | null; inferenceP95Ms?: number | null };
  preprocessingConfiguration: Record<string, unknown>;
  modelName?: string;
  hardware?: Record<string, unknown>;
}
export interface ResearchEvaluationReport {
  modelId: string;
  modelVersion: string;
  modelHash: string;
  classOrder: string[];
  dataset: { id: string; name: string; hash: string };
  metrics: EvaluationMetrics;
  preprocessingConfiguration: Record<string, unknown>;
  latency: { inferenceMedianMs: number; inferenceP95Ms: number };
}
export interface TrainingExportManifest {
  schemaVersion: string;
  datasetId: string;
  datasetHash: string;
  sampleCount: number;
  artifact: { filename: string; bytes: number; sha256: string };
  offlineCommand: string;
  experimentConfig: Record<string, unknown>;
  snr: { knownSamples: number; unavailableSamples: number };
  splits: Record<string, { samples: number; groups: number }>;
}
export interface AttributionDiagnostics {
  channels: Record<"i" | "q", { positiveSum: number; negativeSum: number; absoluteSum: number }>;
  importantRegions: {
    startSample: number;
    endSample: number;
    absoluteImportance: number;
    shareOfAbsoluteImportance: number;
  }[];
  topSamples: { sample: number; importance: number }[];
  quantityDescription: string;
  regionRule: string;
}
export interface SavedExplanation extends AttributionResult {
  quantityUnit: string;
  modelHash: string;
  inputHash: string;
  createdAt: string;
  diagnostics: AttributionDiagnostics;
}
export interface ExplanationReport {
  reportVersion: string;
  analysisId: string;
  modelId: string;
  modelVersion: string;
  modelHash: string;
  inputHash: string;
  createdAt: string;
  prediction: {
    predictedModulation: Modulation;
    confidence: number;
    probabilities: Record<Modulation, number>;
    entropyBits: number;
  };
  explanations: SavedExplanation[];
  methodComparison: MethodRankComparison;
  methodComparisons?: MethodRankComparison[];
}
export interface MethodRankComparison {
  status: string;
  targetClass?: Modulation | null;
  spearmanRho: number | null;
  note: string;
}
interface ComparisonPrediction {
  status: string;
  predictedModulation?: string;
  confidence?: number;
  entropyBits?: number;
  processingMs?: number;
  reason?: string;
}
interface ComparisonSignal {
  captureId: string;
  start: number;
  length: number;
  metadata: ManagedCapture;
  inputHash: string;
  analysis: SignalAnalysis;
  prediction: ComparisonPrediction;
}
interface ComparedPlot {
  waveform: { x: number; i: number; q: number }[];
  constellation: IQSample[];
  spectrum: { frequency: number; psd: number }[];
  phase: { x: number; phaseRad: number | null }[];
}
export interface ManagedComparisonReport {
  id: string;
  reportVersion: string;
  createdAt: string;
  first: ComparisonSignal;
  second: ComparisonSignal;
  sampleRates: { first: number | null; second: number | null };
  normalizedAxes: boolean;
  ratesEqual: boolean;
  confidenceDifference: number | null;
  powerDifferenceDb: number | null;
  sampleMse: number | null;
  note?: string;
  limitations: string[];
  plots: {
    first: ComparedPlot;
    second: ComparedPlot;
    timeUnit: string;
    frequencyUnit: string;
    psdUnit: string;
    amplitudeDistribution: {
      first: { value: number; fraction: number }[];
      second: { value: number; fraction: number }[];
      note: string;
    };
  };
  attributionDifferences: {
    status: string;
    reason?: string;
    note: string;
    comparisons: {
      method: string;
      targetClass: string;
      quantityUnit: string;
      meanAbsoluteDifference: number;
      samples: { sample: number; i: number; q: number }[];
    }[];
  };
}
export interface TimeFrequencyReport {
  frequencies: number[];
  times: number[];
  db: number[][];
  frequencyUnit: string;
  timeUnit: string;
  densityUnit: string;
  parameters: { fftSize: number; overlap: number; window: string };
  note: string;
  peaks: { frequency: number; densityDb: number }[];
}
const base = "/api/research";
export const researchApi = {
  models: () => request<ResearchModel[]>(`${base}/models`),
  evaluate: (datasetId: string, modelId: string, modelVersion: string) =>
    request<Job>(`${base}/evaluations`, {
      method: "POST",
      body: JSON.stringify({ datasetId, modelId, modelVersion }),
    }),
  evaluations: (datasetId = "", offset = 0) =>
    request<{ items: EvaluationSummary[]; total: number; note: string }>(
      `${base}/evaluations?offset=${offset}${datasetId ? `&datasetId=${encodeURIComponent(datasetId)}` : ""}`,
    ),
  evaluationResult: (id: string) =>
    request<ResearchEvaluationReport>(`/api/workspace/jobs/${id}/result`),
  compareEvaluations: (jobIds: string[]) =>
    request<{ evaluations: EvaluationSummary[]; note: string }>(`${base}/evaluations/compare`, {
      method: "POST",
      body: JSON.stringify({ jobIds }),
    }),
  explanations: (id: string) => request<ExplanationReport>(`${base}/analyses/${id}/explanations`),
  saveExplanation: (
    id: string,
    options: { method: string; targetClass: Modulation; steps: number; samples?: IQSample[] },
  ) =>
    request<ExplanationReport>(`${base}/analyses/${id}/explanations`, {
      method: "POST",
      body: JSON.stringify(options),
    }),
  compareCaptures: (body: {
    first: { captureId: string; start: number; length: number };
    second: { captureId: string; start: number; length: number };
    firstAnalysisId?: string;
    secondAnalysisId?: string;
  }) =>
    request<ManagedComparisonReport>(`${base}/comparisons`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  comparison: (id: string) => request<ManagedComparisonReport>(`${base}/comparisons/${id}`),
  comparisons: () =>
    request<{ items: { id: string; createdAt: string; source?: string }[] }>(`${base}/comparisons`),
  deleteComparison: (id: string) => request(`${base}/comparisons/${id}`, { method: "DELETE" }),
  trainingManifest: (id: string, model: string, seed: number, allowSmall: boolean) =>
    request<TrainingExportManifest>(
      `${base}/datasets/${id}/training-manifest?model=${model}&seed=${seed}&allowSmall=${allowSmall}`,
    ),
  timeFrequency: (
    samples: IQSample[],
    sampleRate: number,
    fftSize: number,
    overlap: number,
    window: string,
    sampleRateKnown?: boolean,
  ) =>
    request<TimeFrequencyReport>(`${base}/time-frequency`, {
      method: "POST",
      body: JSON.stringify({
        samples,
        sampleRate,
        fftSize,
        overlap,
        window,
        ...(sampleRateKnown !== undefined ? { sampleRateKnown } : {}),
      }),
    }),
};

export async function researchDownload(path: string, filename: string) {
  const response = await fetch(`${apiBase()}${base}${path}`, {
    headers: { "X-Satmod-Token": sessionStorage.getItem("satmod-access-token:" + apiBase()) ?? "" },
    signal: AbortSignal.timeout(60000),
  });
  if (!response.ok) {
    let detail = `Export failed: HTTP ${response.status}`;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* Keep the HTTP status. */
    }
    throw new Error(detail);
  }
  downloadBlob(await response.blob(), filename);
}
