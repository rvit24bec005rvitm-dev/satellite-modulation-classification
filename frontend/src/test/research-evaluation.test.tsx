// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ExternalEvaluation } from "@/components/external-evaluation";
import { MatchedEvaluations } from "@/components/matched-evaluations";
import { TrainingDatasetExport } from "@/components/training-dataset-export";
import { operationsApi, type DatasetRow, type Job } from "@/lib/operations-api";
import {
  researchApi,
  researchDownload,
  type EvaluationMetrics,
  type EvaluationSummary,
  type ResearchEvaluationReport,
  type TrainingExportManifest,
} from "@/lib/research-api";
import { downloadJson } from "@/lib/satmod-api";

vi.mock("@/lib/research-api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/research-api")>()),
  researchDownload: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@/lib/satmod-api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/satmod-api")>()),
  downloadJson: vi.fn(),
}));

const dataset: DatasetRow = {
  id: "9f41c1f7-b8b2-4bb5-9e9b-c095051cde01",
  name: "Unit controlled source",
  source: "Mock UI fixture",
  hash: "a".repeat(64),
  entries: [{ label: "BPSK" }],
  note: "Importer declared labels",
};
const classes = ["BPSK", "QPSK", "16-QAM"];
const metrics: EvaluationMetrics = {
  samples: 3,
  accuracy: 2 / 3,
  precision: 0.6,
  recall: 2 / 3,
  macroF1: 0.62,
  confusionMatrix: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
  ],
  perClass: classes.map((className) => ({
    className,
    support: 1,
    precision: 0.6,
    recall: 0.6,
    f1: 0.6,
  })),
  snrAccuracy: [],
  calibration: { ece: 0.15, nll: 0.6, brier: 0.2 },
};
const report: ResearchEvaluationReport = {
  modelId: "rf-experimental",
  modelVersion: "controlled-synthetic-rf-v1",
  modelHash: "b".repeat(64),
  classOrder: classes,
  dataset: { id: dataset.id, name: dataset.name, hash: dataset.hash },
  metrics,
  preprocessingConfiguration: { mean: 0.00003, std: 0.006 },
  latency: { inferenceMedianMs: 2.1, inferenceP95Ms: 2.5 },
};
const manifest: TrainingExportManifest = {
  schemaVersion: "satmod-training-export-v1",
  datasetId: dataset.id,
  datasetHash: dataset.hash,
  sampleCount: 27,
  artifact: { filename: `${dataset.id}-training.npz`, bytes: 12345, sha256: "c".repeat(64) },
  experimentConfig: { model: "random-forest", seed: 42 },
  offlineCommand:
    "python scripts/train.py --dataset registry.npz --config experiment-config.json --output experiments/runs/new --allow-small-dataset",
  snr: { knownSamples: 0, unavailableSamples: 27 },
  splits: {
    train: { samples: 9, groups: 3 },
    validation: { samples: 9, groups: 3 },
    test: { samples: 9, groups: 3 },
  },
};
const clients: QueryClient[] = [];
function mount(component: React.ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  clients.push(client);
  return render(<QueryClientProvider client={client}>{component}</QueryClientProvider>);
}
function summary(jobId: string, modelId: string, protocolId = "protocol-A"): EvaluationSummary {
  return {
    jobId,
    datasetId: dataset.id,
    datasetHash: dataset.hash,
    datasetName: dataset.name,
    datasetSource: dataset.source,
    acquisitionType: "Mock controlled fixture",
    modelId,
    modelVersion: modelId + "-version",
    modelHash: "d".repeat(64),
    modelStatus: "Experimental",
    createdAt: "2026-01-01T00:00:00Z",
    classOrder: classes,
    metrics,
    evaluationProtocol: {
      protocolId,
      datasetHash: dataset.hash,
      split: "test",
      testWindowsHash: "e".repeat(64),
    },
    latency: { inferenceMedianMs: 1.3 },
    preprocessingConfiguration: { mean: 0, std: 0.006 },
  };
}
beforeEach(() => {
  vi.spyOn(operationsApi, "datasets").mockResolvedValue({
    items: [dataset],
    total: 1,
    offset: 0,
    limit: 50,
    references: [],
  });
  vi.spyOn(researchApi, "models").mockResolvedValue([
    {
      id: "1d-cnn",
      name: "Production CNN",
      version: "supplied-cnn-v1",
      lifecycle: "Production",
      status: "Production",
      available: true,
      classLabels: classes,
      artifactSha256: "f".repeat(64),
    },
    {
      id: "rf-experimental",
      name: "Experimental Random Forest",
      version: "controlled-synthetic-rf-v1",
      lifecycle: "Experimental",
      status: "Experimental",
      available: true,
      classLabels: classes,
      artifactSha256: "b".repeat(64),
    },
    {
      id: "resnet-synthetic-v1",
      name: "Experimental ResNet",
      version: "resnet-synthetic-v1",
      lifecycle: "Experimental",
      status: "Unavailable",
      available: false,
      classLabels: classes,
      reason: "Optional TensorFlow runtime unavailable",
    },
  ]);
});
afterEach(() => {
  cleanup();
  clients.forEach((client) => client.clear());
  clients.length = 0;
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

describe("Genuine research evaluation controls", () => {
  it("selects an actual model/version, disables unavailable checkpoints and shows measured output", async () => {
    let finish!: (job: Job) => void;
    const evaluate = vi.spyOn(researchApi, "evaluate").mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    vi.spyOn(operationsApi, "job").mockResolvedValue({
      id: "evaluation-job",
      kind: "evaluation",
      state: "completed",
      progress: 1,
      cancelRequested: false,
    });
    vi.spyOn(researchApi, "evaluationResult").mockResolvedValue(report);
    mount(<ExternalEvaluation />);
    fireEvent.click(screen.getByText(/External labeled evaluation and/));
    await screen.findByRole("option", { name: /Experimental Random Forest/ });
    expect(screen.getByRole("option", { name: /Experimental ResNet/ })).toBeDisabled();
    fireEvent.change(screen.getByRole("combobox", { name: "External evaluation model" }), {
      target: { value: "rf-experimental" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Evaluate held-out windows" }));
    await waitFor(() =>
      expect(evaluate).toHaveBeenCalledWith(
        dataset.id,
        "rf-experimental",
        "controlled-synthetic-rf-v1",
      ),
    );
    expect(screen.getByRole("button", { name: "Evaluate held-out windows" })).toBeDisabled();
    finish({
      id: "evaluation-job",
      kind: "evaluation",
      state: "queued",
      progress: 0,
      cancelRequested: false,
    });
    await screen.findByText(/Actual evaluation: rf-experimental · controlled-synthetic-rf-v1/);
    expect(screen.getByText(/no SNR metadata was supplied/)).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "True / predicted" })).toBeInTheDocument();
    expect(screen.getByText(/Model SHA-256 bbbbb/)).toBeInTheDocument();
  });
  it("surfaces selected-model failures without creating result metrics", async () => {
    vi.spyOn(researchApi, "evaluate").mockRejectedValue(new Error("Selected artifact was removed"));
    mount(<ExternalEvaluation />);
    fireEvent.click(screen.getByText(/External labeled evaluation and/));
    const button = await screen.findByRole("button", { name: "Evaluate held-out windows" });
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);
    await screen.findByText("Selected artifact was removed");
    expect(screen.queryByText(/Actual evaluation:/)).not.toBeInTheDocument();
  });
  it("submits only chosen jobs within one displayed held-out protocol", async () => {
    const items = [
      summary("job-production", "1d-cnn"),
      summary("job-rf", "rf-experimental"),
      summary("job-other", "rf-experimental", "protocol-B"),
    ];
    vi.spyOn(researchApi, "evaluations").mockResolvedValue({
      items,
      total: 3,
      note: "Exact matching test protocol",
    });
    const compare = vi
      .spyOn(researchApi, "compareEvaluations")
      .mockResolvedValue({ evaluations: items.slice(0, 2), note: "Matched observed results" });
    const { container } = mount(<MatchedEvaluations />);
    const details = container.querySelector("details")!;
    details.open = true;
    fireEvent(details, new Event("toggle", { bubbles: true }));
    await screen.findByRole("checkbox", { name: "Compare evaluation job-production" });
    expect(screen.getByRole("button", { name: "Compare selected evaluations" })).toBeDisabled();
    fireEvent.click(screen.getByRole("checkbox", { name: "Compare evaluation job-production" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Compare evaluation job-rf" }));
    fireEvent.click(screen.getByRole("button", { name: "Compare selected evaluations" }));
    await waitFor(() => expect(compare).toHaveBeenCalledWith(["job-production", "job-rf"]));
    await screen.findByText("Matched observed results");
    fireEvent.change(screen.getByRole("combobox", { name: "Matched test-window protocol" }), {
      target: { value: "protocol-B" },
    });
    expect(
      screen.queryByRole("checkbox", { name: "Compare evaluation job-production" }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Compare selected evaluations" })).toBeDisabled();
  });
  it("validates explicit small-mode splits and downloads exactly the validated artifact/config", async () => {
    const validate = vi.spyOn(researchApi, "trainingManifest").mockResolvedValue(manifest);
    mount(<TrainingDatasetExport dataset={dataset} />);
    fireEvent.click(screen.getByText("Offline training dataset export"));
    expect(
      screen.queryByRole("button", { name: "Export validated training NPZ" }),
    ).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: /Allow fewer than 1,000 windows/ }));
    fireEvent.click(screen.getByRole("button", { name: "Validate offline training export" }));
    await waitFor(() =>
      expect(validate).toHaveBeenCalledWith(dataset.id, "random-forest", 42, true),
    );
    fireEvent.click(await screen.findByRole("button", { name: "Export validated training NPZ" }));
    await waitFor(() =>
      expect(researchDownload).toHaveBeenCalledWith(
        `/datasets/${dataset.id}/training.npz?model=random-forest&seed=42&allowSmall=true`,
        manifest.artifact.filename,
      ),
    );
    fireEvent.click(screen.getByRole("button", { name: "Export offline experiment config" }));
    expect(downloadJson).toHaveBeenCalledWith(manifest.experimentConfig, "experiment-config.json");
    fireEvent.change(screen.getByRole("spinbutton", { name: "Offline training seed" }), {
      target: { value: "43" },
    });
    expect(
      screen.queryByRole("button", { name: "Export validated training NPZ" }),
    ).not.toBeInTheDocument();
  });
  it("reports split independence failures without offering a training download", async () => {
    vi.spyOn(researchApi, "trainingManifest").mockRejectedValue(
      new Error("At least nine independent source/session groups are required"),
    );
    mount(<TrainingDatasetExport dataset={dataset} />);
    fireEvent.click(screen.getByText("Offline training dataset export"));
    fireEvent.click(screen.getByRole("button", { name: "Validate offline training export" }));
    await screen.findByText("At least nine independent source/session groups are required");
    expect(
      screen.queryByRole("button", { name: "Export validated training NPZ" }),
    ).not.toBeInTheDocument();
  });
});
