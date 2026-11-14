// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ExternalWorkspace } from "@/components/external-workspace";
import { operationsApi, type Diagnostics, type Job, type ManagedCapture } from "@/lib/operations-api";

vi.mock("@/components/scientific-chart", () => ({
  ScientificChart: ({ shapes }: { shapes?: unknown[] }) => <div data-testid="computed-plot" data-shapes={JSON.stringify(shapes ?? [])} />,
}));
vi.mock("@/components/classification-insights", () => ({ ClassificationInsights: () => null }));
vi.mock("@/hooks/use-satmod", () => ({ useSignal: () => ({ setCapture: vi.fn(), setClassification: vi.fn() }) }));

const clients: QueryClient[] = [];
const recording = (id: string): ManagedCapture => ({ id, source: "Synthetic recording " + id,
  sampleCount: 1024, sampleRate: 8000, sampleRateKnown: true, createdAt: "2026-11-14T00:00:00Z",
  sessionId: "synthetic-session-" + id, hash: id, sourceHash: id, split: "unassigned", license: "Generated test fixture",
  centerFrequency: null, sourceOffset: 0 });
const completed: Job = { id: "synthetic-import-job", kind: "capture-import", state: "completed",
  progress: 1, cancelRequested: false, captureId: "second" };
const diagnostic: Diagnostics = { dcOffset: { i: .012, q: -.003 }, iqGainImbalanceDb: null,
  iqCorrelation: null, clippedFraction: null, powerCoefficientOfVariation: 0, frequencyOffset: null,
  frequencyUnit: "Hz", welchUnits: "sample units squared / Hz", welch: [], powerOverTime: [],
  clippingNote: "Synthetic first-recording clipping diagnostic", imbalanceNote: "Synthetic first-recording imbalance diagnostic" };

beforeEach(() => {
  vi.spyOn(operationsApi, "captures").mockResolvedValue({ items: [recording("first"), recording("second")], total: 2, limit: 50, offset: 0 });
  vi.spyOn(operationsApi, "overview").mockImplementation(async (id) => ({ metadata: recording(id), points: [{ sample: 0, power: .001 }], note: "Synthetic overview" }));
  vi.spyOn(operationsApi, "jobs").mockResolvedValue({ items: [], total: 0, limit: 50, offset: 0 });
  vi.spyOn(operationsApi, "job").mockResolvedValue(completed);
  vi.spyOn(operationsApi, "importJob").mockResolvedValue(completed);
  vi.spyOn(operationsApi, "diagnostics").mockResolvedValue(diagnostic);
  vi.spyOn(operationsApi, "domain").mockResolvedValue({ note: "Synthetic first-recording domain diagnostics" });
  vi.spyOn(operationsApi, "activity").mockResolvedValue({ segments: [{ startSample: 0, endSample: 512 }], note: "Synthetic first-recording activity" });
});
afterEach(() => { cleanup(); clients.forEach((client) => client.clear()); clients.length = 0; vi.restoreAllMocks(); });

function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><ExternalWorkspace /></QueryClientProvider>);
  fireEvent.click(screen.getByText("External recordings · capture browser, jobs and prediction timeline"));
}
function chooseImportFile() {
  fireEvent.change(screen.getByLabelText("External recording file", { exact: true }), { target: { files: [new File([new Uint8Array(2048)], "synthetic-second.bin")] } });
  fireEvent.change(screen.getByLabelText("Recording session / group ID"), { target: { value: "synthetic-second-session" } });
}

describe("External workspace completion regressions", () => {
  it("clears previous recording diagnostics and activity when a background import selects a new recording", async () => {
    mount();
    await screen.findByRole("option", { name: /^Synthetic recording first/ });
    fireEvent.change(screen.getByLabelText("Stored capture", { exact: true }), { target: { value: "first" } });
    await screen.findByRole("button", { name: "Run quality diagnostics" });
    fireEvent.click(screen.getByRole("button", { name: "Run quality diagnostics" }));
    fireEvent.click(screen.getByRole("button", { name: "Compare demo-domain statistics" }));
    fireEvent.click(screen.getByRole("button", { name: "Detect signal activity" }));
    await screen.findByText(/Synthetic first-recording clipping diagnostic/);
    await screen.findByText("Synthetic first-recording activity");
    expect(screen.getAllByTestId("computed-plot")[0]!.getAttribute("data-shapes")).toContain("rgba(0,220,160,0.12)");
    chooseImportFile();
    fireEvent.click(screen.getByRole("button", { name: /^Import external recording$/ }));
    await waitFor(() => expect(screen.getByLabelText("Stored capture", { exact: true })).toHaveValue("second"));
    await waitFor(() => expect(screen.queryByText(/Synthetic first-recording clipping diagnostic/)).not.toBeInTheDocument());
    expect(screen.queryByText("Synthetic first-recording activity")).not.toBeInTheDocument();
    expect(screen.queryByText(/Synthetic first-recording domain diagnostics/)).not.toBeInTheDocument();
    expect(screen.getByTestId("computed-plot").getAttribute("data-shapes")).not.toContain("rgba(0,220,160,0.12)");
  });

  it.each(["application", "raw"])("does not forward a hidden SigMF sidecar after switching to %s", async (format) => {
    mount();
    chooseImportFile();
    fireEvent.change(screen.getByLabelText("External import format"), { target: { value: "sigmf" } });
    const metadata = new File(["Intentionally invalid synthetic JSON"], "synthetic.sigmf-meta");
    fireEvent.change(screen.getByLabelText("SigMF metadata file", { exact: true }), { target: { files: [metadata] } });
    fireEvent.change(screen.getByLabelText("External import format"), { target: { value: format } });
    if (format === "raw") {
      fireEvent.change(screen.getByLabelText("Binary datatype and endianness"), { target: { value: "cf32_le" } });
      fireEvent.change(screen.getByLabelText("Recording sample rate (Hz)"), { target: { value: "8000" } });
    }
    fireEvent.click(screen.getByRole("button", { name: /^Import external recording$/ }));
    await waitFor(() => expect(operationsApi.importJob).toHaveBeenCalled());
    expect(vi.mocked(operationsApi.importJob).mock.calls[0]?.[1]).toMatchObject({ format });
    expect(vi.mocked(operationsApi.importJob).mock.calls[0]?.[2]).toBeUndefined();
  });
});
