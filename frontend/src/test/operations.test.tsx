// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor, cleanup } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ExternalWorkspace } from "@/components/external-workspace";
import { PersistentHistory } from "@/components/persistent-history";
import { ExternalEvaluation } from "@/components/external-evaluation";
import { operationsApi } from "@/lib/operations-api";
vi.mock("@/components/scientific-chart", () => ({
  ScientificChart: () => <div>Computed chart</div>,
}));
vi.mock("@/hooks/use-satmod", () => ({
  useSignal: () => ({ setCapture: vi.fn(), setClassification: vi.fn() }),
}));
const clients: QueryClient[] = [];
function mount(node: React.ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  clients.push(client);
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>);
}
beforeEach(() => {
  vi.spyOn(operationsApi, "captures").mockResolvedValue({
    items: [],
    total: 0,
    limit: 50,
    offset: 0,
  });
  vi.spyOn(operationsApi, "jobs").mockResolvedValue({ items: [], total: 0, limit: 50, offset: 0 });
  vi.spyOn(operationsApi, "history").mockResolvedValue({
    items: [],
    total: 0,
    limit: 50,
    offset: 0,
  });
  vi.spyOn(operationsApi, "datasets").mockResolvedValue({
    items: [],
    total: 0,
    limit: 50,
    offset: 0,
    references: [],
  });
});
afterEach(() => {
  cleanup();
  clients.forEach((c) => c.clear());
  clients.length = 0;
  vi.restoreAllMocks();
});
describe("Phase2 operational controls", () => {
  it("requires explicit binary encoding, clock and session before import", async () => {
    mount(<ExternalWorkspace />);
    fireEvent.click(screen.getByText(/External recordings ·/));
    fireEvent.change(screen.getByLabelText("External import format"), { target: { value: "raw" } });
    fireEvent.change(screen.getByLabelText("External recording file"), {
      target: { files: [new File([new Uint8Array(1024)], "unknown.bin")] },
    });
    expect(
      screen.getByRole("button", { name: "Import external recording" }),
    ).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Binary datatype and endianness"), {
      target: { value: "cf32_le" },
    });
    fireEvent.change(screen.getByLabelText("Recording sample rate (Hz)"), {
      target: { value: "8000" },
    });
    fireEvent.change(screen.getByLabelText("Recording session / group ID"), {
      target: { value: "session-A" },
    });
    expect(
      screen.getByRole("button", { name: "Import external recording" }),
    ).toBeEnabled();
  });
  it("surfaces connection errors instead of a fabricated capture", async () => {
    vi.mocked(operationsApi.captures).mockRejectedValue(new Error("Backend offline test"));
    mount(<ExternalWorkspace />);
    fireEvent.click(screen.getByText(/External recordings ·/));
    await waitFor(() => expect(screen.getByText("Backend offline test")).toBeInTheDocument());
    expect(screen.queryByText("External prediction")).not.toBeInTheDocument();
  });
  it("shows persistent job state and sends cancellation to its actual ID", async () => {
    vi.mocked(operationsApi.jobs).mockResolvedValue({
      items: [
        {
          id: "job-test-id",
          kind: "batch",
          state: "running",
          progress: 0.5,
          cancelRequested: false,
        },
      ],
      total: 1,
      limit: 50,
      offset: 0,
    });
    const cancel = vi
      .spyOn(operationsApi, "cancel")
      .mockResolvedValue({
        id: "job-test-id",
        kind: "batch",
        state: "running",
        progress: 0.5,
        cancelRequested: true,
      });
    mount(<ExternalWorkspace />);
    fireEvent.click(screen.getByText(/External recordings ·/));
    await waitFor(() => expect(screen.getByText(/batch · running · 50/)).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Cancel job" }));
    await waitFor(() => expect(cancel).toHaveBeenCalledWith("job-test-id"));
  });
  it("passes history search/filter state to the persistent API", async () => {
    mount(<PersistentHistory />);
    fireEvent.click(screen.getByText(/Persistent analysis history and/));
    fireEvent.change(screen.getByLabelText("Search analysis history"), {
      target: { value: "receiver-A" },
    });
    fireEvent.change(screen.getByLabelText("History class filter"), { target: { value: "BPSK" } });
    await waitFor(() =>
      expect(operationsApi.history).toHaveBeenCalledWith("receiver-A", 0, "BPSK"),
    );
  });
  it("rejects invalid evaluation JSON locally without inventing ground truth", async () => {
    const register = vi.spyOn(operationsApi, "register");
    mount(<ExternalEvaluation />);
    fireEvent.click(screen.getByText(/External labeled evaluation and/));
    fireEvent.change(screen.getByLabelText("Labeled dataset manifest JSON"), {
      target: { value: "broken json" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Register labeled dataset" }));
    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(register).not.toHaveBeenCalled();
  });
  it("uploads into a persistent job and selects the capture only after completion", async () => {
    const imported = vi.spyOn(operationsApi, "importJob").mockResolvedValue({
      id: "import-job", kind: "capture-import", state: "queued", progress: 0, cancelRequested: false,
    });
    vi.spyOn(operationsApi, "job").mockResolvedValue({
      id: "import-job", kind: "capture-import", state: "completed", progress: 1,
      captureId: "capture-after-import", phase: "complete", cancelRequested: false,
    });
    const overview = vi.spyOn(operationsApi, "overview").mockRejectedValue(new Error("Overview selected after import"));
    mount(<ExternalWorkspace />);
    fireEvent.click(screen.getByText(/External recordings ·/));
    fireEvent.change(screen.getByLabelText("External recording file"), {
      target: { files: [new File([new Uint8Array(1024)], "fixture.npy")] },
    });
    fireEvent.change(screen.getByLabelText("Recording session / group ID"), { target: { value: "session-C" } });
    fireEvent.click(screen.getByRole("button", { name: "Import external recording" }));
    await waitFor(() => expect(imported).toHaveBeenCalled());
    await waitFor(() => expect(overview).toHaveBeenCalledWith("capture-after-import"));
    expect(screen.getByText(/Job completed · 100.00% complete · complete/)).toBeInTheDocument();
  });
  it("loads the real timeline selection and surfaces its window error", async () => {
    const job = { id: "timeline-job", kind: "batch", state: "completed", progress: 1, cancelRequested: false };
    vi.mocked(operationsApi.jobs).mockResolvedValue({ items: [job], total: 1, limit: 50, offset: 0 });
    vi.spyOn(operationsApi, "job").mockResolvedValue(job);
    vi.spyOn(operationsApi, "result").mockResolvedValue({
      total: 1, classCounts: { BPSK: 1 }, note: "Dependent windows, not independent measurements",
      predictions: [{ index: 3, start: 384, captureId: "timeline-capture", predictedModulation: "BPSK",
        confidence: .6, entropyBits: 1.2, processingMs: 2, timestamp: null,
        probabilities: { BPSK: .6, QPSK: .3, "16-QAM": .1 } }],
    });
    const windowRequest = vi.spyOn(operationsApi, "window").mockRejectedValue(new Error("Selected window unavailable"));
    vi.spyOn(operationsApi, "overview").mockRejectedValue(new Error("Capture offline"));
    mount(<ExternalWorkspace />);
    fireEvent.click(screen.getByText(/External recordings ·/));
    fireEvent.click(await screen.findByText(/batch · completed/));
    await screen.findByRole("cell", { name: "384" });
    fireEvent.click(screen.getByRole("button", { name: /^3$/ }));
    await waitFor(() => expect(windowRequest).toHaveBeenCalledWith("timeline-capture", 384));
    await screen.findByText("Selected window unavailable");
  });
  it("registers a manifest through a tracked dataset import job", async () => {
    const register = vi.spyOn(operationsApi, "registerJob").mockResolvedValue({
      id: "dataset-job", kind: "dataset-import", state: "queued", progress: 0, cancelRequested: false,
    });
    vi.spyOn(operationsApi, "job").mockResolvedValue({
      id: "dataset-job", kind: "dataset-import", state: "running", progress: .5,
      phase: "validating labeled windows", cancelRequested: false,
    });
    mount(<ExternalEvaluation />);
    fireEvent.click(screen.getByText(/External labeled evaluation and/));
    const manifest = { name: "Independent acquisition", entries: [] };
    fireEvent.change(screen.getByLabelText("Labeled dataset manifest JSON"), { target: { value: JSON.stringify(manifest) } });
    fireEvent.click(screen.getByRole("button", { name: "Register labeled dataset" }));
    await waitFor(() => expect(register).toHaveBeenCalledWith(manifest));
    await screen.findByText(/Dataset import: running · 50% validating labeled windows/);
    expect(screen.getByRole("button", { name: "Cancel dataset operation" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Export external evaluation JSON" })).not.toBeInTheDocument();
  });
});
