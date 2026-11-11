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
});
