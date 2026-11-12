// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ManagedComparison } from "@/components/managed-comparison";
import { operationsApi } from "@/lib/operations-api";
import { researchApi } from "@/lib/research-api";

vi.mock("@/components/scientific-chart", () => ({ ScientificChart: () => <div>Computed plot</div> }));

const clients: QueryClient[] = [];
afterEach(() => {
  cleanup();
  clients.forEach((client) => client.clear());
  clients.length = 0;
  vi.restoreAllMocks();
});

describe("Saved comparison selection", () => {
  it("captures the selected UUID before an asynchronous mutation resets the controlled select", async () => {
    const id = "2d1b218c-d90e-4f48-b8cb-d815c2874b11";
    vi.spyOn(operationsApi, "captures").mockResolvedValue({ items: [], total: 0, limit: 50, offset: 0 });
    vi.spyOn(researchApi, "comparisons").mockResolvedValue({
      items: [{ id, createdAt: "2026-01-01T00:00:00Z", source: "Generated saved comparison fixture" }],
    });
    const load = vi.spyOn(researchApi, "comparison").mockRejectedValue(new Error("Selected comparison read failed"));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    clients.push(client);
    render(<QueryClientProvider client={client}><ManagedComparison /></QueryClientProvider>);
    fireEvent.click(screen.getByText("Managed recording comparison workstation"));
    await screen.findByRole("option", { name: /Generated saved comparison fixture/ });
    fireEvent.change(screen.getByRole("combobox", { name: /^Inspect saved comparison$/ }), { target: { value: id } });
    await waitFor(() => expect(load).toHaveBeenCalledWith(id));
    expect(load).toHaveBeenCalledTimes(1);
    await screen.findByRole("alert");
    expect(screen.getByRole("alert")).toHaveTextContent("Selected comparison read failed");
  });
});
