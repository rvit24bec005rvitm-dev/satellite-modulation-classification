import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { request, satmodApi, apiBase } from "@/lib/satmod-api";
describe("Signal API transport", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    vi.stubGlobal("fetch", fetchMock);
    fetchMock.mockReset();
    const values = new Map<string, string>();
    vi.stubGlobal("window", {});
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
      clear: () => values.clear(),
    });
  });
  afterEach(() => vi.unstubAllGlobals());
  it("uses the configured API address without a trailing slash", async () => {
    localStorage.setItem("satmod-api-url", "http://localhost:8000/");
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ connected: true })));
    expect(apiBase()).toBe("http://localhost:8000");
    await satmodApi.health();
    expect(fetchMock.mock.calls[0]?.[0]).toBe("http://localhost:8000/api/health");
  });
  it("sends full resolution samples with the selected classification window", async () => {
    const samples = Array.from({ length: 1024 }, () => ({ i: 0.01, q: 0.02 }));
    fetchMock.mockResolvedValue(new Response("{}"));
    await satmodApi.classify({ samples, windowStart: 256 });
    const opts = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(JSON.parse(String(opts.body)).samples).toHaveLength(1024);
    expect(JSON.parse(String(opts.body)).windowStart).toBe(256);
  });
  it("uploads a FormData capture without setting a JSON content type", async () => {
    fetchMock.mockResolvedValue(new Response("{}"));
    await satmodApi.upload(new File(["I,Q\n0,0"], "capture.csv"));
    const opts = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(opts.body).toBeInstanceOf(FormData);
    expect(opts.headers).not.toHaveProperty("Content-Type");
  });
  it("shows backend validation errors", async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ detail: "I/Q values must be finite." }), { status: 422 }),
    );
    await expect(satmodApi.upload(new File(["bad"], "bad.npy"))).rejects.toThrow(
      "I/Q values must be finite.",
    );
  });
  it("reports connection failure without manufacturing a prediction", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    await expect(satmodApi.classify({ samples: [] })).rejects.toThrow("Backend connection failed");
  });
  it("preserves caller cancellation", async () => {
    const controller = new AbortController();
    fetchMock.mockRejectedValue(new DOMException("Cancelled", "AbortError"));
    await expect(request("/api/health", { signal: controller.signal })).rejects.toHaveProperty(
      "name",
      "AbortError",
    );
  });
  it("exposes a model-loading failure from health", async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ error: "Model artifact is missing" }), { status: 503 }),
    );
    await expect(satmodApi.health()).rejects.toThrow("Model artifact is missing");
  });
  it("handles an HTML error response", async () => {
    fetchMock.mockResolvedValue(new Response("<h1>Proxy error</h1>", { status: 502 }));
    await expect(satmodApi.health()).rejects.toThrow("HTTP 502");
  });
});
