import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { satmodApi, request, type SignalGenerationRequest } from "@/lib/satmod-api";
interface BridgeResponse {
  id: number;
  status: number;
  headers: Record<string, string>;
  body: string;
  error?: string;
  ready?: boolean;
}
let child: ChildProcessWithoutNullStreams;
let nextId = 0;
const pending = new Map<
  number,
  { resolve: (r: BridgeResponse) => void; reject: (e: Error) => void }
>();
beforeAll(async () => {
  const root = fileURLToPath(new URL("../../../", import.meta.url));
  const python =
    process.env["SATMOD_PYTHON"] ??
    (process.platform === "win32"
      ? `${root}.venv\\Scripts\\python.exe`
      : `${root}.venv/bin/python`);
  await new Promise<void>((resolve, reject) => {
    child = spawn(python, ["-u", "scripts/api_test_bridge.py"], { cwd: root });
    child.on("error", reject);
    child.stderr.on("data", () => {});
    child.on("exit", (code) => {
      if (code) reject(new Error(`API bridge exited with ${code}`));
      for (const p of pending.values()) p.reject(new Error("API bridge stopped"));
    });
    createInterface({ input: child.stdout }).on("line", (line) => {
      const response = JSON.parse(line) as BridgeResponse;
      if (response.ready) resolve();
      else {
        const p = pending.get(response.id);
        pending.delete(response.id);
        if (response.error) p?.reject(new Error(response.error));
        else p?.resolve(response);
      }
    });
  });
  // Only the transport changes: multipart bytes and requests reach the actual
  // FastAPI TestClient in another process. No predictions or metrics are mocked.
  vi.stubGlobal("fetch", async (input: string, options: RequestInit = {}) => {
    const req = new Request(new URL(input, "http://satmod.test"), options);
    const id = ++nextId;
    const response = new Promise<BridgeResponse>((resolve, reject) =>
      pending.set(id, { resolve, reject }),
    );
    const body = Buffer.from(await req.arrayBuffer()).toString("base64");
    child.stdin.write(
      JSON.stringify({
        id,
        method: req.method,
        path: new URL(req.url).pathname + new URL(req.url).search,
        headers: Object.fromEntries(req.headers.entries()),
        body,
      }) + "\n",
    );
    const r = await response;
    return new Response(Uint8Array.from(Buffer.from(r.body, "base64")), {
      status: r.status,
      headers: r.headers,
    });
  });
});
afterAll(() => {
  vi.unstubAllGlobals();
  child?.stdin.end();
});
const params: SignalGenerationRequest = {
  modulation: "BPSK",
  snrDb: 18,
  numSamples: 1024,
  sps: 8,
  sampleRate: 1,
  rrcAlpha: 0.35,
  rrcEnabled: true,
  dopplerHz: 0,
  cfoHz: 0,
  phaseNoiseStd: 0,
  ricianEnabled: false,
  ricianK: 0,
  twtaDrive: 0,
  amplitudeRms: 0.008584335,
  seed: 42,
};
describe("TypeScript API client → real Python model and DSP", () => {
  it("loads health and genuine model metadata", async () => {
    expect((await satmodApi.health()).connected).toBe(true);
    const models = await satmodApi.models();
    expect(models[0]?.parameters).toBe(36355);
    expect(models.find((m) => m.id === "1d-cnn")?.status).toBe("available");
    expect(
      models
        .filter((m) => ["resnet", "cldnn", "random-forest"].includes(m.id))
        .every((m) => m.status === "not-trained"),
    ).toBe(true);
  });
  it("workflow 1: uploads a real saved capture and returns probabilities and FFT", async () => {
    const sample = await satmodApi.sample(97);
    const file = new File(
      ["I,Q\n" + sample.samples.map((p) => `${p.i},${p.q}`).join("\n")],
      "capture.csv",
      { type: "text/csv" },
    );
    const result = await satmodApi.upload(file);
    expect(result.predictedModulation).toBe("BPSK");
    expect(Object.values(result.probabilities).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    expect(result.analysis.spectrum).toHaveLength(128);
    expect(result.samples).toEqual(sample.samples);
  });
  it.each(["BPSK", "QPSK"] as const)(
    "workflows 2/3: generates %s and classifies the full capture",
    async (modulation) => {
      const generated = await satmodApi.generate({ ...params, modulation });
      expect(generated.samples).toHaveLength(1024);
      const result = await satmodApi.classify({
        samples: generated.samples,
        sampleRate: generated.sampleRate,
        source: generated.source,
      });
      expect(result.samples).toEqual(generated.samples);
      expect(result.preprocessing["windowLength"]).toBe(128);
      expect(result.entropyBits).toBeGreaterThanOrEqual(0);
      expect(result.confidence).toBeLessThanOrEqual(1);
    },
  );
  it("workflow 4: returns actual evaluation with consistent counts and SNR", async () => {
    const result = await satmodApi.evaluation();
    expect(result.metrics.accuracy).toBeCloseTo(652 / 900);
    expect(result.confusionMatrix.flat().reduce((a, b) => a + b, 0)).toBe(900);
    expect(result.snrAccuracy).toHaveLength(20);
  });
  it("workflow 5: returns measured CNN timings", async () => {
    const result = await satmodApi.benchmark("1d-cnn", 42, 5);
    expect(result.latenciesMs).toHaveLength(5);
    expect(result.meanMs).toBeGreaterThan(0);
    expect(result.p95Ms).toBeGreaterThanOrEqual(result.medianMs);
  });
  it("workflow 7: propagates malformed upload errors through the real parser", async () => {
    await expect(satmodApi.upload(new File(["bad capture"], "invalid.npy"))).rejects.toThrow(
      "Invalid capture",
    );
  });
  it("analyzes and compares the exact generated samples", async () => {
    const generated = await satmodApi.generate(params);
    const a = await satmodApi.analyze(generated);
    expect(a.sampleCount).toBe(1024);
    const result = await satmodApi.compare(generated, generated);
    expect(result.sampleMse).toBe(0);
    expect(result.powerDifferenceDb).toBe(0);
  });
  it("computes actual class-specific attributions with explicit trust status", async () => {
    const sample = await satmodApi.sample(42);
    const result = await satmodApi.classify({ samples: sample.samples });
    expect(result.trust.calibration.status).toBe("not-calibrated");
    expect(result.trust.ood.detected).toBeNull();
    const explained = await satmodApi.explain({
      samples: sample.samples,
      targetClass: "QPSK",
      method: "integrated-gradients",
      steps: 32,
    });
    expect(explained.samples).toHaveLength(128);
    expect(explained.targetClass).toBe("QPSK");
    expect(explained.samples.some((p) => p.importance > 0)).toBe(true);
  });
  it("normalizes comparison axes while preserving unequal raw captures", async () => {
    const a = await satmodApi.sample(42),
      b = await satmodApi.sample(142);
    const compared = await satmodApi.compare(a, { ...b, sampleRate: 8000 });
    expect(compared.sampleMse).toBeNull();
    expect(compared.plots.frequencyUnit).toBe("cycles/sample");
    expect(compared.predictions.first.status).toBe("available");
  });
  it("loads real experimental evidence without claiming production promotion", async () => {
    const reports = await satmodApi.experiments();
    expect(reports.length).toBeGreaterThanOrEqual(2);
    expect(reports[0]?.productionPromoted).toBe(false);
    expect(reports[0]?.protocolId).toBe(reports[1]?.protocolId);
  });
});
