import { test, expect, type APIRequestContext, type Locator, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
import type { IQSample, ClassificationResult } from "../src/lib/satmod-api";
import type { ExplanationReport, ManagedComparisonReport } from "../src/lib/research-api";

interface RecordedFixture {
  id: string;
  source: string;
  sampleCount: number;
  sampleRate: number;
}

async function recordSynthetic(request: APIRequestContext, seed: number, sampleRate: number | null, tag: string) {
  const generated = await request.post("/api/signal/generate", {
    data: { modulation: "QPSK", numSamples: 1024, sampleRate: sampleRate ?? 8000, seed },
  });
  expect(generated.ok()).toBeTruthy();
  const signal = await generated.json() as { samples: IQSample[] };
  const values = new Float32Array(signal.samples.flatMap((p) => [p.i, p.q]));
  const source = `Phase3 synthetic ${tag} seed ${seed}, not real RF`;
  const imported = await request.post("/api/workspace/captures", {
    multipart: {
      file: sampleRate === null
        ? { name: `synthetic-research-${seed}.json`, mimeType: "application/json", buffer: Buffer.from(JSON.stringify(signal.samples)) }
        : { name: `synthetic-research-${seed}.bin`, mimeType: "application/octet-stream", buffer: Buffer.from(values.buffer) },
      options: JSON.stringify({ ...(sampleRate === null ? {} : { format: "raw", datatype: "cf32_le", sampleRate }), sessionId: `synthetic-${tag}-${seed}`, source, license: "Generated browser-test fixture" }),
    },
  });
  expect(imported.ok()).toBeTruthy();
  return await imported.json() as RecordedFixture;
}

async function classifyRecorded(request: APIRequestContext, captureId: string, start = 0) {
  const classified = await request.post(`/api/workspace/captures/${captureId}/classify?start=${start}`);
  expect(classified.ok()).toBeTruthy();
  return await classified.json() as ClassificationResult;
}

async function saveMethods(request: APIRequestContext, analysisId: string) {
  const response = await request.post(`/api/research/analyses/${analysisId}/explanations`, {
    data: { method: "both", targetClass: "QPSK", steps: 16 },
  });
  expect(response.ok()).toBeTruthy();
  return await response.json() as ExplanationReport;
}

async function downloaded(page: Page, button: Locator) {
  const event = page.waitForEvent("download");
  await button.click();
  const download = await event;
  expect(await download.failure()).toBeNull();
  const path = await download.path();
  if (!path) throw new Error("The completed technical export has no downloaded file.");
  return readFile(path);
}

async function openComparison(page: Page) {
  await page.goto("/intelligence");
  const summary = page.locator("summary").filter({ hasText: /^Managed recording comparison workstation$/ });
  await summary.click();
  return summary.locator("..");
}

test("managed live explanation persists both real methods and full history JSON/PDF after reload", async ({ page, request }) => {
  const capture = await recordSynthetic(request, 831, 8000, `explanation-${Date.now()}`);
  await page.goto("/classifier");
  const summary = page.locator("summary").filter({ hasText: /^External recordings/ });
  await summary.click();
  const workspace = summary.locator("..");
  await workspace.getByRole("combobox", { name: "Stored capture", exact: true }).selectOption(capture.id);
  await workspace.getByRole("spinbutton", { name: "External window start", exact: true }).fill("128");
  const classifiedResponse = page.waitForResponse((r) => r.request().method() === "POST" && r.url().includes(`/captures/${capture.id}/classify?start=128`));
  await workspace.getByRole("button", { name: "Classify external window", exact: true }).click();
  const classified = await classifiedResponse;
  expect(classified.ok()).toBeTruthy();
  const analysis = await classified.json() as ClassificationResult;
  await workspace.locator("summary").filter({ hasText: /^Save explanations and technical PDF$/ }).click();
  const panel = workspace.getByRole("region", { name: "Persistent explanation workspace", exact: true });
  await panel.getByRole("combobox", { name: "Saved explanation target class", exact: true }).selectOption("QPSK");
  await panel.getByRole("spinbutton", { name: "Saved explanation integration steps", exact: true }).fill("16");
  const explainedResponse = page.waitForResponse((r) => r.request().method() === "POST" && r.url().includes(`/analyses/${analysis.id}/explanations`));
  await panel.getByRole("button", { name: "Compute and save both explanation methods", exact: true }).click();
  const explained = await explainedResponse;
  expect(explained.ok()).toBeTruthy();
  let report = await explained.json() as ExplanationReport;
  expect(report.analysisId).toBe(analysis.id);
  expect(report.modelHash).toMatch(/^[a-f0-9]{64}$/);
  expect(report.prediction.probabilities).toEqual(analysis.probabilities);
  expect(report.explanations).toHaveLength(2);
  expect(report.explanations.map((value) => value.method).sort()).toEqual(["integrated-gradients", "saliency"]);
  for (const explanation of report.explanations) {
    expect(explanation.targetClass).toBe("QPSK");
    expect(explanation.samples).toHaveLength(128);
    expect(explanation.samples.every((p) => [p.i, p.q, p.rawI, p.rawQ, p.importance].every(Number.isFinite))).toBeTruthy();
    expect(explanation.inputHash).toBe(report.inputHash);
    expect(explanation.modelHash).toBe(report.modelHash);
  }
  const integrated = report.explanations.find((value) => value.method === "integrated-gradients");
  if (!integrated) throw new Error("The actual Integrated Gradients response is missing.");
  expect(integrated.attributionSum).not.toBeNull();
  expect(integrated.completenessError).not.toBeNull();
  expect(integrated.samples.reduce((sum, p) => sum + p.i + p.q, 0)).toBeCloseTo(integrated.attributionSum!, 6);
  await panel.getByRole("combobox", { name: "Saved explanation view", exact: true }).selectOption(`${integrated.method}:${integrated.targetClass}:${integrated.createdAt}`);
  await expect(panel.getByRole("group", { name: "Signed I/Q class-logit attribution", exact: true })).toHaveAttribute("aria-busy", "false");
  await expect(panel.getByText("Completeness residual", { exact: true })).toBeVisible();
  await expect(panel.getByText(/Descriptive ranking comparison only/)).toBeVisible();
  await panel.getByRole("combobox", { name: "Saved explanation target class", exact: true }).selectOption("BPSK");
  const secondTarget = page.waitForResponse((r) => r.request().method() === "POST" && r.url().includes(`/analyses/${analysis.id}/explanations`));
  await panel.getByRole("button", { name: "Compute and save both explanation methods", exact: true }).click();
  const secondTargetResponse = await secondTarget;
  expect(secondTargetResponse.ok()).toBeTruthy();
  report = await secondTargetResponse.json() as ExplanationReport;
  expect(report.explanations).toHaveLength(4);
  expect(report.methodComparison.targetClass).toBe("BPSK");
  expect(report.methodComparisons?.find((value) => value.targetClass === "QPSK")?.status).toBe("available");
  await panel.getByRole("combobox", { name: "Saved explanation view", exact: true }).selectOption(`${integrated.method}:${integrated.targetClass}:${integrated.createdAt}`);
  await expect(panel.getByText(/Method rank correlation for QPSK:/)).toBeVisible();
  await expect(panel.getByText(/Method rank correlation for BPSK:/)).toHaveCount(0);
  const exported = JSON.parse((await downloaded(page, panel.getByRole("button", { name: "Export full-resolution explanations JSON", exact: true }))).toString("utf8")) as ExplanationReport;
  expect(exported).toEqual(report);
  const pdf = await downloaded(page, panel.getByRole("button", { name: "Download saved explanation PDF", exact: true }));
  expect(pdf.subarray(0, 4).toString()).toBe("%PDF");
  expect(pdf.includes(Buffer.from(report.modelHash))).toBeTruthy();
  expect(pdf.includes(Buffer.from("SIGNED I/Q CLASS-LOGIT ATTRIBUTION"))).toBeTruthy();

  await page.reload();
  const historySummary = page.locator("summary").filter({ hasText: /^Persistent analysis history and technical reports$/ });
  await historySummary.click();
  const history = historySummary.locator("..");
  await history.getByRole("textbox", { name: "Search analysis history", exact: true }).fill(capture.source);
  const row = history.getByRole("row").filter({ hasText: capture.source });
  await expect(row).toHaveCount(1);
  const persistedResponse = page.waitForResponse((r) => r.request().method() === "GET" && r.url().includes(`/analyses/${analysis.id}/explanations`));
  await row.getByRole("button", { name: "Inspect saved explanations", exact: true }).click();
  expect(await (await persistedResponse).json()).toEqual(report);
  const restored = history.getByRole("region", { name: "Persistent explanation workspace", exact: true });
  await expect(restored.getByRole("combobox", { name: "Saved explanation view", exact: true }).locator("option")).toHaveCount(4);
  const restoredJson = JSON.parse((await downloaded(page, restored.getByRole("button", { name: "Export full-resolution explanations JSON", exact: true }))).toString("utf8")) as ExplanationReport;
  expect(restoredJson).toEqual(report);
  expect((await downloaded(page, restored.getByRole("button", { name: "Download saved explanation PDF", exact: true }))).equals(pdf)).toBeTruthy();
  await page.screenshot({ path: `${process.env["SATMOD_SCREENSHOT_DIR"] ?? "../integration/phase3/screenshots"}/research-explanation-history.png`, fullPage: true });
});

test("managed A/B workstation preserves real predictions and normalized PSD across unequal sample clocks", async ({ page, request }) => {
  const tag = `unequal-clocks-${Date.now()}`;
  const a = await recordSynthetic(request, 841, 8000, tag);
  const b = await recordSynthetic(request, 842, 16000, tag);
  const unknown = await recordSynthetic(request, 843, null, `${tag}-clock-metadata-omitted`);
  const analysisA = await classifyRecorded(request, a.id, 128);
  const analysisB = await classifyRecorded(request, b.id, 256);
  const workspace = await openComparison(page);
  await workspace.getByRole("combobox", { name: "Comparison recording A", exact: true }).selectOption(a.id);
  await workspace.getByRole("combobox", { name: "Comparison recording B", exact: true }).selectOption(b.id);
  await workspace.getByRole("spinbutton", { name: "Recording A sample offset", exact: true }).fill("128");
  await workspace.getByRole("spinbutton", { name: "Recording B sample offset", exact: true }).fill("256");
  await workspace.getByRole("spinbutton", { name: "Comparison window length", exact: true }).fill("256");
  await workspace.locator("summary").filter({ hasText: /^Reuse saved predictions \/ attribution/ }).click();
  await workspace.getByRole("textbox", { name: "Recording A analysis ID", exact: true }).fill(analysisA.id);
  await workspace.getByRole("textbox", { name: "Recording B analysis ID", exact: true }).fill(analysisB.id);
  const completed = page.waitForResponse((r) => r.request().method() === "POST" && new URL(r.url()).pathname === "/api/research/comparisons");
  await workspace.getByRole("button", { name: "Compare managed windows", exact: true }).click();
  const response = await completed;
  expect(response.ok()).toBeTruthy();
  const report = await response.json() as ManagedComparisonReport;
  expect(report.first.captureId).toBe(a.id);
  expect(report.second.captureId).toBe(b.id);
  expect(report.first.start).toBe(128);
  expect(report.second.start).toBe(256);
  expect(report.first.length).toBe(256);
  expect(report.sampleRates).toEqual({ first: 8000, second: 16000 });
  expect(report.ratesEqual).toBe(false);
  expect(report.normalizedAxes).toBe(true);
  expect(report.sampleMse).toBeNull();
  expect(report.first.prediction).toMatchObject({ status: "available", predictedModulation: analysisA.predictedModulation });
  expect(report.second.prediction).toMatchObject({ status: "available", predictedModulation: analysisB.predictedModulation });
  expect(report.plots.frequencyUnit).toBe("cycles/sample");
  expect(report.attributionDifferences.status).toBe("unavailable");
  await expect(workspace.getByRole("region", { name: "Stored comparison result", exact: true })).toBeVisible();
  await expect(workspace.getByText("Normalized sample/frequency axes", { exact: true })).toBeVisible();
  await expect(workspace.getByText("Different supplied rates", { exact: true })).toBeVisible();
  await expect(workspace.getByRole("group", { name: "Matched-axis spectral comparison", exact: true })).toHaveAttribute("aria-busy", "false");
  const exported = JSON.parse((await downloaded(page, workspace.getByRole("button", { name: "Export comparison JSON", exact: true }))).toString("utf8")) as ManagedComparisonReport;
  expect(exported).toEqual(report);
  const pdf = await downloaded(page, workspace.getByRole("button", { name: "Export comparison PDF", exact: true }));
  expect(pdf.subarray(0, 4).toString()).toBe("%PDF");
  expect(pdf.includes(Buffer.from("SPECTRAL DENSITY COMPARISON"))).toBeTruthy();

  await page.reload();
  await page.locator("summary").filter({ hasText: /^Managed recording comparison workstation$/ }).click();
  const persisted = page.waitForResponse((r) => r.request().method() === "GET" && new URL(r.url()).pathname === `/api/research/comparisons/${report.id}`);
  await page.getByRole("combobox", { name: "Inspect saved comparison", exact: true }).selectOption(report.id);
  expect(await (await persisted).json()).toEqual(report);
  await expect(page.getByText("Normalized sample/frequency axes", { exact: true })).toBeVisible();
  await page.screenshot({ path: `${process.env["SATMOD_SCREENSHOT_DIR"] ?? "../integration/phase3/screenshots"}/research-comparison-clock-mismatch.png`, fullPage: true });
  await page.getByRole("combobox", { name: "Comparison recording A", exact: true }).selectOption(unknown.id);
  await page.getByRole("combobox", { name: "Comparison recording B", exact: true }).selectOption(b.id);
  const unknownClock = page.waitForResponse((r) => r.request().method() === "POST" && new URL(r.url()).pathname === "/api/research/comparisons");
  await page.getByRole("button", { name: "Compare managed windows", exact: true }).click();
  const unknownClockResponse = await unknownClock;
  expect(unknownClockResponse.ok()).toBeTruthy();
  const unknownReport = await unknownClockResponse.json() as ManagedComparisonReport;
  expect(unknownReport.sampleRates.first).toBeNull();
  expect(unknownReport.first.metadata.sampleRateKnown).toBe(false);
  await expect(page.getByText("At least one rate unknown", { exact: true })).toBeVisible();
  await expect(page.getByText(/clock synchronization and sample alignment are unverified/)).toBeVisible();
});

test("matching historical analyses supply genuine same-target attribution differences in the comparison workstation", async ({ page, request }) => {
  const tag = `saved-attributions-${Date.now()}`;
  const a = await recordSynthetic(request, 851, 8000, tag);
  const b = await recordSynthetic(request, 852, 8000, tag);
  const analysisA = await classifyRecorded(request, a.id);
  const analysisB = await classifyRecorded(request, b.id, 128);
  const explanationA = await saveMethods(request, analysisA.id);
  const explanationB = await saveMethods(request, analysisB.id);
  const workspace = await openComparison(page);
  await workspace.getByRole("combobox", { name: "Comparison recording A", exact: true }).selectOption(a.id);
  await workspace.getByRole("combobox", { name: "Comparison recording B", exact: true }).selectOption(b.id);
  await workspace.getByRole("spinbutton", { name: "Recording B sample offset", exact: true }).fill("128");
  await workspace.locator("summary").filter({ hasText: /^Reuse saved predictions \/ attribution/ }).click();
  await workspace.getByRole("textbox", { name: "Recording A analysis ID", exact: true }).fill(analysisA.id);
  await workspace.getByRole("textbox", { name: "Recording B analysis ID", exact: true }).fill(analysisB.id);
  const completed = page.waitForResponse((r) => r.request().method() === "POST" && new URL(r.url()).pathname === "/api/research/comparisons");
  await workspace.getByRole("button", { name: "Compare managed windows", exact: true }).click();
  const response = await completed;
  expect(response.ok()).toBeTruthy();
  const report = await response.json() as ManagedComparisonReport;
  expect(report.normalizedAxes).toBe(false);
  expect(report.attributionDifferences.status).toBe("available");
  expect(report.attributionDifferences.comparisons).toHaveLength(2);
  for (const value of report.attributionDifferences.comparisons) {
    expect(value.targetClass).toBe("QPSK");
    expect(value.samples).toHaveLength(128);
    const first = explanationA.explanations.find((e) => e.method === value.method && e.targetClass === value.targetClass);
    const second = explanationB.explanations.find((e) => e.method === value.method && e.targetClass === value.targetClass);
    if (!first || !second) throw new Error("Stored same-method, same-target explanation pair is missing.");
    value.samples.forEach((sample, index) => {
      expect(sample.i).toBeCloseTo(first.samples[index]!.i - second.samples[index]!.i, 9);
      expect(sample.q).toBeCloseTo(first.samples[index]!.q - second.samples[index]!.q, 9);
    });
    await expect(workspace.getByRole("group", { name: `A − B ${value.method}: QPSK`, exact: true })).toHaveAttribute("aria-busy", "false");
  }
  await expect(workspace.getByText("Physical time/frequency axes", { exact: true })).toBeVisible();
  await expect(workspace.getByText("Equal supplied rates", { exact: true })).toBeVisible();
  await expect(workspace.getByText(/No physical sample alignment, resampling or causal interpretation is inferred/)).toBeVisible();
  const exported = JSON.parse((await downloaded(page, workspace.getByRole("button", { name: "Export comparison JSON", exact: true }))).toString("utf8")) as ManagedComparisonReport;
  expect(exported.attributionDifferences).toEqual(report.attributionDifferences);
});

test("history explanations clearly reject legacy analyses whose original samples were not retained", async ({ page, request }) => {
  const generated = await request.post("/api/signal/generate", { data: { modulation: "BPSK", numSamples: 256, seed: 861 } });
  expect(generated.ok()).toBeTruthy();
  const signal = await generated.json() as { samples: IQSample[] };
  const source = `Phase3 synthetic legacy no-retention ${Date.now()}, not RF`;
  const classified = await request.post("/api/classify", { data: { samples: signal.samples, sampleRate: 8000, source } });
  expect(classified.ok()).toBeTruthy();
  const analysis = await classified.json() as ClassificationResult;
  await page.goto("/classifier");
  const historySummary = page.locator("summary").filter({ hasText: /^Persistent analysis history and technical reports$/ });
  await historySummary.click();
  const history = historySummary.locator("..");
  await history.getByRole("textbox", { name: "Search analysis history", exact: true }).fill(source);
  const row = history.getByRole("row").filter({ hasText: source });
  await expect(row).toHaveCount(1);
  await row.getByRole("button", { name: "Inspect saved explanations", exact: true }).click();
  const panel = history.getByRole("region", { name: "Persistent explanation workspace", exact: true });
  const rejected = page.waitForResponse((r) => r.request().method() === "POST" && r.url().includes(`/analyses/${analysis.id}/explanations`));
  await panel.getByRole("button", { name: "Compute and save both explanation methods", exact: true }).click();
  expect((await rejected).status()).toBe(409);
  await expect(panel.getByRole("alert")).toContainText("Original samples were not retained");
  await expect(panel.getByRole("button", { name: "Download saved explanation PDF", exact: true })).toHaveCount(0);
});
