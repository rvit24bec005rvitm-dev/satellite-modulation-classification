import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

function syntheticBytes(count = 4096, dc = 0) {
  const bytes = Buffer.alloc(count * 8);
  for (let index = 0; index < count; index++) {
    bytes.writeFloatLE(dc + .006 * Math.cos(2 * Math.PI * index / 8), index * 8);
    bytes.writeFloatLE(.006 * Math.sin(2 * Math.PI * index / 8), index * 8 + 4);
  }
  return bytes;
}

async function storedSyntheticCapture(request: APIRequestContext, source: string, dc = 0) {
  const response = await request.post("/api/workspace/captures", { multipart: {
    file: { name: "synthetic-workspace-completion.bin", mimeType: "application/octet-stream", buffer: syntheticBytes(4096, dc) },
    options: JSON.stringify({ format: "raw", datatype: "cf32_le", sampleRate: 8000, sessionId: source, source }),
  } });
  expect(response.ok(), await response.text()).toBeTruthy();
  return response.json() as Promise<{ id: string }>;
}

async function external(page: Page) {
  await page.locator("summary").filter({ hasText: "External recordings" }).click();
}

async function configureRawImport(page: Page, source: string) {
  await page.getByRole("combobox", { name: "External import format", exact: true }).selectOption("raw");
  await page.getByLabel("External recording file", { exact: true }).setInputFiles({ name: source + ".bin", mimeType: "application/octet-stream", buffer: syntheticBytes() });
  await page.getByRole("combobox", { name: "Binary datatype and endianness", exact: true }).selectOption("cf32_le");
  await page.getByLabel("Recording sample rate (Hz)", { exact: true }).fill("16000");
  await page.getByLabel("Recording session / group ID", { exact: true }).fill(source);
}

async function submitImport(page: Page) {
  const submitted = page.waitForResponse((response) => response.request().method() === "POST" && new URL(response.url()).pathname === "/api/workspace/captures");
  await page.getByRole("button", { name: "Import external recording", exact: true }).click();
  const response = await submitted;
  expect(response.status(), await response.text()).toBe(202);
  await expect(page.getByText(/^Job completed/)).toBeVisible({ timeout: 20000 });
  return response;
}

test("switching away from SigMF imports real raw data without its hidden invalid sidecar", async ({ page, request }) => {
  await page.goto("/classifier");
  await external(page);
  await page.getByRole("combobox", { name: "External import format", exact: true }).selectOption("sigmf");
  await page.getByLabel("SigMF metadata file", { exact: true }).setInputFiles({ name: "synthetic-invalid.sigmf-meta", mimeType: "application/json", buffer: Buffer.from("Intentionally invalid synthetic metadata fixture") });
  await configureRawImport(page, "synthetic-format-switch-completion");
  const submission = await submitImport(page);
  const job = await request.get(`/api/workspace/jobs/${(await submission.json()).id}`);
  expect(job.ok()).toBeTruthy();
  expect((await job.json()).params.sigmf).toBeNull();
  const id = await page.getByRole("combobox", { name: "Stored capture", exact: true }).inputValue();
  expect(id).toBeTruthy();
  const saved = await request.get(`/api/workspace/captures/${id}`);
  expect(saved.ok()).toBeTruthy();
  expect((await saved.json()).metadata).toMatchObject({ format: "raw", datatype: "cf32_le", sampleRate: 16000, sampleCount: 4096 });
});

test("a completed new import clears the previous recording's actual diagnostics and activity", async ({ page, request }) => {
  const first = await storedSyntheticCapture(request, "Synthetic completion diagnostics fixture; not satellite RF", .012);
  await page.goto("/classifier");
  await external(page);
  const selector = page.getByRole("combobox", { name: "Stored capture", exact: true });
  await selector.selectOption(first.id);
  const measured = page.waitForResponse((response) => response.url().includes(`/captures/${first.id}/diagnostics?`));
  await page.getByRole("button", { name: "Run quality diagnostics", exact: true }).click();
  const diagnostics = await measured;
  expect(diagnostics.ok()).toBeTruthy();
  expect((await diagnostics.json()).dcOffset.i).toBeCloseTo(.012, 7);
  await expect(page.getByText("DC I", { exact: true })).toBeVisible();
  const activity = page.waitForResponse((response) => response.url().includes(`/captures/${first.id}/activity?`));
  await page.getByRole("button", { name: "Detect signal activity", exact: true }).click();
  const detected = await activity;
  expect(detected.ok()).toBeTruthy();
  const report = await detected.json();
  expect(report.segments.length).toBeGreaterThan(0);
  await expect(page.getByText(report.note, { exact: true })).toBeVisible();
  await configureRawImport(page, "synthetic-new-context-completion");
  await submitImport(page);
  await expect(selector).not.toHaveValue(first.id);
  await expect(page.getByText("DC I", { exact: true })).toHaveCount(0);
  await expect(page.getByText(report.note, { exact: true })).toHaveCount(0);
});

test("a real completed batch keeps every prediction export action inside the mobile viewport", async ({ page, request }) => {
  const capture = await storedSyntheticCapture(request, "Synthetic completion loaded mobile fixture; not satellite RF");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/classifier");
  await external(page);
  await page.getByRole("combobox", { name: "Stored capture", exact: true }).selectOption(capture.id);
  await page.getByLabel("Batch window count", { exact: true }).fill("2");
  const submission = page.waitForResponse((response) => response.request().method() === "POST" && new URL(response.url()).pathname === "/api/workspace/jobs/batch");
  await page.getByRole("button", { name: "Start batch classification", exact: true }).click();
  const started = await submission;
  expect(started.ok()).toBeTruthy();
  const id = (await started.json()).id;
  await expect(page.getByText(/^Job completed/)).toBeVisible({ timeout: 20000 });
  const result = await request.get(`/api/workspace/jobs/${id}/result`);
  expect(result.ok()).toBeTruthy();
  const actual = await result.json();
  expect(actual.predictions).toHaveLength(2);
  expect(actual.predictions.every((prediction: { captureId: string }) => prediction.captureId === capture.id)).toBe(true);
  await expect(page.getByRole("button", { name: "Export batch PDF summary", exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  for (const name of ["Previous prediction page", "Next prediction page", "Export prediction page", "Export full batch JSON", "Export full batch CSV", "Export batch PDF summary"]) {
    const bounds = await page.getByRole("button", { name, exact: true }).boundingBox();
    expect(bounds, name).not.toBeNull();
    expect(bounds!.x, name).toBeGreaterThanOrEqual(0);
    expect(bounds!.x + bounds!.width, name).toBeLessThanOrEqual(390);
  }
});
