import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";
const group = "phase2-generated-fixture-session";

test("external binary → late window → actual batch timeline → persisted PDF report", async ({
  page,
}) => {
  await page.goto("/classifier");
  await page.locator("summary").filter({ hasText: "External recordings" }).click();
  const floats = Float32Array.from({ length: 131072 * 2 }, (_, i) => 0.006 * Math.sin(i * 0.037));
  await page.getByLabel("External import format").selectOption("raw");
  await page.getByLabel("External recording file", { exact: true }).setInputFiles({
    name: "phase2-external.bin",
    mimeType: "application/octet-stream",
    buffer: Buffer.from(floats.buffer),
  });
  await expect(
    page.getByRole("button", { name: "Import external recording", exact: true }),
  ).toBeDisabled();
  await page.getByLabel("Binary datatype and endianness").selectOption("cf32_le");
  await page.getByLabel("Recording sample rate (Hz)").fill("8000");
  await page.getByLabel("Recording session / group ID").fill(group);
  const importResponse = page.waitForResponse((r) => r.request().method() === "POST" && r.url().includes("/captures?background=true"));
  await page.getByRole("button", { name: "Import external recording", exact: true }).click();
  expect((await importResponse).status()).toBe(202);
  await expect(page.getByText("131072", { exact: true })).toBeVisible({ timeout: 20000 });
  await page.getByLabel("External window start").fill("130560");
  await page.getByLabel("Batch window count").fill("4");
  await page.getByRole("button", { name: "Start batch classification" }).click();
  await expect(page.getByText(/Job completed/)).toBeVisible({ timeout: 30000 });
  await expect(page.getByRole("cell", { name: "130560", exact: true })).toBeVisible();
  const batchDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export full batch CSV" }).click();
  const batchCsv = await readFile(await (await batchDownload).path(), "utf8");
  expect(batchCsv).toContain("rawSoftmax");
  expect(batchCsv).toContain("modelHash");
  expect(batchCsv).toContain("preprocessing");
  expect(batchCsv.trim().split("\n")).toHaveLength(5);
  const batchPdfDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export batch PDF summary" }).click();
  expect((await readFile(await (await batchPdfDownload).path())).subarray(0, 4).toString()).toBe("%PDF");
  const selectedWindow = page.waitForResponse((r) => r.url().includes("/window?start=130688"));
  await page.getByRole("row").filter({ has: page.getByRole("cell", { name: "130688", exact: true }) }).getByRole("button", { name: "1", exact: true }).click();
  expect((await selectedWindow).ok()).toBeTruthy();
  await expect(page.getByLabel("External window start")).toHaveValue("130688");
  await page.getByLabel("External window start").fill("130560");
  const classified = page.waitForResponse(
    (r) => r.url().includes("/api/workspace/captures/") && r.url().includes("/classify"),
  );
  await page.getByRole("button", { name: "Classify external window" }).click();
  const result = await (await classified).json();
  expect(result.capturePosition.sourceSample).toBe(130560);
  expect(["BPSK", "QPSK", "16-QAM"]).toContain(result.predictedModulation);
  await page.locator("summary").filter({ hasText: "Persistent analysis history" }).click();
  await page.getByLabel("Search analysis history").fill("phase2-external.bin");
  const row = page
    .getByRole("row")
    .filter({ hasText: "phase2-external.bin @ sample 130560" })
    .first();
  await expect(row).toBeVisible({ timeout: 15000 });
  const pdfDownload = page.waitForEvent("download");
  await row.getByRole("button", { name: "PDF", exact: true }).click();
  expect((await readFile(await (await pdfDownload).path())).subarray(0, 4).toString()).toBe("%PDF");
});

test("SigMF metadata supplies sample clock and UTC window timestamp", async ({ page }) => {
  await page.goto("/intelligence");
  await page.locator("summary").filter({ hasText: "External recordings" }).click();
  await page.getByLabel("External import format").selectOption("sigmf");
  const raw = new Float32Array(1024 * 2).fill(0.006);
  await page.getByLabel("External recording file", { exact: true }).setInputFiles({
    name: "fixture.sigmf-data",
    mimeType: "application/octet-stream",
    buffer: Buffer.from(raw.buffer),
  });
  await page.getByLabel("SigMF metadata file").setInputFiles({
    name: "fixture.sigmf-meta",
    mimeType: "application/json",
    buffer: Buffer.from(
      JSON.stringify({
        global: {
          "core:datatype": "cf32_le",
          "core:sample_rate": 8000,
          "core:description": "Synthetic test fixture, not satellite RF",
        },
        captures: [
          {
            "core:sample_start": 0,
            "core:datetime": "2026-01-01T00:00:00Z",
            "core:frequency": 137000000,
          },
        ],
      }),
    ),
  });
  await page.getByLabel("Recording session / group ID").fill("phase2-sigmf-fixture");
  await page.getByRole("button", { name: "Import external recording", exact: true }).click();
  await expect(page.getByText("1024", { exact: true })).toBeVisible({ timeout: 15000 });
  await page.getByLabel("External window start").fill("128");
  const inspected = page.waitForResponse((r) => r.url().includes("/window?start=128"));
  await page.getByRole("button", { name: "Inspect selected window" }).click();
  const capture = await (await inspected).json();
  expect(capture.position.timestamp).toBe("2026-01-01T00:00:00.016000+00:00");
  expect(capture.position.centerFrequency).toBe(137000000);
});

test("registered independent labels produce an actual external evaluation report", async ({
  page,
  request,
}) => {
  const generated = await (
    await request.post("/api/signal/generate", {
      data: { modulation: "BPSK", numSamples: 1024, seed: 782 },
    })
  ).json();
  const values = new Float32Array(
    generated.samples.flatMap((p: { i: number; q: number }) => [p.i, p.q]),
  );
  const imported = await request.post("/api/workspace/captures", {
    multipart: {
      file: {
        name: "labeled-synthetic-fixture.bin",
        mimeType: "application/octet-stream",
        buffer: Buffer.from(values.buffer),
      },
      options: JSON.stringify({
        format: "raw",
        datatype: "cf32_le",
        sampleRate: 8000,
        sessionId: "phase2-labeled-synthetic-session",
        source: "Synthetic known BPSK fixture, not real RF",
      }),
    },
  });
  expect(imported.ok()).toBeTruthy();
  const capture = await imported.json();
  await page.goto("/evaluation");
  await page.locator("summary").filter({ hasText: "External labeled evaluation" }).click();
  await page.getByLabel("Labeled dataset manifest JSON").fill(
    JSON.stringify({
      name: "Phase2 known-label test",
      source: "Generated test data, not real RF",
      license: "Test fixture",
      entries: Array.from({ length: 201 }, (_, start) => ({ captureId: capture.id, start, label: "BPSK", split: "test" })),
    }),
  );
  await page.getByRole("button", { name: "Register labeled dataset", exact: true }).click();
  const card = page
    .locator("div.border")
    .filter({ has: page.getByText(/Phase2 known-label test · 201 labeled windows/) })
    .last();
  await expect(card).toBeVisible({ timeout: 15000 });
  await card.getByRole("button", { name: "Evaluate held-out windows" }).click();
  await expect(page.getByText(/External evaluation: completed/)).toBeVisible({ timeout: 20000 });
  await expect(page.getByRole("button", { name: "Export external evaluation JSON" })).toBeVisible();
  const evaluationDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export external evaluation JSON" }).click();
  const evaluation = JSON.parse(await readFile(await (await evaluationDownload).path(), "utf8"));
  expect(evaluation.predictions).toHaveLength(201);
  expect(evaluation.metrics.samples).toBe(201);
  expect(evaluation.predictions.every((p: { groundTruth: string }) => p.groundTruth === "BPSK")).toBeTruthy();
  expect(evaluation.dataset.hash).toMatch(/^[a-f0-9]{64}$/);
  const evaluationPdfDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export external evaluation PDF" }).click();
  expect((await readFile(await (await evaluationPdfDownload).path())).subarray(0, 4).toString()).toBe("%PDF");
});

test("a running batch can be cancelled through its persistent job controls", async ({ page, request }) => {
  const values = Float32Array.from({ length: 16000 * 2 }, (_, i) => .006 * Math.sin(i * .031));
  const imported = await request.post("/api/workspace/captures", {
    multipart: {
      file: { name: "cancel-fixture.bin", mimeType: "application/octet-stream", buffer: Buffer.from(values.buffer) },
      options: JSON.stringify({ format: "raw", datatype: "cf32_le", sampleRate: 8000, sessionId: "cancel-synthetic-session", source: "Generated cancellation fixture, not RF" }),
    },
  });
  expect(imported.ok()).toBeTruthy();
  const capture = await imported.json();
  await page.goto("/classifier");
  await page.locator("summary").filter({ hasText: "External recordings" }).click();
  await page.getByRole("combobox", { name: "Stored capture", exact: true }).selectOption(capture.id);
  await page.getByLabel("Batch window count").fill("10000");
  await page.getByLabel("Batch stride (samples)").fill("1");
  await page.getByRole("button", { name: "Start batch classification" }).click();
  const jobControl = page.getByRole("button", { name: /batch · (queued|running)/ });
  await expect(jobControl).toBeVisible();
  await jobControl.locator("..").getByRole("button", { name: "Cancel job" }).click();
  await expect(page.getByText(/Job cancelled/)).toBeVisible({ timeout: 20000 });
});
