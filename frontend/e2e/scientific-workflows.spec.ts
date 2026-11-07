import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";

test("impaired signal → real CNN → Integrated Gradients → exported scientific report", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/studio");
  await page.getByLabel("Sample rate (Hz; 1 = normalized)", { exact: true }).fill("8000");
  await page.getByLabel("Doppler offset (Hz)", { exact: true }).fill("25");
  await page.getByLabel("Phase jitter σ (rad)", { exact: true }).fill(".02");
  await page.getByRole("button", { name: "Generate signal", exact: true }).click();
  await expect(page.getByText("Generated capture", { exact: true })).toBeVisible();
  await page.getByRole("link", { name: "Send to classifier" }).click();
  const inference = page.waitForResponse(
    (r) => r.url().endsWith("/api/classify") && r.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Run CNN classification", exact: true }).click();
  const actual = await (await inference).json();
  expect(actual.trust.calibration.status).toBe("not-calibrated");
  expect(actual.trust.ood.detected).toBeNull();
  await expect(page.getByText("Trust and model explanation", { exact: true })).toBeVisible();
  await expect(page.getByText("not-calibrated", { exact: true })).toBeVisible();
  await page
    .getByRole("combobox", { name: "Explain target class", exact: true })
    .selectOption("QPSK");
  const attribution = page.waitForResponse((r) => r.url().endsWith("/api/explain"));
  await page.getByRole("button", { name: "Explain classification", exact: true }).click();
  const explained = await (await attribution).json();
  expect(explained.method).toBe("integrated-gradients");
  expect(explained.targetClass).toBe("QPSK");
  expect(explained.samples).toHaveLength(128);
  expect(explained.integrationSteps).toBeGreaterThanOrEqual(64);
  expect(explained.samples.some((v: { importance: number }) => v.importance > 0)).toBe(true);
  await expect(page.getByText("Completeness residual", { exact: true })).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export results", exact: true }).click();
  const report = JSON.parse(await readFile((await (await download).path())!, "utf8"));
  expect(report.explanation.targetClass).toBe("QPSK");
  expect(report.samples).toHaveLength(1024);
  expect(report.trust.ood.status).toBe("not-validated");
  expect(errors).toEqual([]);
  await page.screenshot({
    path: "../integration/audit/classifier-explanation.png",
    fullPage: true,
  });
});

test("upload preserves raw samples and applies an explicitly changed sample clock", async ({
  page,
  request,
}) => {
  const raw = await (await request.get("/api/samples/97")).json();
  await page.goto("/classifier");
  await page.getByLabel("Upload I/Q file").setInputFiles({
    name: "audit.csv",
    mimeType: "text/csv",
    buffer: Buffer.from(
      "I,Q\n" + raw.samples.map((v: { i: number; q: number }) => `${v.i},${v.q}`).join("\n"),
    ),
  });
  await expect(page.getByText("Shannon entropy", { exact: true })).toBeVisible();
  await expect(page.getByText("valid", { exact: true })).toBeVisible();
  await page.getByLabel("Capture sample rate (Hz; 1 = normalized)", { exact: true }).fill("8000");
  const response = page.waitForResponse((r) => r.url().endsWith("/api/classify"));
  await page.getByRole("button", { name: "Run CNN classification", exact: true }).click();
  const result = await (await response).json();
  expect(result.sampleRate).toBe(8000);
  expect(result.analysis.sampleRate).toBe(8000);
  expect(result.samples).toEqual(raw.samples);
  expect(result.preprocessing.amplitudePolicy).toContain("training z-score only");
});

test("GMSK and BFSK controls drive real frequency modulation without fake symbol eyes", async ({
  page,
}) => {
  await page.goto("/studio");
  await page.getByRole("combobox", { name: "Modulation scheme", exact: true }).selectOption("GMSK");
  await page.getByLabel("Gaussian BT product", { exact: true }).fill(".15");
  await page.getByLabel("Sample rate (Hz; 1 = normalized)", { exact: true }).fill("8000");
  let response = page.waitForResponse((r) => r.url().endsWith("/api/signal/generate"));
  await page.getByRole("button", { name: "Generate signal", exact: true }).click();
  const gmsk = await (await response).json();
  expect(gmsk.parameters.gaussianBt).toBe(0.15);
  expect(gmsk.generation.modulationIndex).toBe(0.5);
  expect(gmsk.analysis.eye).toBeNull();
  expect(gmsk.reference).toEqual([]);
  await expect(page.getByRole("tab", { name: "eye", exact: true })).toHaveCount(0);
  await page.getByRole("tab", { name: "instantaneous frequency", exact: true }).click();
  await expect(
    page.getByText("Adjacent-sample wrapped phase difference", { exact: false }),
  ).toBeVisible();
  await page.getByRole("combobox", { name: "Modulation scheme", exact: true }).selectOption("BFSK");
  await page.getByLabel("BFSK deviation (Hz; default Rs/4)", { exact: true }).fill("200");
  await page.getByLabel("Continuous-phase BFSK", { exact: true }).uncheck();
  response = page.waitForResponse((r) => r.url().endsWith("/api/signal/generate"));
  await page.getByRole("button", { name: "Generate signal", exact: true }).click();
  const bfsk = await (await response).json();
  expect(bfsk.generation.frequencyDeviationHz).toBe(200);
  expect(bfsk.generation.phaseContinuous).toBe(false);
  expect(bfsk.classifierSupported).toBe(false);
});

test("two captures with mixed known/unknown rates use common normalized axes and real predictions", async ({
  page,
  request,
}) => {
  const raw = await (await request.get("/api/samples/97")).json();
  await page.goto("/studio");
  await page.getByLabel("Sample rate (Hz; 1 = normalized)", { exact: true }).fill("8000");
  await page.getByRole("button", { name: "Generate signal", exact: true }).click();
  await page.getByRole("link", { name: "Analyze capture" }).click();
  await page
    .getByLabel("Upload signal B for analysis (no classification required)", { exact: true })
    .setInputFiles({
      name: "reference.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify({ samples: raw.samples })),
    });
  await expect(page.getByText("B: reference.json", { exact: true })).toBeVisible();
  const response = page.waitForResponse((r) => r.url().endsWith("/api/signal/compare"));
  await page.getByRole("button", { name: "Compare signals", exact: true }).click();
  const compared = await (await response).json();
  expect(compared.sampleMse).toBeNull();
  expect(compared.plots.frequencyUnit).toBe("cycles/sample");
  expect(compared.predictions.first.status).toBe("available");
  expect(compared.predictions.second.status).toBe("available");
  await expect(page.getByText("Capture A · real CNN", { exact: true })).toBeVisible();
  await expect(
    page.getByText("At least one sample rate is unknown", { exact: false }),
  ).toBeVisible();
  await page.screenshot({ path: "../integration/audit/signal-comparison.png", fullPage: true });
});

test("real demo diagnostics and matched experimental comparisons remain distinct from historical scores", async ({
  page,
}) => {
  await page.goto("/evaluation");
  const response = page.waitForResponse((r) => r.url().endsWith("/api/evaluation/diagnostics"));
  await page.getByRole("button", { name: "Load real demo diagnostics", exact: true }).click();
  const diagnostic = await (await response).json();
  expect(diagnostic.samples).toBe(300);
  expect(diagnostic.independentTest).toBe(false);
  expect(diagnostic.comparisons["training-zscore-only"].accuracy).toBe(0.71);
  await expect(page.getByText("Expected calibration error", { exact: true })).toBeVisible();
  await page.goto("/arena");
  await expect(page.getByText("rf-control-v1", { exact: true })).toBeVisible();
  await expect(page.getByText("rf-awgn-v1", { exact: true })).toBeVisible();
  await page.getByRole("combobox", { name: "Model", exact: true }).selectOption("rf-experimental");
  const benchmark = page.waitForResponse((r) => r.url().endsWith("/api/benchmark"));
  await page.getByRole("button", { name: "Run selected model benchmark", exact: true }).click();
  const result = await (await benchmark).json();
  expect(result.modelId).toBe("rf-experimental");
  expect(result.hardware.batchSize).toBe(1);
  expect(result.preprocessing.medianMs).toBeGreaterThan(0);
  await expect(page.getByText("Median preprocessing", { exact: true })).toBeVisible();
  await page.screenshot({ path: "../integration/audit/model-arena.png", fullPage: true });
});
