import { test, expect, type APIRequestContext, type Page } from "@playwright/test";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

const labels = ["BPSK", "QPSK", "16-QAM"] as const;

async function registerControlledFixture(
  request: APIRequestContext,
  name: string,
  groups: number,
  seedBase: number,
) {
  const entries = [];
  for (let group = 0; group < groups; group++) {
    const responses = await Promise.all(
      labels.map((modulation, index) =>
        request.post("/api/signal/generate", {
          data: {
            modulation,
            numSamples: 128,
            sampleRate: 8000,
            snrDb: 6,
            seed: seedBase + group * 17 + index,
          },
        }),
      ),
    );
    const generated = [];
    for (const response of responses) {
      expect(response.ok()).toBeTruthy();
      const signal = await response.json();
      expect(signal.samples).toHaveLength(128);
      generated.push(signal);
    }
    const values = new Float32Array(
      generated.flatMap((signal) =>
        signal.samples.flatMap((point: { i: number; q: number }) => [point.i, point.q]),
      ),
    );
    const imported = await request.post("/api/workspace/captures", {
      multipart: {
        file: {
          name: `${name}-${group}.bin`,
          mimeType: "application/octet-stream",
          buffer: Buffer.from(values.buffer),
        },
        options: JSON.stringify({
          format: "raw",
          datatype: "cf32_le",
          sampleRate: 8000,
          sessionId: `${name}-generated-session-${group}`,
          source: "Controlled seeded synthetic fixture, not real satellite RF",
        }),
      },
    });
    expect(imported.ok()).toBeTruthy();
    const capture = await imported.json();
    for (let index = 0; index < labels.length; index++) {
      entries.push({
        captureId: capture.id,
        start: index * 128,
        label: labels[index],
        split: groups === 9 ? ["train", "validation", "test"][Math.floor(group / 3)] : "test",
      });
    }
  }
  const registered = await request.post("/api/workspace/datasets", {
    data: {
      name,
      source:
        "Controlled seeded synthetic groups; no independent satellite-domain validation claim",
      license: "Synthetic browser test fixture",
      entries,
    },
  });
  expect(registered.ok()).toBeTruthy();
  return registered.json();
}

async function evaluateInBrowser(
  page: Page,
  request: APIRequestContext,
  datasetId: string,
  modelId: string,
) {
  await page
    .getByRole("combobox", { name: "External evaluation model", exact: true })
    .selectOption(modelId);
  const response = page.waitForResponse(
    (value) =>
      value.request().method() === "POST" &&
      new URL(value.url()).pathname === "/api/research/evaluations",
  );
  await page
    .getByTestId(`registered-dataset-${datasetId}`)
    .getByRole("button", { name: "Evaluate held-out windows", exact: true })
    .click();
  const accepted = await response;
  expect(accepted.status()).toBe(202);
  const job = await accepted.json();
  await expect
    .poll(async () => (await (await request.get(`/api/workspace/jobs/${job.id}`)).json()).state)
    .toBe("completed");
  await expect(page.getByText(new RegExp(`Actual evaluation: ${modelId}`))).toBeVisible();
  const result = await request.get(`/api/workspace/jobs/${job.id}/result`);
  expect(result.ok()).toBeTruthy();
  return { job, report: await result.json() };
}

test("available registered models execute real held-out inference and compare on an identical protocol", async ({
  page,
  request,
}) => {
  const dataset = await registerControlledFixture(
    request,
    "phase3-model-selection-synthetic",
    1,
    47210,
  );
  const catalog = await (await request.get("/api/research/models")).json();
  await page.goto("/evaluation");
  await page
    .locator("summary")
    .filter({ hasText: "External labeled evaluation and dataset provenance" })
    .click();
  for (const model of catalog.filter((item: { available: boolean }) => !item.available)) {
    const option = page
      .getByRole("combobox", { name: "External evaluation model", exact: true })
      .locator(`option[value="${model.id}"]`);
    // Playwright's disabled-state matcher follows the surrounding Field label
    // to its enabled select. Verify the native option itself instead.
    await expect(option).toHaveJSProperty("disabled", true);
  }
  const first = await evaluateInBrowser(page, request, dataset.id, "1d-cnn");
  expect(first.report.classOrder).toEqual(labels);
  expect(first.report.dataset.hash).toBe(dataset.hash);
  expect(first.report.metrics.samples).toBe(3);
  expect(first.report.metrics.snrAccuracy).toEqual([]);
  expect(
    first.report.predictions.every((item: { modelId: string }) => item.modelId === "1d-cnn"),
  ).toBeTruthy();
  const rf = catalog.find((item: { id: string }) => item.id === "rf-experimental");
  if (!rf?.available) {
    test.info().annotations.push({
      type: "runtime-limitation",
      description:
        "Recorded experimental RF runtime is unavailable; UI correctly disabled its option. Genuine RF inference is verified only on compatible runtimes.",
    });
    return;
  }
  const second = await evaluateInBrowser(page, request, dataset.id, "rf-experimental");
  expect(second.report.modelHash).toBe(rf.artifactSha256);
  expect(second.report.modelVersion).toBe(rf.version);
  expect(
    second.report.predictions.every(
      (item: { modelId: string }) => item.modelId === "rf-experimental",
    ),
  ).toBeTruthy();
  expect(second.report.evaluationProtocol.protocolId).toBe(
    first.report.evaluationProtocol.protocolId,
  );
  await page.goto("/arena");
  await page.locator("summary").filter({ hasText: "Matched external model evaluations" }).click();
  await page
    .getByRole("combobox", { name: "Comparison evaluation dataset", exact: true })
    .selectOption(dataset.id);
  await page
    .getByRole("checkbox", { name: `Compare evaluation ${first.job.id}`, exact: true })
    .check();
  await page
    .getByRole("checkbox", { name: `Compare evaluation ${second.job.id}`, exact: true })
    .check();
  const compared = page.waitForResponse(
    (value) =>
      value.request().method() === "POST" &&
      new URL(value.url()).pathname === "/api/research/evaluations/compare",
  );
  await page.getByRole("button", { name: "Compare selected evaluations", exact: true }).click();
  const result = await compared;
  expect(result.ok()).toBeTruthy();
  const comparison = await result.json();
  expect(comparison.evaluations).toHaveLength(2);
  expect(
    comparison.evaluations.every(
      (item: { datasetHash: string }) => item.datasetHash === dataset.hash,
    ),
  ).toBeTruthy();
  await expect(page.getByText(/Matched observed results only/)).toBeVisible();
});

test("registered training windows export a provenance manifest, compatible NPZ and offline config", async ({
  page,
  request,
}) => {
  const dataset = await registerControlledFixture(
    request,
    "phase3-training-export-synthetic",
    9,
    58210,
  );
  await page.goto("/evaluation");
  await page
    .locator("summary")
    .filter({ hasText: "External labeled evaluation and dataset provenance" })
    .click();
  const card = page.getByTestId(`registered-dataset-${dataset.id}`);
  await card.locator("summary").filter({ hasText: "Offline training dataset export" }).click();
  await card.getByRole("checkbox", { name: /Allow fewer than 1,000 windows/ }).check();
  const validationResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "GET" &&
      new URL(response.url()).pathname === `/api/research/datasets/${dataset.id}/training-manifest`,
  );
  await card.getByRole("button", { name: "Validate offline training export", exact: true }).click();
  const validated = await validationResponse;
  expect(validated.ok(), await validated.text()).toBeTruthy();
  await expect(
    card.getByRole("button", { name: "Export validated training NPZ", exact: true }),
  ).toBeVisible();
  const manifestDownload = page.waitForEvent("download");
  await card.getByRole("button", { name: "Export training manifest JSON", exact: true }).click();
  const manifest = JSON.parse(await readFile(await (await manifestDownload).path(), "utf8"));
  expect(manifest.datasetHash).toBe(dataset.hash);
  expect(manifest.sampleCount).toBe(27);
  expect(Object.keys(manifest.splits).sort()).toEqual(["test", "train", "validation"]);
  expect(manifest.splits.train.samples).toBe(9);
  expect(manifest.splits.test.samples).toBe(9);
  expect(manifest.snr.knownSamples).toBe(0);
  expect(manifest.snr.unavailableSamples).toBe(27);
  expect(manifest.offlineCommand).toContain("--allow-small-dataset");
  const npzDownload = page.waitForEvent("download");
  await card.getByRole("button", { name: "Export validated training NPZ", exact: true }).click();
  const bytes = await readFile(await (await npzDownload).path());
  expect(bytes.subarray(0, 2).toString()).toBe("PK");
  expect(createHash("sha256").update(bytes).digest("hex")).toBe(manifest.artifact.sha256);
  const configDownload = page.waitForEvent("download");
  await card.getByRole("button", { name: "Export offline experiment config", exact: true }).click();
  expect(JSON.parse(await readFile(await (await configDownload).path(), "utf8"))).toEqual(
    manifest.experimentConfig,
  );
});

test("test-only labels cannot be silently reassigned into a training export", async ({
  page,
  request,
}) => {
  const dataset = await registerControlledFixture(
    request,
    "phase3-holdout-only-synthetic",
    1,
    69210,
  );
  await page.goto("/evaluation");
  await page
    .locator("summary")
    .filter({ hasText: "External labeled evaluation and dataset provenance" })
    .click();
  const card = page.getByTestId(`registered-dataset-${dataset.id}`);
  await card.locator("summary").filter({ hasText: "Offline training dataset export" }).click();
  await card.getByRole("checkbox", { name: /Allow fewer than 1,000 windows/ }).check();
  await card.getByRole("button", { name: "Validate offline training export", exact: true }).click();
  await expect(card.getByRole("alert")).toContainText("nine independent");
  await expect(
    card.getByRole("button", { name: "Export validated training NPZ", exact: true }),
  ).toHaveCount(0);
  expect(
    (await (await request.get("/api/workspace/datasets")).json()).items
      .find((item: { id: string }) => item.id === dataset.id)
      .entries.every((entry: { split: string }) => entry.split === "test"),
  ).toBeTruthy();
});
