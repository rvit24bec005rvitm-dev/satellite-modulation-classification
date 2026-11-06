import { expect, test } from "@playwright/test";

test("workflow 1: upload a real saved I/Q frame and display CNN output", async ({
  page,
  request,
}) => {
  const sample = await request.get("/api/samples/97");
  const data = (await sample.json()) as { samples: { i: number; q: number }[] };
  await page.goto("/classifier");
  await page.getByLabel("Upload I/Q file").setInputFiles({
    name: "capture.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("I,Q\n" + data.samples.map((p) => `${p.i},${p.q}`).join("\n")),
  });
  await expect(page.getByText("Shannon entropy", { exact: true })).toBeVisible();
  await expect(page.getByText("Inference latency", { exact: true })).toBeVisible();
  await expect(page.locator(".js-plotly-plot").first()).toBeVisible({ timeout: 15000 });
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export results", exact: true }).click();
  expect((await download).suggestedFilename()).toContain("classification");
});

for (const modulation of ["BPSK", "QPSK"])
  test(`workflow ${modulation === "BPSK" ? 2 : 3}: generate ${modulation} and classify the same full capture`, async ({
    page,
  }) => {
    await page.goto("/studio");
    await page.getByLabel("Modulation scheme").selectOption(modulation);
    await page
      .getByLabel("AWGN SNR (dB)", { exact: true })
      .fill(modulation === "QPSK" ? "6" : "18");
    await page.getByLabel("Capture length (samples)").fill("1024");
    await page.getByRole("button", { name: "Generate signal", exact: true }).click();
    await expect(page.getByText("Generated capture", { exact: true })).toBeVisible();
    await page.getByRole("link", { name: "Send to classifier" }).click();
    const captureRequest = page.waitForRequest(
      (r) => r.url().endsWith("/api/classify") && r.method() === "POST",
    );
    await page.getByRole("button", { name: "Run CNN classification" }).click();
    const payload = (await captureRequest).postDataJSON() as { samples: unknown[]; source: string };
    expect(payload.samples).toHaveLength(1024);
    expect(payload.source).toContain(modulation);
    await expect(page.getByText("Shannon entropy", { exact: true })).toBeVisible();
  });

test("workflow 4: actual evaluation matrices, SNR curves, and per-class metrics", async ({
  page,
}) => {
  await page.goto("/evaluation");
  await expect(
    page
      .getByText("Overall accuracy", { exact: true })
      .locator("..")
      .getByText("72.44%", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText("900 OBSERVATIONS")).toBeVisible();
  await expect(page.locator(".js-plotly-plot")).toHaveCount(3, { timeout: 15000 });
  await page.getByLabel("Row percentages").check();
  await expect(page.getByText("Misclassification analysis")).toBeVisible();
});

test("workflow 5: measured CNN latency and unavailable model comparison", async ({ page }) => {
  await page.goto("/arena");
  await page.getByRole("button", { name: "Run CNN benchmark" }).click();
  await expect(page.getByText("P95 latency", { exact: true })).toBeVisible();
  await expect(page.getByText("Throughput", { exact: true })).toBeVisible();
  await expect(page.getByText("not trained", { exact: true })).toHaveCount(3);
});

test("workflow 6: backend disconnection never produces predictions", async ({ page }) => {
  await page.route("**/api/**", (r) => r.abort("connectionrefused"));
  await page.goto("/classifier");
  await expect(page.getByRole("alert").first()).toContainText("Backend connection failed");
  await expect(page.getByText("API OFFLINE", { exact: true })).toBeVisible();
  await expect(page.getByText("Shannon entropy", { exact: true })).toHaveCount(0);
});

test("workflow 7: malformed upload is rejected visibly", async ({ page }) => {
  await page.goto("/classifier");
  await page.getByLabel("Upload I/Q file").setInputFiles({
    name: "invalid.npy",
    mimeType: "application/octet-stream",
    buffer: Buffer.from("invalid capture"),
  });
  await expect(page.getByRole("alert").first()).toContainText("Invalid capture");
  await expect(page.getByText("Shannon entropy", { exact: true })).toHaveCount(0);
});

test("all routes remain reachable on a narrow viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  for (const route of [
    "/",
    "/classifier",
    "/studio",
    "/evaluation",
    "/arena",
    "/intelligence",
    "/theory",
    "/settings",
  ]) {
    await page.goto(route);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    await expect(page.getByRole("button", { name: "Open navigation" })).toBeVisible();
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1),
    ).toBe(true);
  }
  await page.getByRole("button", { name: "Open navigation" }).click();
  await page.getByRole("link", { name: "Live classifier", exact: true }).click();
  await expect(page).toHaveURL(/classifier/);
});
