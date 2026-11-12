import { test, expect, type APIRequestContext, type Page } from "@playwright/test";
import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";

interface PlotElement extends HTMLElement {
  _fullLayout: { xaxis: { range: number[] }; shapes?: { y0: number; y1: number }[] };
  _context: { displayModeBar: boolean; scrollZoom: boolean };
  data: { type: string; x?: number[]; y?: number[]; z?: number[] | number[][] }[];
}

function generatedTone(count = 1024, fs = 8000, frequency = 1000) {
  return Array.from({ length: count }, (_, index) => ({
    i: Math.fround(.006 * Math.cos(2 * Math.PI * frequency * index / fs)),
    q: Math.fround(.006 * Math.sin(2 * Math.PI * frequency * index / fs)),
  }));
}

async function importTone(request: APIRequestContext, name: string, fs = 8000, count = 1024) {
  const samples = generatedTone(count, fs, fs / 8);
  const bytes = Buffer.alloc(count * 8);
  for (let index = 0; index < samples.length; index++) {
    bytes.writeFloatLE(samples[index]!.i, index * 8);
    bytes.writeFloatLE(samples[index]!.q, index * 8 + 4);
  }
  const response = await request.post("/api/workspace/captures", { multipart: {
    file: { name: name + ".bin", mimeType: "application/octet-stream", buffer: bytes },
    options: JSON.stringify({ format: "raw", datatype: "cf32_le", sampleRate: fs,
      sessionId: name, source: "Generated tone fixture, not recorded satellite RF: " + name }),
  } });
  expect(response.ok(), await response.text()).toBeTruthy();
  return response.json() as Promise<{ id: string }>;
}

async function openExternal(page: Page) {
  await page.locator("summary").filter({ hasText: "External recordings" }).click();
}

async function uploadLegacyTone(page: Page, filename: string) {
  const response = page.waitForResponse((item) => item.request().method() === "POST" && item.url().includes("/api/classify/upload"));
  await page.getByLabel("Upload I/Q file", { exact: true }).setInputFiles({
    name: filename, mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ samples: generatedTone() })),
  });
  const classified = await response;
  expect(classified.ok(), await classified.text()).toBeTruthy();
  await expect(page.getByLabel("Window start (sample offset)")).toHaveAttribute("max", "896");
}

function inspectSigmfBundle(path: string) {
  const python = process.env["SATMOD_PYTHON"] ?? (process.platform === "win32" ? "..\\.venv\\Scripts\\python.exe" : "../.venv/bin/python");
  const result = spawnSync(python, ["-c", [
    "import hashlib,json,struct,sys,zipfile",
    "with zipfile.ZipFile(sys.argv[1]) as z:",
    " data=z.read('capture.sigmf-data')",
    " print(json.dumps({'metadata':json.loads(z.read('capture.sigmf-meta')),'source':json.loads(z.read('source.sigmf-meta')),'samples':list(struct.iter_unpack('<ff',data)),'sha512':hashlib.sha512(data).hexdigest()}))",
  ].join("\n"), path], { encoding: "utf8" });
  expect(result.status, result.error?.message ?? result.stderr).toBe(0);
  return JSON.parse(result.stdout);
}

test("unsigned real multichannel SigMF requires explicit conversion and exports consistent derived metadata", async ({ page }) => {
  const count = 512;
  const raw = Buffer.alloc(count * 2 * 2);
  for (let index = 0; index < count; index++) {
    raw.writeUInt16BE(32768 + 100, index * 4);
    raw.writeUInt16BE(32768 + (index % 17) - 8, index * 4 + 2);
  }
  const metadata = {
    global: { "core:version": "1.2.6", "core:datatype": "ru16_be", "core:sample_rate": 16000,
      "core:num_channels": 2, "core:offset": 1000, "core:description": "Generated unsigned-real fixture, not satellite RF",
      "core:extensions": [{ name: "antenna", version: "1.0.0", optional: true }], "antenna:model": "Synthetic metadata fixture" },
    captures: [
      { "core:sample_start": 1000, "core:global_index": 7000, "core:frequency": 137000000, "core:datetime": "2026-11-12T00:00:00Z" },
      { "core:sample_start": 1256, "core:global_index": 10000, "core:frequency": 138000000, "core:datetime": "2026-11-12T00:00:01Z" },
    ],
    annotations: [{ "core:sample_start": 1200, "core:sample_count": 180, "core:label": "Synthetic annotated region" }],
  };
  await page.goto("/classifier");
  await openExternal(page);
  await page.getByRole("combobox", { name: "External import format" }).selectOption("sigmf");
  await page.getByLabel("External recording file", { exact: true }).setInputFiles({ name: "phase3-unsigned-real.sigmf-data", mimeType: "application/octet-stream", buffer: raw });
  await page.getByLabel("SigMF metadata file", { exact: true }).setInputFiles({ name: "phase3-unsigned-real.sigmf-meta", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(metadata)) });
  await page.getByLabel("Recording session / group ID").fill("phase3-generated-unsigned-real-session");
  await page.getByLabel("Import sample offset").fill("64");
  await page.getByLabel("Import window length (blank = remainder)").fill("384");
  await page.getByLabel("Explicit sample scaling").fill("0.0001");
  await page.getByRole("button", { name: "Import external recording", exact: true }).click();
  await expect(page.getByText(/Multichannel SigMF requires an explicit channelIndex/).first()).toBeVisible({ timeout: 15000 });
  await page.getByLabel("SigMF channel index (required for multiple channels)").fill("1");
  await page.getByLabel("Unsigned ADC zero level (required for unsigned data)").fill("32768");
  await page.getByRole("checkbox", { name: /Real-valued SigMF treatment/ }).check();
  await page.getByRole("button", { name: "Import external recording", exact: true }).click();
  await expect(page.getByText(/^Job completed/)).toBeVisible({ timeout: 15000 });
  await expect(page.getByLabel("External window start")).toHaveAttribute("max", "256");
  const captureId = await page.getByRole("combobox", { name: "Stored capture", exact: true }).inputValue();
  await page.getByLabel("External window start").fill("192");
  const selected = page.waitForResponse((response) => response.url().includes(`/captures/${captureId}/window?start=192`));
  await page.getByRole("button", { name: "Inspect selected window", exact: true }).click();
  const inspected = await (await selected).json();
  expect(inspected.metadata).toMatchObject({ sourceChannels: 2, channelIndex: 1, sampleZero: 32768, realMode: "i-only", sourceOffset: 64, sigmfSampleOffset: 1000 });
  expect(inspected.position).toMatchObject({ fileSample: 256, sourceSample: 1256, globalSampleIndex: 10000, centerFrequency: 138000000, timestamp: "2026-11-12T00:00:01+00:00" });
  for (let index = 0; index < 128; index++) {
    expect(inspected.samples[index].i).toBeCloseTo((((256 + index) % 17) - 8) * .0001, 9);
    expect(inspected.samples[index].q).toBe(0);
  }
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export metadata-preserving SigMF window", exact: true }).click();
  const path = await (await download).path();
  expect(path).toBeTruthy();
  const exported = inspectSigmfBundle(path);
  expect(exported.source).toEqual(metadata);
  expect(exported.metadata.global).toMatchObject({ "core:datatype": "cf32_le", "core:num_channels": 1, "core:offset": 1256, "core:sha512": exported.sha512 });
  expect(exported.metadata.global["satmod:provenance"]).toMatchObject({ sourceDatatype: "ru16_be", selectedChannel: 1, sourceChannels: 2, sampleZero: 32768, scale: .0001, realMode: "i-only" });
  expect(exported.metadata.captures[0]).toMatchObject({ "core:sample_start": 1256, "core:global_index": 10000, "core:datetime": "2026-11-12T00:00:01Z" });
  expect(exported.metadata.annotations[0]).toMatchObject({ "core:sample_start": 1256, "core:sample_count": 124 });
  expect(exported.samples).toEqual(inspected.samples.map((sample: { i: number; q: number }) => [sample.i, sample.q]));
});

test("time-frequency controls, marker, waterfall and export use actual computed density", async ({ page }) => {
  await page.goto("/classifier");
  await uploadLegacyTone(page, "phase3-generated-time-frequency.json");
  await page.getByLabel("Capture sample rate (Hz; 1 = normalized)").fill("8000");
  const classified = page.waitForResponse((response) => response.request().method() === "POST" && new URL(response.url()).pathname === "/api/classify");
  await page.getByRole("button", { name: "Run CNN classification", exact: true }).click();
  expect((await classified).ok()).toBeTruthy();
  await expect(page.getByText(/phase3-generated-time-frequency\.json · Fs 8000 Hz/)).toBeVisible();
  const initial = page.waitForResponse((response) => response.url().includes("/api/research/time-frequency"));
  await page.getByRole("tab", { name: "spectrogram", exact: true }).click();
  const report = await (await initial).json();
  expect(report.frequencyUnit).toBe("Hz");
  expect(report.timeUnit).toBe("s");
  expect(report.peaks[0].frequency).toBe(1000);
  await page.getByRole("combobox", { name: "STFT FFT size", exact: true }).selectOption("128");
  await page.getByLabel("STFT overlap fraction", { exact: true }).fill("0.75");
  await page.getByRole("combobox", { name: "STFT window", exact: true }).selectOption("boxcar");
  const recomputed = page.waitForResponse((response) => response.url().includes("/api/research/time-frequency"));
  await page.getByRole("button", { name: "Recompute time-frequency analysis", exact: true }).click();
  const response = await recomputed;
  const actual = await response.json();
  expect(response.request().postDataJSON()).toMatchObject({ fftSize: 128, overlap: .75, window: "boxcar", sampleRate: 8000 });
  expect(actual.parameters).toMatchObject({ fftSize: 128, overlapSamples: 96, hopSamples: 32, sampledTimeWindows: 29, detrend: false, padding: false, displayFloorDb: -300 });
  expect(actual.peaks[0].frequency).toBe(1000);
  await page.getByLabel("Baseband frequency marker", { exact: true }).fill("1000");
  const heatmap = page.getByRole("group", { name: "Computed time-frequency PSD", exact: true });
  await expect(heatmap).toHaveAttribute("aria-busy", "false");
  await expect.poll(() => heatmap.evaluate((element) => (element as PlotElement)._fullLayout.shapes?.[0]?.y0)).toBe(1000);
  await page.getByRole("combobox", { name: "Time-frequency view", exact: true }).selectOption("waterfall");
  const waterfall = page.getByRole("group", { name: "Computed PSD waterfall", exact: true });
  await expect(waterfall).toHaveAttribute("aria-busy", "false");
  expect(await waterfall.evaluate((element) => (element as PlotElement).data.map((trace) => trace.type))).toEqual(["surface", "scatter3d"]);
  expect(await waterfall.evaluate((element) => (element as PlotElement).data[0]?.z)).toEqual(actual.db);
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export time-frequency JSON", exact: true }).click();
  const exported = JSON.parse(await readFile(await (await download).path(), "utf8"));
  expect(exported.analysis.inputHash).toBe(actual.inputHash);
  expect(exported.analysis.db).toEqual(actual.db);
  expect(exported.view).toMatchObject({ mode: "waterfall", operatorFrequencyMarker: 1000 });
  expect(exported.analysis.note).toContain("not a measured noise floor");
});

test("imported window synchronizes the mounted Classifier sample clock and CNN window offset", async ({ page, request }) => {
  const previous = await importTone(request, "phase3-delayed-clock-generated", 8000);
  const imported = await importTone(request, "phase3-clock-sync-generated", 16000);
  await page.goto("/classifier");
  await uploadLegacyTone(page, "phase3-old-clock-generated.json");
  await page.getByLabel("Capture sample rate (Hz; 1 = normalized)").fill("4000");
  await page.getByLabel("Window start (sample offset)").fill("256");
  await openExternal(page);
  const stalePath = `/api/workspace/captures/${previous.id}/window?start=0`;
  let releasePrevious: () => void = () => {};
  let actualFetched: () => void = () => {};
  const paused = new Promise<void>((resolve) => { releasePrevious = resolve; });
  const fetched = new Promise<void>((resolve) => { actualFetched = resolve; });
  // Only response timing changes: route.fetch obtains genuine API samples and
  // metadata, which are delivered unchanged after the second selection wins.
  await page.route(`**${stalePath}`, async (route) => {
    const response = await route.fetch();
    actualFetched();
    await paused;
    await route.fulfill({ response });
  });
  const previousResponse = page.waitForResponse((response) => response.url().endsWith(stalePath));
  await page.getByRole("combobox", { name: "Stored capture", exact: true }).selectOption(previous.id);
  await page.getByRole("button", { name: "Inspect selected window", exact: true }).click();
  await fetched;
  try {
    await expect(page.getByRole("button", { name: "Inspecting selected window…", exact: true })).toBeDisabled();
    await page.getByRole("combobox", { name: "Stored capture", exact: true }).selectOption(imported.id);
    await page.getByLabel("External window start").fill("128");
    const inspected = page.waitForResponse((response) => response.url().includes(`/captures/${imported.id}/window?start=128`));
    await page.getByRole("button", { name: "Inspect selected window", exact: true }).click();
    expect((await inspected).ok()).toBeTruthy();
    await expect(page.getByLabel("Capture sample rate (Hz; 1 = normalized)")).toHaveValue("16000");
  } finally {
    releasePrevious();
  }
  expect((await previousResponse).ok()).toBeTruthy();
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  await expect(page.getByLabel("Capture sample rate (Hz; 1 = normalized)")).toHaveValue("16000");
  await expect(page.getByLabel("Window start (sample offset)")).toHaveValue("0");
  const classified = page.waitForResponse((response) => response.request().method() === "POST" && new URL(response.url()).pathname === "/api/classify");
  await page.getByRole("button", { name: "Run CNN classification", exact: true }).click();
  const response = await classified;
  expect(response.request().postDataJSON()).toMatchObject({ sampleRate: 16000, windowStart: 0 });
  const current = await response.json();
  expect(current.sampleRate).toBe(16000);
  expect(current.analysis.sampleRate).toBe(16000);
  const direct = await request.post(`/api/workspace/captures/${imported.id}/classify?start=128`);
  expect(direct.ok()).toBeTruthy();
  expect(current.probabilities).toEqual((await direct.json()).probabilities);
});

test("mobile keyboard navigation traps focus, closes with Escape and restores the opener", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/classifier");
  await page.keyboard.press("Tab");
  const skip = page.getByRole("link", { name: "Skip to workspace content", exact: true });
  await expect(skip).toBeFocused();
  await skip.press("Enter");
  await expect(page.getByRole("main")).toBeFocused();
  const opener = page.getByRole("button", { name: "Open navigation", exact: true });
  await opener.focus();
  await page.keyboard.press("Enter");
  const dialog = page.getByRole("dialog", { name: "Workspace navigation", exact: true });
  await expect(dialog).toBeVisible();
  const focusables = dialog.locator('a[href],button:not([disabled])');
  await expect(focusables.first()).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(focusables.last()).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(focusables.first()).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(opener).toBeFocused();
  await opener.press("Enter");
  await dialog.getByRole("link", { name: "Signal intelligence", exact: true }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { name: "Signal Intelligence", exact: true })).toBeVisible();
  await expect(page.getByRole("dialog", { name: "Workspace navigation", exact: true })).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  for (const path of ["/intelligence", "/classifier"]) {
    if (new URL(page.url()).pathname !== path) await page.goto(path);
    await openExternal(page);
    await expect(page.getByLabel("External recording file", { exact: true })).toBeVisible();
    await page.locator("summary").filter({ hasText: "Persistent analysis history and technical reports" }).click();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
    const clear = await page.getByRole("button", { name: "Clear saved history", exact: true }).boundingBox();
    expect(clear).not.toBeNull();
    expect(clear!.x).toBeGreaterThanOrEqual(0);
    expect(clear!.x + clear!.width).toBeLessThanOrEqual(390);
  }
});

test("actual job polling preserves a genuine Plotly zoom and its rendered SVG", async ({ page, request }) => {
  const imported = await importTone(request, "phase3-generated-zoom-polling", 8000, 65536);
  await page.goto("/classifier");
  await openExternal(page);
  await page.getByRole("combobox", { name: "Stored capture", exact: true }).selectOption(imported.id);
  const chart = page.locator("details[open] .js-plotly-plot").first();
  await expect(chart).toHaveAttribute("aria-busy", "false");
  await chart.scrollIntoViewIfNeeded();
  const original = await chart.evaluate((element) => [...(element as PlotElement)._fullLayout.xaxis.range]);
  const surface = chart.locator(".nsewdrag").first();
  const bounds = await surface.boundingBox();
  expect(bounds).not.toBeNull();
  await page.mouse.move(bounds!.x + bounds!.width * .2, bounds!.y + bounds!.height * .2);
  await page.mouse.down();
  await page.mouse.move(bounds!.x + bounds!.width * .7, bounds!.y + bounds!.height * .7, { steps: 3 });
  await page.mouse.up();
  const zoomed = await chart.evaluate((element) => [...(element as PlotElement)._fullLayout.xaxis.range]);
  expect(Math.abs(zoomed[1]! - zoomed[0]!)).toBeLessThan(Math.abs(original[1]! - original[0]!) * .8);
  const svg = await chart.locator(".main-svg").first().elementHandle();
  expect(svg).not.toBeNull();
  await page.getByLabel("Batch window count").fill("10000");
  await page.getByLabel("Batch stride (samples)").fill("1");
  let activeId = "";
  let polls = 0;
  page.on("response", (response) => { if (activeId && new URL(response.url()).pathname === `/api/workspace/jobs/${activeId}`) polls++; });
  const submitted = page.waitForResponse((response) => response.request().method() === "POST" && response.url().includes("/jobs/batch"));
  await page.getByRole("button", { name: "Start batch classification", exact: true }).click();
  const submission = await submitted;
  expect(submission.ok()).toBeTruthy();
  activeId = (await submission.json()).id;
  try {
    await expect.poll(async () => (await (await request.get(`/api/workspace/jobs/${activeId}`)).json()).state).toBe("running");
    await expect.poll(() => polls, { timeout: 15000 }).toBeGreaterThanOrEqual(3);
    const after = await chart.evaluate((element) => [...(element as PlotElement)._fullLayout.xaxis.range]);
    expect(after[0]).toBeCloseTo(zoomed[0]!, 6);
    expect(after[1]).toBeCloseTo(zoomed[1]!, 6);
    expect(await svg!.evaluate((node) => node.isConnected)).toBeTruthy();
  } finally {
    await request.post(`/api/workspace/jobs/${activeId}/cancel`);
    await expect.poll(async () => (await (await request.get(`/api/workspace/jobs/${activeId}`)).json()).state, { timeout: 15000 }).toMatch(/^(cancelled|completed)$/);
  }
});

test("Theory controls show backend constellation and orbit values with the stated noise and amplifier formulas", async ({ page }) => {
  const references = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/constellations");
  await page.goto("/theory");
  const referenceResponse = await references;
  expect(referenceResponse.ok()).toBeTruthy();
  const reference = await referenceResponse.json();
  await page.getByRole("combobox", { name: "Ideal constellation", exact: true }).selectOption("16-APSK");
  const constellation = page.getByRole("group", { name: "Quadrature Q versus In-phase I", exact: true });
  await expect(constellation).toHaveAttribute("aria-busy", "false");
  await expect.poll(() => constellation.evaluate((element) => {
    const trace = (element as PlotElement).data[0];
    return { x: trace?.x, y: trace?.y };
  })).toEqual({ x: reference.modulations["16-APSK"].map((point: { i: number }) => point.i),
    y: reference.modulations["16-APSK"].map((point: { q: number }) => point.q) });
  await page.getByLabel("Sample SNR (dB)", { exact: true }).fill("20");
  await expect(page.getByText("Complex noise RMS (sample power = 1)", { exact: true }).locator("..")).toContainText("0.1");
  await expect(page.getByText("Per-quadrature noise σ", { exact: true }).locator("..")).toContainText("0.0707");
  await page.getByLabel("Altitude (km)", { exact: true }).fill("850");
  const changedOrbit = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return url.pathname === "/api/orbit" && url.searchParams.get("altitudeKm") === "850" && url.searchParams.get("carrierGhz") === "2.2";
  });
  await page.getByLabel("Carrier (GHz)", { exact: true }).fill("2.2");
  const orbitResponse = await changedOrbit;
  expect(orbitResponse.ok()).toBeTruthy();
  const orbit = await orbitResponse.json();
  expect(orbit.period_min).toBeCloseTo(2 * Math.PI * Math.sqrt((6378.137 + 850) ** 3 / 398600.4418) / 60, 9);
  for (const index of [0, 100, 200]) {
    expect(orbit.doppler_khz[index]).toBeCloseTo(-orbit.range_rate_kms[index] / 299792.458 * 2.2e6, 9);
  }
  const doppler = page.getByRole("group", { name: "Doppler shift (kHz) versus Time from closest approach (s)", exact: true });
  await expect.poll(() => doppler.evaluate((element) => {
    const trace = (element as PlotElement).data[0];
    return { x: trace?.x, y: trace?.y };
  })).toEqual({ x: orbit.time_sec, y: orbit.doppler_khz });
  const period = await page.getByText("Orbital period", { exact: true }).locator("..").locator(".font-mono").textContent();
  expect(Math.abs(parseFloat(period!) - orbit.period_min) / orbit.period_min).toBeLessThan(.005);
  await page.getByLabel("AM/AM gain αa", { exact: true }).fill("3.2");
  const amplifier = page.getByRole("group", { name: "Output amplitude versus Input amplitude", exact: true });
  await expect.poll(() => amplifier.evaluate((element) => (element as PlotElement).data[0]?.y?.[25])).toBeCloseTo(3.2 * .75 / (1 + 1.1517 * .75 ** 2), 12);
  const transfer = await amplifier.evaluate((element) => (element as PlotElement).data[0]);
  expect(transfer?.y).toEqual(transfer?.x?.map((radius) => 3.2 * radius / (1 + 1.1517 * radius * radius)));
});

test("Settings persist real workspace preferences, reject an invalid API address and reconnect through the same-origin proxy", async ({ page }) => {
  await page.goto("/settings");
  await expect(page.getByRole("button", { name: "Check connection", exact: true })).toBeEnabled();
  await page.getByRole("combobox", { name: "Theme", exact: true }).selectOption("light");
  await page.getByRole("combobox", { name: "Scientific chart controls", exact: true }).selectOption("simple");
  await page.getByRole("combobox", { name: "Default modulation", exact: true }).selectOption("16-QAM");
  await page.getByLabel("Default SNR (dB)", { exact: true }).fill("21");
  await page.getByLabel("Default samples per symbol", { exact: true }).fill("7");
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem("satmod-preferences") ?? "null"))).toMatchObject({
    theme: "light", charts: "simple", defaultModulation: "16-QAM", defaultSnr: 21, defaultSps: 7,
  });
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(page.getByRole("combobox", { name: "Scientific chart controls", exact: true })).toHaveValue("simple");
  await expect(page.getByRole("combobox", { name: "Default modulation", exact: true })).toHaveValue("16-QAM");
  await expect(page.getByLabel("Default SNR (dB)", { exact: true })).toHaveValue("21");
  await expect(page.getByLabel("Default samples per symbol", { exact: true })).toHaveValue("7");
  const address = page.getByLabel("API base URL (blank uses the same-origin /api proxy)", { exact: true });
  await address.fill("https://user:password@example.invalid");
  await page.getByRole("button", { name: "Save API address", exact: true }).click();
  await expect(page.getByRole("alert").filter({ hasText: "Use an HTTP(S) API URL without credentials, query parameters, or fragments." })).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem("satmod-api-url"))).toBeNull();
  await address.fill("");
  const reloaded = page.waitForEvent("domcontentloaded");
  await page.getByRole("button", { name: "Save API address", exact: true }).click();
  await reloaded;
  await expect(page.getByRole("heading", { name: "System Settings", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Check connection", exact: true })).toBeEnabled();
  expect(await page.evaluate(() => localStorage.getItem("satmod-api-url"))).toBe("");
  const checked = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/health" && response.request().method() === "GET");
  await page.getByRole("button", { name: "Check connection", exact: true }).click();
  const health = await checked;
  expect(health.ok()).toBeTruthy();
  expect(new URL(health.url()).origin).toBe(new URL(page.url()).origin);
  expect((await health.json()).connected).toBe(true);
  await expect(page.getByText("CONNECTED", { exact: true }).first()).toBeVisible();
  await page.getByRole("link", { name: "Satellite studio", exact: true }).click();
  await expect(page.getByRole("combobox", { name: "Modulation scheme", exact: true })).toHaveValue("16-QAM");
  await expect(page.getByLabel("AWGN SNR (dB)", { exact: true })).toHaveValue("21");
  await expect(page.getByLabel("Samples per symbol", { exact: true })).toHaveValue("7");
  await page.reload();
  await expect(page.getByRole("combobox", { name: "Modulation scheme", exact: true })).toHaveValue("16-QAM");
  await expect(page.getByLabel("AWGN SNR (dB)", { exact: true })).toHaveValue("21");
  await expect(page.getByLabel("Samples per symbol", { exact: true })).toHaveValue("7");
  await page.getByRole("link", { name: "Satellite theory", exact: true }).click();
  const chart = page.getByRole("group", { name: "Quadrature Q versus In-phase I", exact: true });
  await expect(chart).toHaveAttribute("aria-busy", "false");
  expect(await chart.evaluate((element) => {
    const context = (element as PlotElement)._context;
    return { displayModeBar: context.displayModeBar, scrollZoom: context.scrollZoom };
  })).toEqual({ displayModeBar: false, scrollZoom: false });
});

test("Studio downloads an actual generated WAV with the current workspace session token", async ({ page }) => {
  await page.goto("/studio");
  await page.evaluate(() => sessionStorage.setItem("satmod-access-token:", "synthetic-browser-token-fixture"));
  await page.getByRole("combobox", { name: "Modulation scheme", exact: true }).selectOption("QPSK");
  await page.getByLabel("Sample rate (Hz; 1 = normalized)", { exact: true }).fill("8000");
  const generated = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/signal/generate" && response.request().method() === "POST");
  await page.getByRole("button", { name: "Generate signal", exact: true }).click();
  const generation = await generated;
  expect(generation.ok()).toBeTruthy();
  const capture = await generation.json();
  const audioResponse = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/signal/audio" && response.request().method() === "POST");
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download audible heterodyne", exact: true }).click();
  const audio = await audioResponse;
  expect(audio.ok()).toBeTruthy();
  expect(await audio.request().headerValue("X-Satmod-Token")).toBe("synthetic-browser-token-fixture");
  expect(audio.request().postDataJSON()).toMatchObject({ samples: capture.samples, sampleRate: 8000, source: capture.source });
  const bytes = await readFile(await (await download).path());
  expect(bytes.subarray(0, 4).toString("ascii")).toBe("RIFF");
  expect(bytes.subarray(8, 12).toString("ascii")).toBe("WAVE");
  expect(bytes.readUInt32LE(4) + 8).toBe(bytes.length);
  expect(bytes.length).toBeGreaterThan(44);
  expect(audio.headers()["content-type"]).toContain("audio/wav");
  await expect(page.getByRole("button", { name: "Download audible heterodyne", exact: true })).toBeEnabled();
});
