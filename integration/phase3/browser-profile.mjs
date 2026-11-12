// Reproducible actual-network/chart probe using explicitly synthetic samples.
import { chromium } from "../../frontend/node_modules/@playwright/test/index.mjs";
import { writeFile } from "node:fs/promises";
const baseURL = process.argv[2];
const output = process.argv[3];
if (!baseURL || !output) throw Error("Usage: node integration/phase3/browser-profile.mjs URL OUTPUT.json");
const browser = await chromium.launch({ channel: "chrome", args: ["--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1440, height: 1080 } });
const request = page.request;
const samples = Float32Array.from({ length: 65536 * 2 }, (_, i) => .006 * Math.sin(i * .037));
const get = async (path) => {
  const response = await request.get(baseURL + path);
  if (!response.ok()) throw Error(await response.text());
  return response.json();
};
const post = async (path, data) => {
  const response = await request.post(baseURL + path, { data });
  if (!response.ok()) throw Error(await response.text());
  return response.json();
};
const terminal = async (job) => {
  for (let i = 0; i < 1000; i++) {
    const state = await get(`/api/workspace/jobs/${job.id}`);
    if (["completed", "cancelled", "failed"].includes(state.state)) return state;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  throw Error("Profile job did not finish");
};
try {
  const t0 = performance.now();
  const imported = await request.post(baseURL + "/api/workspace/captures", { multipart: {
    file: { name: "phase3-profile-synthetic.bin", mimeType: "application/octet-stream", buffer: Buffer.from(samples.buffer) },
    options: JSON.stringify({ format: "raw", datatype: "cf32_le", sampleRate: 8000, sessionId: "phase3-profile-synthetic", source: "Synthetic performance probe, not RF" }),
  } });
  if (!imported.ok()) throw Error(await imported.text());
  const capture = await imported.json();
  const importRttMs = performance.now() - t0;
  const body = { samples: Array.from({ length: 128 }, (_, i) => ({ i: samples[2*i], q: samples[2*i+1] })), source: "Synthetic performance probe", sampleRate: 8000 };
  await post("/api/classify", body);
  const requestMs = [];
  for (let i = 0; i < 10; i++) {
    const start = performance.now();
    await post("/api/classify", body);
    requestMs.push(performance.now() - start);
  }
  const beginBatch = performance.now();
  const batch = await post("/api/workspace/jobs/batch", { captureId: capture.id, windows: 512 });
  const completed = await terminal(batch);
  if (completed.state !== "completed") throw Error(JSON.stringify(completed));
  const batchMs = performance.now() - beginBatch;
  const result = await get(`/api/workspace/jobs/${batch.id}/result`);
  await page.goto(baseURL + "/classifier");
  await page.locator("summary").filter({ hasText: "External recordings" }).click();
  await page.getByRole("combobox", { name: "Stored capture", exact: true }).selectOption(capture.id);
  const chart = page.locator("details[open] .js-plotly-plot").first();
  await chart.waitFor();
  await page.waitForFunction(() => !!document.querySelector("details[open] .js-plotly-plot")?._fullLayout?.xaxis);
  const drag = chart.locator(".nsewdrag").first();
  const bounds = await drag.boundingBox();
  if (!bounds) throw Error("No real Plotly drag surface");
  await page.mouse.move(bounds.x + bounds.width*.2, bounds.y + bounds.height*.2);
  await page.mouse.down();
  await page.mouse.move(bounds.x + bounds.width*.7, bounds.y + bounds.height*.7, { steps: 3 });
  await page.mouse.up();
  await page.waitForTimeout(150);
  const before = await chart.evaluate((el) => [...el._fullLayout.xaxis.range]);
  await page.evaluate(() => {
    window.__profile = { removals: 0, heartbeat: [], last: performance.now() };
    const chart = document.querySelector("details[open] .js-plotly-plot");
    window.__observer = new MutationObserver((events) => {
      for (const event of events) for (const node of event.removedNodes)
        if (node.nodeType === 1) window.__profile.removals += node.matches?.(".main-svg") ? 1 : node.querySelectorAll?.(".main-svg").length ?? 0;
    });
    window.__observer.observe(chart, { childList: true, subtree: true });
    window.__heartbeat = setInterval(() => { const now = performance.now(); window.__profile.heartbeat.push(now-window.__profile.last); window.__profile.last=now; }, 16);
  });
  let polls = 0;
  page.on("response", (response) => { if (response.url().includes("/api/workspace/jobs")) polls++; });
  await page.getByLabel("Batch window count").fill("10000");
  await page.getByLabel("Batch stride (samples)").fill("1");
  const pending = page.waitForResponse((r) => r.request().method()==="POST" && r.url().includes("/jobs/batch"));
  await page.getByRole("button", { name: "Start batch classification" }).click();
  const active = await (await pending).json();
  await page.waitForTimeout(6500);
  const after = await chart.evaluate((el) => [...el._fullLayout.xaxis.range]);
  const rendering = await page.evaluate(() => { clearInterval(window.__heartbeat); window.__observer.disconnect(); return window.__profile; });
  await post(`/api/workspace/jobs/${active.id}/cancel`, {});
  const stopped = await terminal(active);
  const heartbeat = rendering.heartbeat.slice(1).sort((a,b)=>a-b);
  const measured = {
    scope: "Synthetic 65,536-sample explicit binary; real HTTP requests and rendered Plotly zoom under genuine CNN job polling. No RF accuracy claim.",
    browser: await browser.version(), requestMs, importRttMs,
    batch: { windows: result.total, wallMsIncludingPolling: batchMs, windowsPerSecond: result.total*1000/batchMs, firstPageModelMs: result.predictions.map((p)=>p.processingMs) },
    chart: { beforeRange: before, afterRange: after, zoomRetained: before.every((v,i)=>Math.abs(v-after[i])<1e-6), removedSvgNodes: rendering.removals, apiPollResponses: polls,
      heartbeatCount: heartbeat.length, heartbeatP95Ms: heartbeat[Math.floor(heartbeat.length*.95)] ?? null, heartbeatMaxMs: heartbeat.at(-1) ?? null, cancelledProbeState: stopped.state },
  };
  await writeFile(output, JSON.stringify(measured, null, 2) + "\n");
  console.log(JSON.stringify({ batchWindowsPerSecond: measured.batch.windowsPerSecond, ...measured.chart }));
} finally { await browser.close(); }
