import {
  expect,
  request as createRequest,
  test,
  type APIRequestContext,
  type Page,
} from "@playwright/test";
import { spawn, type ChildProcess } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// Disposable local fixture credentials. No hosted credentials or user data.
const token = "synthetic-protected-browser-fixture-token";
const repo = fileURLToPath(new URL("../..", import.meta.url));
let origin: string;
let directory: string;
let child: ChildProcess | undefined;
let api: APIRequestContext | undefined;
let stderr = "";

test.beforeAll(async () => {
  directory = await mkdtemp(join(tmpdir(), "satmod-protected-browser-"));
  const listener = createServer();
  listener.listen(0, "127.0.0.1");
  await once(listener, "listening");
  const address = listener.address();
  if (!address || typeof address === "string") throw new Error("No fixture port assigned");
  const port = address.port;
  await new Promise<void>((done, fail) =>
    listener.close((error) => (error ? fail(error) : done())),
  );
  origin = `http://127.0.0.1:${port}`;
  const configuredPython = process.env["SATMOD_PYTHON"];
  const python = configuredPython
    ? /[/\\]/.test(configuredPython)
      ? resolve(process.cwd(), configuredPython)
      : configuredPython
    : join(repo, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
  child = spawn(
    python,
    ["-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", String(port)],
    {
      cwd: repo,
      env: {
        ...process.env,
        SATMOD_PUBLIC: "1",
        SATMOD_ACCESS_TOKEN: token,
        SATMOD_DATA_DIR: directory,
        SATMOD_SERVE_FRONTEND: "1",
        WEB_CONCURRENCY: "1",
        SATMOD_REPLICAS: "1",
      },
      stdio: ["ignore", "ignore", "pipe"],
    },
  );
  let spawnError: Error | undefined;
  child.on("error", (error) => {
    spawnError = error;
  });
  child.stderr?.on("data", (data) => {
    stderr = (stderr + String(data)).slice(-4000);
  });
  api = await createRequest.newContext({ baseURL: origin });
  await expect
    .poll(
      async () => {
        if (spawnError) throw spawnError;
        if (child?.exitCode !== null) throw new Error(`Fixture server exited: ${stderr}`);
        return (await api!.get("/api/ready").catch(() => null))?.status();
      },
      { timeout: 30000 },
    )
    .toBe(200);
});

test.afterAll(async () => {
  await api?.dispose();
  if (child && child.exitCode === null && child.signalCode === null) {
    const exited = once(child, "close");
    child.kill("SIGTERM");
    const deadline = setTimeout(() => child?.kill("SIGKILL"), 10000);
    await exited;
    clearTimeout(deadline);
  }
  if (directory) await rm(directory, { recursive: true, force: true });
});

async function unlock(page: Page) {
  await page.getByLabel("Workspace access token", { exact: true }).fill(token);
  await page.getByRole("button", { name: "Unlock workspace", exact: true }).click();
  await expect(page.getByRole("button", { name: "Lock workspace", exact: true })).toBeVisible();
}

test("a public host rejects anonymous API access and verifies access again after reload", async ({
  page,
}) => {
  expect((await api!.get("/api/models")).status()).toBe(401);
  await page.goto(origin + "/classifier");
  await expect(page.getByRole("heading", { name: "Unlock your workspace" })).toBeVisible();
  await expect(page.getByLabel("Upload I/Q file")).toHaveCount(0);
  await page
    .getByLabel("Workspace access token", { exact: true })
    .fill("incorrect-synthetic-token");
  await page.getByRole("button", { name: "Unlock workspace", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("Workspace access token required");
  expect(await page.evaluate(() => sessionStorage.getItem("satmod-access-token:"))).toBeNull();
  await unlock(page);
  await expect(page.getByLabel("Upload I/Q file")).toBeVisible();
  const verified = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/session",
  );
  await page.reload();
  expect((await verified).status()).toBe(200);
  await expect(page.getByLabel("Upload I/Q file")).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem("satmod-access-token:"))).toBeNull();
  expect(page.url()).not.toContain(token);
});

test("token editing preserves the valid session and locking clears actual classification context", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(origin + "/settings");
  await unlock(page);
  await page.getByLabel("Workspace access token (session only)").fill("incorrect-draft-token");
  expect(await page.evaluate(() => sessionStorage.getItem("satmod-access-token:"))).toBe(token);
  const rejected = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/session",
  );
  await page.getByRole("button", { name: "Verify workspace token", exact: true }).click();
  expect((await rejected).status()).toBe(401);
  await expect(page.getByRole("alert").first()).toContainText("Workspace access token required");
  expect(await page.evaluate(() => sessionStorage.getItem("satmod-access-token:"))).toBe(token);
  await expect(page.getByRole("button", { name: "Lock workspace", exact: true })).toBeVisible();
  await page.goto(origin + "/classifier");
  const sample = await api!.get("/api/samples/97", { headers: { "X-Satmod-Token": token } });
  expect(sample.ok()).toBeTruthy();
  const data = (await sample.json()) as { samples: { i: number; q: number }[] };
  const classified = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/classify/upload",
  );
  await page.getByLabel("Upload I/Q file").setInputFiles({
    name: "synthetic-protected-fixture.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("I,Q\n" + data.samples.map((point) => `${point.i},${point.q}`).join("\n")),
  });
  const result = await classified;
  expect(result.status()).toBe(200);
  const probabilities = Object.values((await result.json()).probabilities) as number[];
  expect(probabilities).toHaveLength(3);
  expect(probabilities.every(Number.isFinite)).toBe(true);
  expect(probabilities.reduce((total, value) => total + value, 0)).toBeCloseTo(1, 5);
  await expect(page.getByText("Shannon entropy", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Lock workspace", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Unlock your workspace" })).toBeVisible();
  expect(await page.evaluate(() => sessionStorage.getItem("satmod-access-token:"))).toBeNull();
  await unlock(page);
  await expect(page.getByText("Shannon entropy", { exact: true })).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});

test("a delayed genuine token verification cannot restore access after the operator locks", async ({
  page,
}) => {
  await page.goto(origin + "/settings");
  await unlock(page);
  let release!: () => void;
  let requested!: () => void;
  let finished!: () => void;
  const blocked = new Promise<void>((done) => {
    release = done;
  });
  const received = new Promise<void>((done) => {
    requested = done;
  });
  const delivered = new Promise<void>((done) => {
    finished = done;
  });
  await page.route("**/api/session", async (route) => {
    const response = await route.fetch();
    expect(response.status()).toBe(200);
    requested();
    await blocked;
    await route.fulfill({ response }).catch(() => {});
    finished();
  });
  await page.getByRole("button", { name: "Verify workspace token", exact: true }).click();
  await received;
  await page.getByRole("button", { name: "Lock workspace", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Unlock your workspace" })).toBeVisible();
  release();
  await delivered;
  await page.unroute("**/api/session");
  expect(await page.evaluate(() => sessionStorage.getItem("satmod-access-token:"))).toBeNull();
  await page.reload();
  await expect(page.getByRole("heading", { name: "Unlock your workspace" })).toBeVisible();
});

test("an actual audio-export authorization failure locks the protected workspace", async ({
  page,
}) => {
  await page.goto(origin + "/studio");
  await unlock(page);
  await page.getByLabel("Capture length (samples)").fill("512");
  await page.getByRole("button", { name: "Generate signal", exact: true }).click();
  await expect(page.getByText("Generated capture", { exact: true })).toBeVisible();
  await page.evaluate(() =>
    sessionStorage.setItem("satmod-access-token:", "revoked-browser-fixture-token"),
  );
  const denied = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/signal/audio",
  );
  await page.getByRole("button", { name: "Download audible heterodyne", exact: true }).click();
  expect((await denied).status()).toBe(401);
  await expect(page.getByRole("heading", { name: "Unlock your workspace" })).toBeVisible();
  expect(await page.evaluate(() => sessionStorage.getItem("satmod-access-token:"))).toBeNull();
});
