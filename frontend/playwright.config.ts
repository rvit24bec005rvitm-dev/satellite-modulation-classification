import { defineConfig, devices } from "@playwright/test";
const python =
  process.env["SATMOD_PYTHON"] ??
  (process.platform === "win32" ? "..\\.venv\\Scripts\\python.exe" : "../.venv/bin/python");
const baseURL = process.env["SATMOD_TEST_BASE_URL"] ?? "http://127.0.0.1:5173";
const apiURL = process.env["SATMOD_TEST_API_URL"] ?? "http://127.0.0.1:8000";
const apiPort = new URL(apiURL).port || "8000";
const uiPort = new URL(baseURL).port || "5173";
export default defineConfig({
  testDir: "./e2e",
  timeout: 60000,
  workers: 1,
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    ...devices["Desktop Chrome"],
    ...(process.env["PLAYWRIGHT_CHANNEL"] ? { channel: process.env["PLAYWRIGHT_CHANNEL"] } : {}),
  },
  webServer: [
    {
      command: `"${python}" -m uvicorn backend.main:app --app-dir .. --host 127.0.0.1 --port ${apiPort}`,
      url: `${apiURL}/api/health`,
      reuseExistingServer: !process.env["CI"],
      timeout: 60000,
    },
    {
      command: "npm run dev",
      url: baseURL,
      env: { VITE_API_PROXY_TARGET: apiURL, VITE_DEV_PORT: uiPort },
      reuseExistingServer: !process.env["CI"],
      timeout: 60000,
    },
  ],
});
