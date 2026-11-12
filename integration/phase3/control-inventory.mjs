// Rendered control inventory complements, but does not replace, action E2Es.
import { chromium } from "../../frontend/node_modules/@playwright/test/index.mjs";
import { writeFile } from "node:fs/promises";

const [baseURL, destination] = process.argv.slice(2);
if (!baseURL || !destination) throw Error("Usage: node integration/phase3/control-inventory.mjs URL OUTPUT.json");
const browser = await chromium.launch({ channel: "chrome", args: ["--enable-unsafe-swiftshader"] });
const rows = [];
try {
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1080 } });
    for (const path of ["/", "/classifier", "/studio", "/evaluation", "/arena", "/intelligence", "/theory", "/settings"]) {
      await page.goto(baseURL + path);
      await page.getByRole("heading", { level: 1 }).waitFor();
      await page.evaluate(() => {
        document.querySelectorAll("main details").forEach((node) => { node.open = true; });
      });
      await page.waitForTimeout(1200);
      const snapshot = await page.locator("body").ariaSnapshot();
      const rendered = await page.evaluate(() => {
        const visible = (node) => node.getClientRects().length > 0 && getComputedStyle(node).visibility !== "hidden";
        const name = (node) => node.getAttribute("aria-label") ||
          node.getAttribute("aria-labelledby")?.split(/\s+/).map((id) => document.getElementById(id)?.textContent || "").join(" ") ||
          [...(node.labels || [])].map((label) => label.textContent).join(" ") ||
          node.getAttribute("data-title") || node.getAttribute("title") || node.textContent || "";
        return {
          horizontalOverflow: document.documentElement.scrollWidth > innerWidth + 1,
          controls: [...document.querySelectorAll("button, input, select, textarea, summary, a[href], [role=button]")]
            .filter(visible).map((node) => ({
              tag: node.tagName.toLowerCase(), type: node.getAttribute("type"),
              name: name(node).replace(/\s+/g, " ").trim(),
              disabled: Boolean(node.disabled), tabIndex: node.tabIndex,
              href: node.getAttribute("href"),
            })),
          alerts: [...document.querySelectorAll('main [role="alert"]')].filter(visible).map((node) => node.textContent),
        };
      });
      rows.push({ path, width, ...rendered, accessibilitySnapshot: snapshot });
    }
    await page.close();
  }
  const result = {
    scope: "Actual compiled application; all eight routes at 1440 and 390 px, with native details expanded. Inventory and accessibility-tree inspection, not proof that every permutation or assistive technology was exercised. Empty data-dependent controls are included as disabled where appropriate. Action coverage is recorded separately in browser tests.",
    browser: await browser.version(), rows,
  };
  await writeFile(destination, JSON.stringify(result, null, 2) + "\n");
  console.log(JSON.stringify(rows.map(({ path, width, horizontalOverflow, controls, alerts }) => ({ path, width, horizontalOverflow, controlCount: controls.length, unnamedControls: controls.filter((node) => !node.name).length, alerts }))));
} finally {
  await browser.close();
}
