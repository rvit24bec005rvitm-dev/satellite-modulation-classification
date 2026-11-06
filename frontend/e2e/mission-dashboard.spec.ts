import { expect, test, type Page } from "@playwright/test";
const viewport = (page: Page) => page.getByTestId("earth-viewport");
async function openGlobe(page: Page) {
  await page.goto("/");
  await expect(viewport(page)).toHaveAttribute("data-frames", "active", { timeout: 45000 });
  await expect(page.locator(".ground-track-map .world-country").first()).toBeAttached();
}
async function rotation(page: Page) {
  return Number(await viewport(page).getAttribute("data-rotation"));
}

test("mission dashboard renders real WebGL, geographical map, and honest data states", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  await openGlobe(page);
  await expect(page.locator(".earth-viewport canvas")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Mission Control", exact: true })).toBeVisible();
  await expect(page.getByText("Understand the signal.", { exact: false })).toHaveCount(0);
  await expect(page.locator(".probability-track > i")).toHaveCount(0);
  await expect(page.getByText("BACKEND CONNECTED", { exact: true })).toBeVisible();
  await expect(page.getByText("Simulated geometry · no live spacecraft or RF link.")).toBeVisible();
  expect(
    await page.locator(".ground-track-map").evaluate((e) => getComputedStyle(e).backgroundColor),
  ).toBe("rgb(7, 20, 33)");
  await page.screenshot({ path: "../integration/validation/mission-desktop.png", fullPage: true });
  expect(errors).toEqual([]);
});

test("Earth rotates continuously without pointer input and obeys its pause toggle", async ({
  page,
}) => {
  await openGlobe(page);
  const before = await rotation(page);
  await page.waitForTimeout(1700);
  const after = await rotation(page);
  expect(after - before).toBeGreaterThan(0.025);
  await page.getByRole("button", { name: "Toggle Earth auto rotation" }).click();
  await expect(page.getByRole("button", { name: "Toggle Earth auto rotation" })).toHaveAttribute(
    "aria-pressed",
    "false",
  );
  await page.waitForTimeout(700);
  const paused = await rotation(page);
  await page.waitForTimeout(900);
  expect(await rotation(page)).toBeCloseTo(paused, 5);
  await page.getByRole("button", { name: "Toggle Earth auto rotation" }).click();
  await page.waitForTimeout(1000);
  expect(await rotation(page)).toBeGreaterThan(paused);
});

test("camera zoom/reset, scroll zoom, and drag interaction work", async ({ page }) => {
  await openGlobe(page);
  await page.waitForTimeout(400);
  const initial = Number(await viewport(page).getAttribute("data-camera-distance"));
  await page.getByRole("button", { name: "Zoom in", exact: true }).click();
  await expect
    .poll(async () => Number(await viewport(page).getAttribute("data-camera-distance")))
    .toBeLessThan(initial);
  await page.getByRole("button", { name: "Zoom out", exact: true }).click();
  await page.getByRole("button", { name: "Reset globe camera", exact: true }).click();
  await expect
    .poll(async () => Number(await viewport(page).getAttribute("data-camera-distance")))
    .toBeCloseTo(initial, 2);
  const bounds = await page.locator(".earth-viewport canvas").boundingBox();
  expect(bounds).not.toBeNull();
  const x = bounds!.x + bounds!.width / 2,
    y = bounds!.y + bounds!.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.wheel(0, -300);
  await expect
    .poll(async () => Number(await viewport(page).getAttribute("data-camera-distance")))
    .toBeLessThan(initial);
  await page.getByRole("button", { name: "Reset globe camera", exact: true }).click();
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + 75, y + 25, { steps: 15 });
  await page.waitForTimeout(650);
  const during = await rotation(page);
  await page.waitForTimeout(700);
  expect(await rotation(page)).toBeCloseTo(during, 4);
  await page.mouse.up();
  await page.waitForTimeout(2000);
  expect(await rotation(page)).toBeGreaterThan(during);
});

test("satellite selector and clickable marker labels update satellite details", async ({
  page,
}) => {
  await openGlobe(page);
  await page.getByRole("button", { name: "Toggle Earth auto rotation" }).click();
  await page.getByLabel("Selected satellite").selectOption("SAT-002");
  await expect(page.getByText("1,350 km", { exact: true })).toBeVisible();
  await expect(page.locator(".selected-orbit-dot")).toContainText("SAT-002");
  const visible = page.locator(".satellite-label:visible").first();
  await expect(visible).toBeVisible();
  const label = (await visible.textContent())!;
  await visible.click();
  const id = label.match(/SAT-00[123]/)![0];
  await expect(page.getByLabel("Selected satellite")).toHaveValue(id);
});

test("shared classifier result populates actual three-class probabilities on Overview", async ({
  page,
}) => {
  await page.goto("/classifier");
  await page.getByRole("button", { name: "Load demo sample" }).click();
  await page.getByRole("button", { name: "Run CNN classification" }).click();
  await expect(page.getByText("Shannon entropy", { exact: true })).toBeVisible();
  await page.getByRole("link", { name: "Overview", exact: true }).click();
  await expect(page.locator(".probability-track > i")).toHaveCount(3);
  const widths = await page
    .locator(".probability-track > i")
    .evaluateAll((elements) => elements.map((e) => parseFloat((e as HTMLElement).style.width)));
  expect(widths.reduce((a, b) => a + b, 0)).toBeCloseTo(100, 3);
});

test("reduced-motion preference disables automatic movement by default", async ({ browser }) => {
  const context = await browser.newContext({ reducedMotion: "reduce" });
  const page = await context.newPage();
  await page.goto("/");
  await expect(page.locator(".earth-viewport canvas")).toBeVisible({ timeout: 45000 });
  await expect(page.getByRole("button", { name: "Toggle Earth auto rotation" })).toHaveAttribute(
    "aria-pressed",
    "false",
  );
  await page.waitForTimeout(1000);
  const before = await rotation(page);
  await page.waitForTimeout(1100);
  expect(await rotation(page)).toBeCloseTo(before, 5);
  await context.close();
});

test("backend offline state is explicit while the demonstration remains usable", async ({
  page,
}) => {
  await page.route("**/api/**", (r) => r.abort("connectionrefused"));
  await openGlobe(page);
  await expect(page.getByText("BACKEND DISCONNECTED", { exact: true })).toBeVisible();
  await expect(page.getByText("BACKEND OFFLINE", { exact: true })).toBeVisible();
  await expect(page.locator(".probability-track > i")).toHaveCount(0);
});

test("WebGL unavailable state offers an honest geographical fallback", async ({ page }) => {
  await page.addInitScript(() => {
    const getContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (
      this: HTMLCanvasElement,
      kind: string,
      ...args: unknown[]
    ) {
      if (kind === "webgl" || kind === "webgl2" || kind === "experimental-webgl") return null;
      return Reflect.apply(getContext, this, [kind, ...args]);
    } as typeof getContext;
  });
  await page.goto("/");
  await expect(page.getByText("3D view unavailable", { exact: true })).toBeVisible({
    timeout: 45000,
  });
  await expect(page.getByLabel("Selected satellite")).toBeVisible();
});

for (const width of [1440, 1024, 820, 390])
  test(`mission layout remains usable at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await openGlobe(page);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
      true,
    );
    await expect(page.getByLabel("Selected satellite")).toBeVisible();
    const upper = await page.locator(".mission-upper").boundingBox();
    const lower = await page.locator(".mission-lower").boundingBox();
    const globe = await page.locator(".mission-globe-panel").boundingBox();
    expect(globe!.y + globe!.height).toBeLessThanOrEqual(upper!.y + upper!.height + 1);
    expect(lower!.y).toBeGreaterThanOrEqual(upper!.y + upper!.height);
    const clipped = await page
      .locator(".satellite-info, .mission-right-column > section")
      .evaluateAll((elements) =>
        elements
          .filter((e) => e.scrollHeight > e.clientHeight + 2)
          .map((e) => e.querySelector("h2")?.textContent),
      );
    expect(clipped).toEqual([]);
    await page.screenshot({
      path: `../integration/validation/mission-${width}.png`,
      fullPage: true,
    });
    if (width <= 1100) {
      await page.getByRole("button", { name: "Open navigation" }).click();
      await page.getByRole("link", { name: "Live classifier", exact: true }).click();
      await expect(page).toHaveURL(/classifier/);
    }
  });
