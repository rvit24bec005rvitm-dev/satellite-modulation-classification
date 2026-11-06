import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Message, Stats, percent, Empty } from "@/components/workspace-ui";
describe("Workspace feedback and measured values", () => {
  it("renders the actual service error as an accessible alert", () => {
    const markup = renderToStaticMarkup(<Message error={new Error("Capture contains NaN")} />);
    expect(markup).toContain('role="alert"');
    expect(markup).toContain("Capture contains NaN");
  });
  it("does not invent an error or loading message when idle", () => {
    expect(renderToStaticMarkup(<Message />)).toBe("");
  });
  it("shows a status while connecting", () => {
    expect(renderToStaticMarkup(<Message loading />)).toContain('role="status"');
  });
  it("preserves zero as a measured value", () => {
    expect(renderToStaticMarkup(<Stats items={[["Signal power", 0]]} />)).toContain(">0<");
  });
  it("distinguishes unavailable metrics from a real zero score", () => {
    expect(percent(undefined)).toBe("—");
    expect(percent(0)).toBe("0.00%");
  });
  it("explains the empty capture state", () => {
    expect(renderToStaticMarkup(<Empty />)).toContain("Select or generate a capture");
  });
});
