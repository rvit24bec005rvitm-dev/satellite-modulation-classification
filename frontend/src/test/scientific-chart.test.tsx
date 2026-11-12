// @vitest-environment jsdom
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ScientificChart } from "@/components/scientific-chart";

const plot = vi.hoisted(() => ({
  react: vi.fn(async (...args: unknown[]) => {
    void args;
  }),
  purge: vi.fn(),
  Plots: { resize: vi.fn() },
}));
vi.mock("plotly.js-dist-min", () => ({ default: plot }));
vi.mock("@/hooks/use-satmod", () => ({
  useSignal: () => ({ preferences: { charts: "interactive", theme: "dark" } }),
}));
vi.mock("@/components/time-frequency-inspector", () => ({ TimeFrequencyInspector: () => null }));
beforeEach(() => {
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    clearRect() {},
    fillRect() {},
    fillStyle: "",
    getImageData: () => ({ data: new Uint8ClampedArray([20, 40, 60, 255]) }),
  } as unknown as CanvasRenderingContext2D);
});
afterEach(async () => {
  await act(async () => cleanup());
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("Scientific plot update and resource contract", () => {
  it("serializes slow plot updates so an older completion cannot purge the selected data", async () => {
    let finishFirst: () => void = () => {};
    plot.react.mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          finishFirst = resolve;
        }),
    );
    const { rerender, unmount } = render(
      <ScientificChart title="Queued plot" data={[{ x: [0], y: [1] }]} />,
    );
    await waitFor(() => expect(plot.react).toHaveBeenCalledTimes(1));
    rerender(<ScientificChart title="Queued plot" data={[{ x: [0], y: [2] }]} />);
    await act(async () => {
      await Promise.resolve();
    });
    expect(plot.react).toHaveBeenCalledTimes(1);
    expect(plot.purge).not.toHaveBeenCalled();
    await act(async () => finishFirst());
    await waitFor(() => expect(plot.react).toHaveBeenCalledTimes(2));
    expect(plot.react.mock.calls[1]?.[1]).toEqual([{ x: [0], y: [2] }]);
    await waitFor(() =>
      expect(screen.getByRole("group", { name: "Queued plot" })).toHaveAttribute(
        "aria-busy",
        "false",
      ),
    );
    expect(plot.purge).not.toHaveBeenCalled();
    unmount();
    await waitFor(() => expect(plot.purge).toHaveBeenCalledTimes(1));
  });
  it("does not purge a zoomed plot for newly allocated equivalent polling data", async () => {
    const { rerender } = render(
      <ScientificChart
        data={[{ x: [0, 1, 2], y: [0.1, 0.2, 0.3] }]}
        xTitle="Sample"
        yTitle="Power"
      />,
    );
    await waitFor(() => expect(plot.react).toHaveBeenCalledTimes(1));
    await act(async () =>
      rerender(
        <ScientificChart
          data={[{ x: [0, 1, 2], y: [0.1, 0.2, 0.3] }]}
          xTitle="Sample"
          yTitle="Power"
        />,
      ),
    );
    expect(plot.react).toHaveBeenCalledTimes(1);
    expect(plot.purge).not.toHaveBeenCalled();
  });
  it("updates real changed values and closes rendering resources on unmount", async () => {
    const { rerender, unmount } = render(<ScientificChart data={[{ x: [0, 1], y: [1, 2] }]} />);
    await waitFor(() => expect(plot.react).toHaveBeenCalledTimes(1));
    rerender(<ScientificChart data={[{ x: [0, 1], y: [1, 3] }]} />);
    await waitFor(() => expect(plot.react).toHaveBeenCalledTimes(2));
    expect(plot.react.mock.calls[1]?.[1]).toEqual([{ x: [0, 1], y: [1, 3] }]);
    expect(plot.purge).not.toHaveBeenCalled();
    unmount();
    await waitFor(() => expect(plot.purge).toHaveBeenCalledTimes(1));
  });
  it("retains unavailable values and correct physical axis names in accessible data", async () => {
    render(
      <ScientificChart
        data={[{ name: "Measured phase", x: [0, 1], y: [0, null] }]}
        xTitle="Time (s)"
        yTitle="Phase (rad)"
      />,
    );
    await waitFor(() => expect(plot.react).toHaveBeenCalledTimes(1));
    expect(screen.getByRole("group", { name: "Phase (rad) versus Time (s)" })).toHaveAttribute(
      "tabindex",
      "0",
    );
    expect(screen.getByText("Unavailable")).toBeInTheDocument();
    expect(screen.getAllByText("Phase (rad)").length).toBeGreaterThan(0);
    expect(plot.react.mock.calls[0]?.[3]).toMatchObject({
      modeBarButtonsToRemove: ["sendChartToCloud"],
    });
  });
});
