// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { TimeFrequencyInspector } from "@/components/time-frequency-inspector";
import { researchApi, type TimeFrequencyReport } from "@/lib/research-api";
import type { Capture } from "@/lib/satmod-api";

const { chart } = vi.hoisted(() => ({ chart: vi.fn() }));
vi.mock("@/components/scientific-chart", () => ({
  ScientificChart: (props: { title: string; xTitle: string; yTitle: string }) => {
    chart(props);
    return <div role="img" aria-label={props.title}>{props.xTitle} / {props.yTitle}</div>;
  },
}));

const report: TimeFrequencyReport = {
  frequencies: [-125, 0, 125], times: [.004, .008], db: [[-50, -49], [-30, -29], [-12, -11]],
  frequencyUnit: "Hz", timeUnit: "s", densityUnit: "dB re 1 sample-unit²/Hz",
  parameters: { fftSize: 64, overlap: .5, window: "hann" },
  note: "Actual sampled windows; not independent measurements or validated carrier detections.",
  peaks: [{ frequency: 125, densityDb: -12 }],
};
function capture(source = "Generated UI fixture"): Capture {
  return { source, sampleRate: 8000,
    samples: Array.from({ length: 128 }, (_, i) => ({ i: .01 * Math.cos(i), q: .01 * Math.sin(i) })),
    analysis: {} as Capture["analysis"],
  };
}

beforeEach(() => {
  chart.mockClear();
  vi.spyOn(researchApi, "timeFrequency").mockResolvedValue(report);
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("Computed time-frequency inspector", () => {
  it("requests actual samples and presents units and complete-window limitations", async () => {
    const input = capture();
    render(<TimeFrequencyInspector capture={input} />);
    await screen.findByRole("img", { name: "Computed time-frequency PSD" });
    expect(researchApi.timeFrequency).toHaveBeenCalledWith(input.samples, 8000, 64, .5, "hann", undefined);
    expect(screen.getByText(report.note)).toBeInTheDocument();
    expect(screen.getByText(/Density unit: dB re 1 sample-unit/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Export time-frequency JSON" })).toBeEnabled();
  });

  it("applies operator FFT/overlap/window settings only on explicit recomputation", async () => {
    const input = capture();
    render(<TimeFrequencyInspector capture={input} />);
    await screen.findByRole("img", { name: "Computed time-frequency PSD" });
    fireEvent.change(screen.getByLabelText("STFT FFT size", { exact: true }), { target: { value: "32" } });
    fireEvent.change(screen.getByLabelText("STFT overlap fraction", { exact: true }), { target: { value: "0.25" } });
    fireEvent.change(screen.getByLabelText("STFT window", { exact: true }), { target: { value: "boxcar" } });
    expect(researchApi.timeFrequency).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Recompute time-frequency analysis" }));
    await waitFor(() => expect(researchApi.timeFrequency).toHaveBeenLastCalledWith(input.samples, 8000, 32, .25, "boxcar", undefined));
  });

  it("adds an explicit operator marker without altering the actual PSD or rerunning DSP", async () => {
    render(<TimeFrequencyInspector capture={capture()} />);
    await screen.findByRole("img", { name: "Computed time-frequency PSD" });
    fireEvent.click(screen.getByRole("button", { name: /Peak 125 Hz/ }));
    expect(screen.getByLabelText("Baseband frequency marker")).toHaveValue(125);
    const props = chart.mock.calls.at(-1)![0];
    expect(props.data[0].z).toBe(report.db);
    expect(props.shapes[0]).toMatchObject({ y0: 125, y1: 125, type: "line" });
    expect(researchApi.timeFrequency).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Clear frequency marker" }));
    expect(screen.getByLabelText("Baseband frequency marker")).toHaveValue(null);
  });

  it("uses only computed density values in the optional waterfall and nearest-bin trace", async () => {
    render(<TimeFrequencyInspector capture={capture()} />);
    await screen.findByRole("img", { name: "Computed time-frequency PSD" });
    fireEvent.change(screen.getByLabelText("Baseband frequency marker"), { target: { value: "120" } });
    fireEvent.change(screen.getByLabelText("Time-frequency view", { exact: true }), { target: { value: "waterfall" } });
    await screen.findByRole("img", { name: "Computed PSD waterfall" });
    const props = chart.mock.calls.at(-1)![0];
    expect(props.data[0]).toMatchObject({ type: "surface", z: report.db });
    expect(props.data[1]).toMatchObject({ type: "scatter3d", y: [125, 125], z: report.db[2] });
    expect(props.zTitle).toBe(report.densityUnit);
    expect(screen.getByText(/operator annotation/)).toBeInTheDocument();
  });

  it("shows backend errors without displaying a fabricated spectral plot", async () => {
    vi.mocked(researchApi.timeFrequency).mockRejectedValue(new Error("Selected clock metadata is inconsistent"));
    render(<TimeFrequencyInspector capture={capture()} />);
    await screen.findByText("Selected clock metadata is inconsistent");
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Export time-frequency JSON" })).toBeDisabled();
  });

  it("ignores a late response from a previously selected capture", async () => {
    let resolveOld: (value: TimeFrequencyReport) => void = () => {};
    vi.mocked(researchApi.timeFrequency).mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve; }))
      .mockResolvedValueOnce({ ...report, note: "New capture result" });
    const view = render(<TimeFrequencyInspector capture={capture("Old fixture")} />);
    expect(screen.getByRole("status")).toHaveTextContent("Computing actual");
    view.rerender(<TimeFrequencyInspector capture={capture("New fixture")} />);
    await screen.findByText("New capture result");
    await act(async () => { resolveOld({ ...report, note: "Old capture result" }); });
    expect(screen.queryByText("Old capture result")).not.toBeInTheDocument();
    expect(screen.getByText("New capture result")).toBeInTheDocument();
  });

  it("sends explicit known-clock metadata rather than assuming Fs=1 is physical", async () => {
    const input = { ...capture(), sampleRate: 1, metadata: { sampleRateKnown: true } };
    render(<TimeFrequencyInspector capture={input} />);
    await screen.findByRole("img", { name: "Computed time-frequency PSD" });
    expect(researchApi.timeFrequency).toHaveBeenCalledWith(input.samples, 1, 64, .5, "hann", true);
  });
});
