import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { satmodApi } from "@/lib/satmod-api";
import {
  Surface,
  SectionHeading,
  Field,
  NumberField,
  Message,
  Stats,
  numeric,
} from "@/components/workspace-ui";
import { ScientificChart } from "@/components/scientific-chart";
const topics = [
  [
    "Complex I/Q signals",
    "x[n] = I[n] + jQ[n]; |x| = √(I² + Q²); φ = atan2(Q, I)",
    "Two orthogonal components preserve amplitude and phase at baseband. Sample rate determines the physical frequency axis. Sample amplitudes are relative unless the receiver is calibrated.",
  ],
  [
    "Digital modulation",
    "Rb = Rs log₂(M)",
    "Symbols select waveforms from an alphabet of size M. Bit rate depends on symbol rate and modulation order before coding overhead; spectral efficiency also depends on pulse shaping.",
  ],
  [
    "BPSK",
    "sₖ = exp(jπk), k ∈ {0, 1}",
    "Two antipodal phases encode one bit per symbol. Ideal coherent AWGN BER is Q(√(2Eb/N₀)). Accurate carrier recovery is required.",
  ],
  [
    "QPSK and M-PSK",
    "sₖ = exp(j(2πk/M + θ₀))",
    "QPSK has four phases and two bits per symbol; 8-PSK carries three. Higher order PSK narrows angular decision margins. Gray coding reduces bit errors between adjacent symbols.",
  ],
  [
    "Quadrature amplitude modulation",
    "16-QAM: I,Q ∈ {−3,−1,1,3}/√10",
    "Square 16-QAM carries four bits per symbol and has unit average symbol energy with this scaling. Amplitude compression and phase errors both distort its decision regions.",
  ],
  [
    "APSK satellite constellations",
    "sₖ = rℓ exp(jθₖ); Es = Σℓ Nℓrℓ²/M",
    "16-APSK uses 4+12 points on two rings; 32-APSK uses 4+12+16 on three. Ring ratios and phase offsets depend on the design. The simulator uses illustrative ratios 1:2.53:4.30, not a full DVB-S2 implementation.",
  ],
  [
    "FSK and GMSK",
    "x[n] = exp(jφ[n]); φ[n] = φ[n−1] + 2πf[n]/Fs",
    "Frequency-shift keying encodes information in frequency. The BFSK simulator uses continuous phase with h=0.5. GMSK Gaussian-filters the NRZ frequency pulse at BT=0.3 before phase integration; both have a constant clean envelope. They have no fixed discrete I/Q constellation.",
  ],
  [
    "SNR and AWGN",
    "SNRdB = 10 log₁₀(Ps/Pn); σ²I = σ²Q = Pn/2",
    "Complex AWGN splits noise power equally between I and Q. Es/N₀, Eb/N₀, and sample SNR are different quantities; conversion depends on bits per symbol and samples per symbol.",
  ],
  [
    "RRC pulse shaping",
    "B ≈ (1+α)Rs; 0 < α ≤ 1",
    "Root-raised-cosine transmit and receive filters combine to a raised-cosine response. Finite filter span and timing error can still introduce intersymbol interference. FSK/GMSK use their own frequency pulse shaping.",
  ],
  [
    "Satellite Doppler and carrier offset",
    "Δf = −(vr/c)fc; x′[n] = x[n] exp(j2πΔf n/Fs)",
    "Range rate shifts carrier frequency during a LEO pass. A fixed receiver oscillator error adds CFO. Studio applies a constant offset over a short capture; the orbital explorer below shows an approximate time-varying pass.",
  ],
  [
    "Phase noise and Rician fading",
    "h = √(K/(K+1)) + (a+jb)/√(2(K+1))",
    "K is a linear line-of-sight to scattered-power ratio. K=0 gives Rayleigh fading when enabled. The simulator uses independent scatter per sample and independent Gaussian phase jitter; it does not model a Doppler-correlated channel.",
  ],
  [
    "Saleh TWTA amplifier",
    "A(r) = αa r/(1+βa r²); Φ(r) = αφ r²/(1+βφ r²)",
    "Traveling-wave tube saturation changes amplitude and phase. The simulator applies both AM/AM and AM/PM distortion before fading and AWGN. More amplitude levels can increase sensitivity to nonlinear distortion.",
  ],
  [
    "CNN automatic modulation classification",
    "p = softmax(Wz+b); H(p) = −Σk pk log₂(pk)",
    "The supplied CNN reads 128×2 I/Q windows after the saved scalar z-score transform. Three convolution blocks, batch normalization, pooling, and dense layers estimate BPSK, QPSK, and 16-QAM probabilities. Entropy is ambiguity within those three classes, not a calibrated unknown-signal detector.",
  ],
];
export function TheoryPage() {
  const [scheme, setScheme] = useState("QPSK");
  const [snr, setSnr] = useState(12);
  const [altitude, setAltitude] = useState(550);
  const [carrier, setCarrier] = useState(8.4);
  const [gain, setGain] = useState(2.1587);
  const orbit = useQuery({
    queryKey: ["orbit", altitude, carrier],
    queryFn: () => satmodApi.orbit(altitude, carrier),
    retry: 1,
    enabled:
      Number.isFinite(altitude) &&
      altitude >= 200 &&
      altitude <= 2000 &&
      Number.isFinite(carrier) &&
      carrier > 0 &&
      carrier <= 100,
  });
  const n = scheme === "BPSK" ? 2 : scheme === "QPSK" ? 4 : 8;
  const points =
    scheme === "16-QAM"
      ? [-3, -1, 1, 3].flatMap((i) =>
          [-3, -1, 1, 3].map((q) => ({ i: i / Math.sqrt(10), q: q / Math.sqrt(10) })),
        )
      : Array.from({ length: n }, (_, k) => ({
          i: Math.cos((2 * Math.PI * k) / n + (n === 4 ? Math.PI / 4 : 0)),
          q: Math.sin((2 * Math.PI * k) / n + (n === 4 ? Math.PI / 4 : 0)),
        }));
  const radii = Array.from({ length: 100 }, (_, i) => i * 0.03);
  return (
    <div className="space-y-5">
      <Surface className="p-5">
        <SectionHeading title="Modulation explorer" note="IDEAL EDUCATIONAL REFERENCE" />
        <div className="grid gap-5 lg:grid-cols-2">
          <div>
            <Field label="Ideal constellation">
              <select
                className="control"
                value={scheme}
                onChange={(e) => setScheme(e.target.value)}
              >
                {["BPSK", "QPSK", "8-PSK", "16-QAM"].map((s) => (
                  <option key={s}>{s}</option>
                ))}
              </select>
            </Field>
            <NumberField label="Sample SNR (dB)" value={snr} onChange={setSnr} min={-30} max={60} />
            <Stats
              items={[
                ["Complex noise RMS (Es=1)", numeric(Math.sqrt(10 ** (-snr / 10)))],
                ["Per-quadrature noise σ", numeric(Math.sqrt(10 ** (-snr / 10) / 2))],
              ]}
            />
            <p className="mt-4 text-xs leading-6 text-muted-foreground">
              Points show the ideal symbol alphabet. The noise figures use unit signal power.
              Generate an actual noisy waveform in Satellite Studio. 8-PSK is a simulator and theory
              option; the CNN supports only three classes.
            </p>
          </div>
          <ScientificChart
            data={[
              {
                x: points.map((p) => p.i),
                y: points.map((p) => p.q),
                type: "scatter",
                mode: "markers",
                marker: { size: 12 },
              },
            ]}
            xTitle="In-phase I"
            yTitle="Quadrature Q"
            equal
          />
        </div>
      </Surface>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {topics.map(([title, equation, text]) => (
          <Surface key={title} className="p-5">
            <h2 className="font-display text-lg">{title}</h2>
            <p className="my-4 break-words rounded-sm border border-primary/20 bg-primary/5 p-3 font-mono text-xs leading-6 text-primary">
              {equation}
            </p>
            <p className="text-xs leading-6 text-muted-foreground">{text}</p>
          </Surface>
        ))}
      </div>
      <Surface className="p-5">
        <SectionHeading title="Approximate LEO pass" note="ILLUSTRATIVE GEOMETRY" />
        <div className="grid gap-4 sm:grid-cols-2">
          <NumberField
            label="Altitude (km)"
            value={altitude}
            onChange={setAltitude}
            min={200}
            max={2000}
            step={50}
          />
          <NumberField
            label="Carrier (GHz)"
            value={carrier}
            onChange={setCarrier}
            min={0.1}
            max={100}
            step={0.1}
          />
        </div>
        <Message error={orbit.error} loading={orbit.isPending} />
        {orbit.data && (
          <>
            <ScientificChart
              data={[
                {
                  x: orbit.data.time_sec,
                  y: orbit.data.doppler_khz,
                  mode: "lines",
                  type: "scatter",
                  name: "Doppler",
                },
              ]}
              xTitle="Time from closest approach (s)"
              yTitle="Doppler shift (kHz)"
            />
            <Stats
              items={[
                ["Orbital speed", `${numeric(orbit.data.v_orbit_kms)} km/s`],
                ["Orbital period", `${numeric(orbit.data.period_min)} min`],
              ]}
            />
          </>
        )}
        <p className="mt-3 text-xs text-muted-foreground">
          Approximate straight-line overflight at 78° peak elevation; this is educational geometry,
          not an ephemeris or operational tracker.
        </p>
      </Surface>
      <Surface className="p-5">
        <SectionHeading title="TWTA transfer characteristics" note="SALEH MODEL" />
        <NumberField
          label="AM/AM gain αa"
          value={gain}
          onChange={setGain}
          min={0.1}
          max={4}
          step={0.1}
        />
        <div className="grid lg:grid-cols-2">
          <ScientificChart
            data={[
              {
                x: radii,
                y: radii.map((r) => (gain * r) / (1 + 1.1517 * r * r)),
                type: "scatter",
                mode: "lines",
              },
            ]}
            xTitle="Input amplitude"
            yTitle="Output amplitude"
          />
          <ScientificChart
            data={[
              {
                x: radii,
                y: radii.map((r) => (4.0033 * r * r) / (1 + 9.104 * r * r)),
                type: "scatter",
                mode: "lines",
              },
            ]}
            xTitle="Input amplitude"
            yTitle="Phase rotation (rad)"
          />
        </div>
      </Surface>
    </div>
  );
}
