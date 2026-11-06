import { describe, expect, it } from "vitest";
import {
  advanceRotation,
  DEMO_SATELLITES,
  DISPLAY_ROTATION_SECONDS,
  EARTH_RADIUS_KM,
  earthAngle,
  elevationDeg,
  geographicPosition,
  groundTrack,
  orbitalPeriod,
  rotateY,
  satellitePosition,
  splitDateline,
  subsatellitePoint,
  TAU,
} from "@/lib/orbital-simulation";
describe("Simulated orbital geometry and display timing", () => {
  it.each(DEMO_SATELLITES)("$id maintains a circular orbit at the stated altitude", (sat) => {
    for (const t of [0, 100, 2000, 6000])
      expect(Math.hypot(...satellitePosition(sat, t))).toBeCloseTo(
        1 + sat.altitudeKm / EARTH_RADIUS_KM,
        12,
      );
    const a = satellitePosition(sat, 0),
      b = satellitePosition(sat, orbitalPeriod(sat));
    a.forEach((v, i) => expect(v).toBeCloseTo(b[i]!, 12));
  });
  it.each(DEMO_SATELLITES)(
    "$id map positions and globe positions describe the same point",
    (sat) => {
      for (const t of [0, 530, 4600]) {
        const [lat, lon] = subsatellitePoint(sat, t),
          r = 1 + sat.altitudeKm / EARTH_RADIUS_KM;
        const reconstructed = rotateY(geographicPosition(lat, lon, r), earthAngle(t));
        const actual = satellitePosition(sat, t);
        actual.forEach((v, i) => expect(v).toBeCloseTo(reconstructed[i]!, 10));
        expect(Math.abs(lat)).toBeLessThanOrEqual(
          Math.min(sat.inclinationDeg, 180 - sat.inclinationDeg) + 1e-9,
        );
      }
    },
  );
  it("accounts for Earth rotation rather than repeating the same ground longitude", () => {
    const sat = DEMO_SATELLITES[0]!;
    const first = subsatellitePoint(sat, 0),
      next = subsatellitePoint(sat, orbitalPeriod(sat));
    expect(next[0]).toBeCloseTo(first[0], 10);
    expect(next[1]).not.toBeCloseTo(first[1], 2);
  });
  it("splits antimeridian crossings without drawing a line through the whole map", () => {
    const parts = splitDateline([
      [10, 179],
      [12, -179],
    ]);
    expect(parts).toEqual([
      [
        [10, 179],
        [11, 180],
      ],
      [
        [11, -180],
        [12, -179],
      ],
    ]);
    for (const sat of DEMO_SATELLITES)
      for (const segment of groundTrack(sat, 1300))
        for (let i = 1; i < segment.length; i++)
          expect(Math.abs(segment[i]![1] - segment[i - 1]![1])).toBeLessThanOrEqual(180);
  });
  it("computes a visible initial reference-station pass and an occulted opposite side", () => {
    const sat = DEMO_SATELLITES[0]!;
    expect(elevationDeg(sat, 0)).toBeGreaterThan(80);
    expect(elevationDeg(sat, orbitalPeriod(sat) / 2)).toBeLessThan(0);
  });
  it("uses delta time for a seamless 108-second revolution at different frame rates", () => {
    for (const fps of [20, 30, 60, 120]) {
      let angle = 0;
      for (let n = 0; n < fps * DISPLAY_ROTATION_SECONDS; n++)
        angle = advanceRotation(angle, 1 / fps);
      expect(angle).toBeCloseTo(TAU, 10);
    }
    expect(advanceRotation(TAU, 1)).toBeGreaterThan(TAU);
  });
});
