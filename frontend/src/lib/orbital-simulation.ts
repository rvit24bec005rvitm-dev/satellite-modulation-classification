/** Circular two-body demonstration, not an ephemeris or live spacecraft feed.
 * ECI uses the equatorial frame; ECEF includes sidereal Earth rotation.
 * The display's 108-second turntable is independent of physical propagation.
 */
export const EARTH_RADIUS_KM = 6371;
export const EARTH_MU = 398600.4418; // km³/s²
export const SIDEREAL_DAY_SECONDS = 86164.0905;
export const SIMULATION_RATE = 30;
export const DISPLAY_ROTATION_SECONDS = 108;
export const TAU = Math.PI * 2;
const DEG = Math.PI / 180;
export type Vec3 = [number, number, number];
export interface DemoSatellite {
  id: string;
  name: string;
  altitudeKm: number;
  inclinationDeg: number;
  raanDeg: number;
  phaseDeg: number;
  color: string;
  orbitType: string;
}
export const DEMO_SATELLITES: readonly DemoSatellite[] = [
  {
    id: "SAT-001",
    name: "AURORA",
    altitudeKm: 550,
    inclinationDeg: 53,
    raanDeg: 67.45,
    phaseDeg: 16.4,
    color: "#00D9FF",
    orbitType: "LEO / inclined",
  },
  {
    id: "SAT-002",
    name: "POLARIS",
    altitudeKm: 1350,
    inclinationDeg: 97.4,
    raanDeg: 132,
    phaseDeg: 155,
    color: "#699CFF",
    orbitType: "LEO / near-polar",
  },
  {
    id: "SAT-003",
    name: "EQUINOX",
    altitudeKm: 1950,
    inclinationDeg: 0,
    raanDeg: 0,
    phaseDeg: 235,
    color: "#00E0A4",
    orbitType: "LEO / equatorial",
  },
];
export const GROUND_STATION = {
  id: "BLR-01",
  name: "Bengaluru reference site",
  latitude: 12.9716,
  longitude: 77.5946,
};
export interface SimulationClock {
  seconds: number;
  visibleSeconds: number;
  enabled: boolean;
  pauseUntil: number;
}
export function orbitalPeriod(satellite: DemoSatellite): number {
  return TAU * Math.sqrt((EARTH_RADIUS_KM + satellite.altitudeKm) ** 3 / EARTH_MU);
}
export function earthAngle(seconds: number): number {
  return (TAU * seconds) / SIDEREAL_DAY_SECONDS;
}
export function geographicPosition(latitude: number, longitude: number, radius = 1): Vec3 {
  const lat = latitude * DEG,
    lon = longitude * DEG;
  return [
    radius * Math.cos(lat) * Math.cos(lon),
    radius * Math.sin(lat),
    -radius * Math.cos(lat) * Math.sin(lon),
  ];
}
export function rotateY([x, y, z]: Vec3, angle: number): Vec3 {
  return [x * Math.cos(angle) + z * Math.sin(angle), y, -x * Math.sin(angle) + z * Math.cos(angle)];
}
export function orbitPoint(satellite: DemoSatellite, anomaly: number): Vec3 {
  const radius = 1 + satellite.altitudeKm / EARTH_RADIUS_KM;
  const node = satellite.raanDeg * DEG,
    inclination = satellite.inclinationDeg * DEG;
  const x =
    Math.cos(node) * Math.cos(anomaly) - Math.sin(node) * Math.sin(anomaly) * Math.cos(inclination);
  const y =
    Math.sin(node) * Math.cos(anomaly) + Math.cos(node) * Math.sin(anomaly) * Math.cos(inclination);
  const z = Math.sin(anomaly) * Math.sin(inclination);
  return [x * radius, z * radius, -y * radius];
}
export function satellitePosition(satellite: DemoSatellite, seconds: number): Vec3 {
  return orbitPoint(
    satellite,
    satellite.phaseDeg * DEG + (TAU * seconds) / orbitalPeriod(satellite),
  );
}
export function groundPosition(seconds: number): Vec3 {
  return rotateY(
    geographicPosition(GROUND_STATION.latitude, GROUND_STATION.longitude, 1.004),
    earthAngle(seconds),
  );
}
export function subsatellitePoint(satellite: DemoSatellite, seconds: number): [number, number] {
  const [x, y, z] = rotateY(satellitePosition(satellite, seconds), -earthAngle(seconds));
  const radius = Math.hypot(x, y, z);
  return [Math.asin(Math.max(-1, Math.min(1, y / radius))) / DEG, Math.atan2(-z, x) / DEG];
}
export function elevationDeg(satellite: DemoSatellite, seconds: number): number {
  const s = satellitePosition(satellite, seconds),
    g = groundPosition(seconds);
  const line: Vec3 = [s[0] - g[0], s[1] - g[1], s[2] - g[2]];
  const dot =
    (line[0] * g[0] + line[1] * g[1] + line[2] * g[2]) / (Math.hypot(...line) * Math.hypot(...g));
  return Math.asin(Math.max(-1, Math.min(1, dot))) / DEG;
}
export function splitDateline(points: [number, number][]): [number, number][][] {
  const segments: [number, number][][] = [];
  let segment: [number, number][] = [];
  for (const point of points) {
    const previous = segment.at(-1);
    if (previous && Math.abs(point[1] - previous[1]) > 180) {
      const adjusted = point[1] + (point[1] < previous[1] ? 360 : -360);
      const edge = previous[1] > 0 ? 180 : -180;
      const latitude =
        previous[0] + ((point[0] - previous[0]) * (edge - previous[1])) / (adjusted - previous[1]);
      segment.push([latitude, edge]);
      segments.push(segment);
      segment = [[latitude, -edge]];
    }
    segment.push(point);
  }
  if (segment.length) segments.push(segment);
  return segments;
}
export function groundTrack(satellite: DemoSatellite, seconds: number): [number, number][][] {
  const period = orbitalPeriod(satellite);
  return splitDateline(
    Array.from({ length: 241 }, (_, i) =>
      subsatellitePoint(satellite, seconds - period / 2 + (i * period) / 240),
    ),
  );
}
export function advanceRotation(angle: number, deltaSeconds: number): number {
  return angle + (deltaSeconds * TAU) / DISPLAY_ROTATION_SECONDS;
}
export function elapsedLabel(seconds: number): string {
  const s = Math.floor(seconds);
  return [Math.floor(s / 3600), Math.floor(s / 60) % 60, s % 60]
    .map((n) => String(n).padStart(2, "0"))
    .join(":");
}
