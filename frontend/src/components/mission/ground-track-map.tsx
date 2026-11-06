import { useEffect, useMemo, useState } from "react";
import { CircleMarker, GeoJSON, MapContainer, Polyline, Tooltip, useMap } from "react-leaflet";
import { CRS, type LatLngExpression } from "leaflet";
import type { FeatureCollection } from "geojson";
import "leaflet/dist/leaflet.css";
import {
  GROUND_STATION,
  groundTrack,
  subsatellitePoint,
  type DemoSatellite,
} from "@/lib/orbital-simulation";
function ResizeMap() {
  const map = useMap();
  useEffect(() => {
    const resize = () => {
      map.invalidateSize();
      map.fitBounds(
        [
          [-87, -180],
          [87, 180],
        ],
        { animate: false, padding: [6, 6] },
      );
    };
    const observer = new ResizeObserver(resize);
    observer.observe(map.getContainer());
    resize();
    return () => observer.disconnect();
  }, [map]);
  return null;
}
export default function GroundTrackMap({
  satellite,
  seconds,
}: {
  satellite: DemoSatellite;
  seconds: number;
}) {
  const [countries, setCountries] = useState<FeatureCollection | null>(null),
    [error, setError] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    fetch("/maps/countries.geojson", { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error("Map unavailable");
        return r.json() as Promise<FeatureCollection>;
      })
      .then(setCountries)
      .catch((e) => {
        if (e instanceof Error && e.name !== "AbortError") setError(true);
      });
    return () => controller.abort();
  }, []);
  const track = useMemo(() => groundTrack(satellite, seconds), [satellite, seconds]);
  const position = subsatellitePoint(satellite, seconds);
  const graticule = useMemo(
    () => [
      ...[-60, -30, 0, 30, 60].map(
        (lat) =>
          [
            [lat, -180],
            [lat, 180],
          ] as LatLngExpression[],
      ),
      ...[-120, -60, 0, 60, 120].map(
        (lon) =>
          [
            [-85, lon],
            [85, lon],
          ] as LatLngExpression[],
      ),
    ],
    [],
  );
  if (error)
    return (
      <div className="map-empty" role="status">
        Geographical basemap unavailable. Orbital calculations remain active.
      </div>
    );
  return (
    <MapContainer
      className="ground-track-map"
      center={[0, 0]}
      zoom={0}
      minZoom={-2}
      maxZoom={4}
      zoomSnap={0}
      crs={CRS.EPSG4326}
      zoomControl={false}
      scrollWheelZoom={false}
      attributionControl={false}
      maxBounds={[
        [-90, -185],
        [90, 185],
      ]}
      maxBoundsViscosity={1}
      aria-label="Simulated satellite ground track on a geographical world map"
    >
      <ResizeMap />
      {countries && (
        <GeoJSON
          data={countries}
          style={{
            className: "world-country",
            color: "#254056",
            weight: 0.65,
            fillColor: "#142b3c",
            fillOpacity: 1,
          }}
        />
      )}
      {graticule.map((line, i) => (
        <Polyline
          key={i}
          positions={line}
          pathOptions={{ color: "#254056", weight: 0.5, opacity: 0.55 }}
        />
      ))}
      {track.map((segment, i) => (
        <Polyline
          key={`${satellite.id}-${i}`}
          positions={segment}
          pathOptions={{ color: satellite.color, weight: 1.6, opacity: 0.9 }}
        >
          <Tooltip sticky>
            {satellite.id} · simulated circular orbit · Earth rotation included
          </Tooltip>
        </Polyline>
      ))}
      <CircleMarker
        center={[GROUND_STATION.latitude, GROUND_STATION.longitude]}
        radius={4}
        pathOptions={{ color: "#00e0a4", fillColor: "#00e0a4", fillOpacity: 1, weight: 1 }}
      >
        <Tooltip>
          {GROUND_STATION.id} · simulated ground station
          <br />
          {GROUND_STATION.latitude.toFixed(2)}° N, {GROUND_STATION.longitude.toFixed(2)}° E
        </Tooltip>
      </CircleMarker>
      <CircleMarker
        center={position}
        radius={5}
        pathOptions={{ color: "#e8f2ff", fillColor: satellite.color, fillOpacity: 1, weight: 1.5 }}
      >
        <Tooltip>
          {satellite.id} · DEMO
          <br />
          {position[0].toFixed(2)}°, {position[1].toFixed(2)}° · {satellite.altitudeKm} km
        </Tooltip>
      </CircleMarker>
    </MapContainer>
  );
}
