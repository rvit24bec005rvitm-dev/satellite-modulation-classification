import {
  Component,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ComponentRef,
  type RefObject,
  type ReactNode,
} from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html, Line, OrbitControls, useTexture } from "@react-three/drei";
import {
  AdditiveBlending,
  BackSide,
  BufferGeometry,
  Float32BufferAttribute,
  Group,
  Mesh,
  Line as ThreeLine,
  LineBasicMaterial,
  MeshPhongMaterial,
  SRGBColorSpace,
  Vector2,
  Vector3,
} from "three";
import { Minus, Plus, RotateCcw, Pause, Play, Maximize2, Globe2 } from "lucide-react";
import {
  advanceRotation,
  DEMO_SATELLITES,
  earthAngle,
  elevationDeg,
  groundPosition,
  orbitPoint,
  satellitePosition,
  type DemoSatellite,
  type SimulationClock,
} from "@/lib/orbital-simulation";

interface GlobeProps {
  clock: SimulationClock;
  selected: DemoSatellite;
  onSelect: (id: string) => void;
  autoRotate: boolean;
  onAutoRotate: () => void;
  motionEnabled: boolean;
}
interface ControlsApi {
  zoom: (factor: number) => void;
  reset: () => void;
}
const CAMERA: [number, number, number] = [2.2, 1.05, -3.0];
const SUN = new Vector3(-3, 3, -5).normalize();
const AXIS = new Vector3(0, 1, 0);
const ASSETS = {
  day: "/textures/earth-day.jpg",
  night: "/textures/earth-night.png",
  normal: "/textures/earth-normal.jpg",
  specular: "/textures/earth-specular.jpg",
  clouds: "/textures/earth-clouds.png",
};

function EarthSurface({ sun, occluder }: { sun: Vector3; occluder: RefObject<Mesh> }) {
  const textures = useTexture(ASSETS);
  useEffect(() => {
    [textures.day, textures.night, textures.clouds].forEach((texture) => {
      texture.colorSpace = SRGBColorSpace;
      texture.anisotropy = 4;
      texture.needsUpdate = true;
    });
  }, [textures.day, textures.night, textures.clouds]);
  const shade = useCallback(
    (shader: Parameters<MeshPhongMaterial["onBeforeCompile"]>[0]) => {
      shader.uniforms["missionSun"] = { value: sun };
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", "#include <common>\nvarying vec3 missionNormal;")
        .replace(
          "#include <begin_vertex>",
          "#include <begin_vertex>\nmissionNormal = normalize(mat3(modelMatrix) * objectNormal);",
        );
      shader.fragmentShader = shader.fragmentShader
        .replace(
          "#include <common>",
          "#include <common>\nvarying vec3 missionNormal;\nuniform vec3 missionSun;",
        )
        .replace(
          "#include <emissivemap_fragment>",
          "#include <emissivemap_fragment>\ntotalEmissiveRadiance *= 1.0 - smoothstep(-0.12,0.22,dot(normalize(missionNormal),normalize(missionSun)));",
        );
    },
    [sun],
  );
  return (
    <>
      <mesh ref={occluder} name="earth-surface">
        <sphereGeometry args={[1, 96, 64]} />
        <meshPhongMaterial
          map={textures.day}
          normalMap={textures.normal}
          normalScale={new Vector2(0.35, 0.35)}
          specularMap={textures.specular}
          specular="#2b5576"
          shininess={12}
          emissiveMap={textures.night}
          emissive="#ffd7a0"
          emissiveIntensity={1.55}
          onBeforeCompile={shade}
        />
      </mesh>
      <mesh scale={1.009} name="earth-clouds">
        <sphereGeometry args={[1, 64, 48]} />
        <meshPhongMaterial
          map={textures.clouds}
          alphaMap={textures.clouds}
          opacity={0.38}
          transparent
          depthWrite={false}
        />
      </mesh>
      {/* Soft local glow — frames Earth without a hard outline */}
      <mesh scale={1.06} name="earth-soft-halo">
        <sphereGeometry args={[1, 48, 32]} />
        <shaderMaterial
          transparent
          side={BackSide}
          depthWrite={false}
          blending={AdditiveBlending}
          vertexShader={
            "varying vec3 n; varying vec3 v; void main(){ vec4 p=modelViewMatrix*vec4(position,1.0); n=normalize(normalMatrix*normal); v=normalize(-p.xyz); gl_Position=projectionMatrix*p; }"
          }
          fragmentShader={
            "varying vec3 n; varying vec3 v; void main(){ float rim=pow(1.0-abs(dot(normalize(n),normalize(v))),4.5); gl_FragColor=vec4(0.35,0.62,1.0,rim*0.18); }"
          }
        />
      </mesh>
    </>
  );
}
function makeStarShell(count: number, radius: number, seed0: number) {
  const data = new Float32Array(count * 3);
  let seed = seed0;
  for (let i = 0; i < count; i++) {
    seed = (seed * 16807) % 2147483647;
    const z = 2 * (seed / 2147483647) - 1;
    seed = (seed * 16807) % 2147483647;
    const jitter = 0.85 + 0.3 * (seed / 2147483647);
    const angle = i * Math.PI * (3 - Math.sqrt(5));
    const r = radius * jitter;
    const xy = Math.sqrt(Math.max(0, 1 - z * z));
    data[i * 3] = r * xy * Math.cos(angle);
    data[i * 3 + 1] = r * z;
    data[i * 3 + 2] = r * xy * Math.sin(angle);
  }
  return data;
}
/** Concentrate background stars around the galactic plane, using a fixed seed. */
function makeGalacticStars(count: number, radius: number) {
  const data = new Float32Array(count * 3);
  let seed = 777001;
  const next = () => {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647;
  };
  for (let i = 0; i < count; i++) {
    const angle = next() * Math.PI * 2;
    const latitude = Math.max(
      -0.4,
      Math.min(0.4, Math.sqrt(-2 * Math.log(next())) * Math.cos(next() * Math.PI * 2) * 0.095),
    );
    const r = radius * (0.94 + next() * 0.12);
    data.set(
      [
        r * Math.cos(latitude) * Math.cos(angle),
        r * Math.sin(latitude),
        r * Math.cos(latitude) * Math.sin(angle),
      ],
      i * 3,
    );
  }
  return data;
}
/** Direction on the unit sphere → world position at radius. */
function skyPoint(x: number, y: number, z: number, radius: number): Vector3 {
  return new Vector3(x, y, z).normalize().multiplyScalar(radius);
}
/** Place decorative corner patterns clear of Earth in the opening camera view. */
function openingSkyPoint(x: number, y: number): [number, number, number] {
  const forward = new Vector3(...CAMERA).normalize().negate();
  const right = new Vector3().crossVectors(forward, new Vector3(0, 1, 0)).normalize();
  const up = new Vector3().crossVectors(right, forward);
  return forward
    .addScaledVector(right, x * 0.39)
    .addScaledVector(up, y * 0.345)
    .normalize()
    .toArray() as [number, number, number];
}
/** Stylized constellation graphs (decorative, not sky-accurate). */
const CONSTELLATIONS: { points: [number, number, number][]; links: [number, number][] }[] = [
  {
    points: [
      openingSkyPoint(-0.96, 0.6),
      openingSkyPoint(-0.89, 0.79),
      openingSkyPoint(-0.79, 0.67),
      openingSkyPoint(-0.69, 0.85),
      openingSkyPoint(-0.58, 0.75),
    ],
    links: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
    ],
  },
  {
    points: [
      openingSkyPoint(0.65, -0.88),
      openingSkyPoint(0.77, -0.73),
      openingSkyPoint(0.94, -0.78),
      openingSkyPoint(0.89, -0.91),
    ],
    links: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 0],
    ],
  },
  {
    // Cassiopeia-like W
    points: [
      [0.82, 0.42, -0.38],
      [0.7, 0.55, -0.45],
      [0.55, 0.4, -0.73],
      [0.38, 0.58, -0.72],
      [0.22, 0.45, -0.86],
    ],
    links: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
    ],
  },
  {
    // Orion-like
    points: [
      [-0.55, 0.62, -0.55],
      [-0.42, 0.35, -0.84],
      [-0.62, 0.28, -0.72],
      [-0.48, 0.05, -0.87],
      [-0.68, -0.05, -0.73],
      [-0.35, -0.22, -0.91],
      [-0.72, -0.28, -0.63],
    ],
    links: [
      [0, 1],
      [0, 2],
      [1, 3],
      [2, 4],
      [3, 4],
      [3, 5],
      [4, 6],
    ],
  },
  {
    // Dipper-like
    points: [
      [-0.88, 0.28, 0.38],
      [-0.78, 0.42, 0.46],
      [-0.65, 0.48, 0.59],
      [-0.52, 0.35, 0.78],
      [-0.48, 0.18, 0.86],
      [-0.62, 0.08, 0.78],
      [-0.74, 0.15, 0.65],
    ],
    links: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6],
      [6, 3],
    ],
  },
  {
    // Southern Cross-like
    points: [
      [0.35, -0.72, 0.6],
      [0.48, -0.55, 0.68],
      [0.22, -0.58, 0.78],
      [0.42, -0.42, 0.8],
      [0.28, -0.85, 0.45],
    ],
    links: [
      [0, 1],
      [0, 2],
      [0, 3],
      [0, 4],
    ],
  },
  {
    // Triangle / Lyra-like
    points: [
      [0.15, 0.78, 0.6],
      [0.32, 0.65, 0.68],
      [-0.02, 0.68, 0.73],
      [0.18, 0.55, 0.82],
    ],
    links: [
      [0, 1],
      [0, 2],
      [1, 3],
      [2, 3],
    ],
  },
];
function Constellations({ radius = 34 }: { radius?: number }) {
  const stars = useMemo(() => {
    const positions: number[] = [];
    for (const c of CONSTELLATIONS) {
      for (const p of c.points) {
        const v = skyPoint(p[0], p[1], p[2], radius);
        positions.push(v.x, v.y, v.z);
      }
    }
    return new Float32Array(positions);
  }, [radius]);
  const links = useMemo(() => {
    const positions: number[] = [];
    for (const graph of CONSTELLATIONS) {
      const points = graph.points.map((p) => skyPoint(...p, radius));
      for (const [a, b] of graph.links)
        positions.push(...points[a]!.toArray(), ...points[b]!.toArray());
    }
    return new Float32Array(positions);
  }, [radius]);
  return (
    <group>
      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[stars, 3]} />
        </bufferGeometry>
        <pointsMaterial
          color="#f4f8ff"
          size={0.11}
          transparent
          opacity={0.92}
          sizeAttenuation
          depthWrite={false}
        />
      </points>
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[links, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#9eb8d8" transparent opacity={0.22} depthWrite={false} />
      </lineSegments>
    </group>
  );
}
function GalaxyBand() {
  return (
    <mesh scale={48} rotation={[0.55, 0.35, -0.4]} name="galaxy-band">
      <sphereGeometry args={[1, 64, 48]} />
      <shaderMaterial
        transparent
        side={BackSide}
        depthWrite={false}
        blending={AdditiveBlending}
        vertexShader={`
          varying vec3 vDir;
          void main() {
            vDir = normalize(position);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `}
        fragmentShader={`
          varying vec3 vDir;
          float hash(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
          float noise(vec3 p) {
            vec3 i = floor(p), f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            return mix(mix(mix(hash(i), hash(i+vec3(1,0,0)), f.x),
                           mix(hash(i+vec3(0,1,0)), hash(i+vec3(1,1,0)), f.x), f.y),
                       mix(mix(hash(i+vec3(0,0,1)), hash(i+vec3(1,0,1)), f.x),
                           mix(hash(i+vec3(0,1,1)), hash(i+vec3(1,1,1)), f.x), f.y), f.z);
          }
          void main() {
            vec3 d = normalize(vDir);
            float band = exp(-pow(d.y * 4.0, 2.0));
            float mist = noise(d*14.0)*0.55 + noise(d*35.0)*0.30 + noise(d*80.0)*0.15;
            float lane = 0.3 + 0.7*smoothstep(0.01, 0.09, abs(d.y + 0.025*sin(d.x*12.0)));
            float core = exp(-pow(length(d-normalize(vec3(0.55,0.08,0.7)))*2.4,2.0));
            vec3 color = mix(vec3(0.18,0.30,0.52), vec3(0.46,0.33,0.28), core*0.55);
            // Background luminance stays far below Earth's surface and atmosphere.
            gl_FragColor = vec4(color, band*(0.075+mist*0.30)*lane + core*0.065);
          }
        `}
      />
    </mesh>
  );
}
function CosmicBackdrop({ moving }: { moving: boolean }) {
  const sky = useRef<Group>(null);
  const near = useMemo(() => makeStarShell(1100, 20, 8123), []);
  const mid = useMemo(() => makeStarShell(2000, 30, 13337), []);
  const far = useMemo(() => makeStarShell(2800, 44, 424242), []);
  const cluster = useMemo(() => makeGalacticStars(900, 38), []);
  useFrame((_, delta) => {
    if (moving && !document.hidden && sky.current)
      sky.current.rotation.y += Math.min(delta, 0.05) * 0.0045;
  });
  return (
    <group>
      <color attach="background" args={["#01030a"]} />
      <fog attach="fog" args={["#01030a", 32, 58]} />
      <GalaxyBand />
      <group ref={sky}>
        <points>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[far, 3]} />
          </bufferGeometry>
          <pointsMaterial
            color="#6f91b5"
            size={0.016}
            transparent
            opacity={0.48}
            sizeAttenuation
            depthWrite={false}
          />
        </points>
        <points>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[mid, 3]} />
          </bufferGeometry>
          <pointsMaterial
            color="#a9c4e4"
            size={0.026}
            transparent
            opacity={0.62}
            sizeAttenuation
            depthWrite={false}
          />
        </points>
        <points>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[near, 3]} />
          </bufferGeometry>
          <pointsMaterial
            color="#dce8f8"
            size={0.038}
            transparent
            opacity={0.72}
            sizeAttenuation
            depthWrite={false}
          />
        </points>
        <points rotation={[0.55, 0.35, -0.4]}>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[cluster, 3]} />
          </bufferGeometry>
          <pointsMaterial
            color="#c8daf2"
            size={0.055}
            transparent
            opacity={0.5}
            sizeAttenuation
            depthWrite={false}
          />
        </points>
        <Constellations />
      </group>
    </group>
  );
}
function OrbitalRing({ satellite, selected }: { satellite: DemoSatellite; selected: boolean }) {
  const points = useMemo(
    () =>
      Array.from(
        { length: 161 },
        (_, i) => new Vector3(...orbitPoint(satellite, (i * Math.PI * 2) / 160)),
      ),
    [satellite],
  );
  return (
    <Line
      points={points}
      color={satellite.color}
      lineWidth={selected ? 1 : 0.65}
      transparent
      opacity={selected ? 0.58 : 0.25}
      depthWrite={false}
    />
  );
}
function Spacecraft({
  satellite,
  clock,
  selected,
  onSelect,
  occluder,
}: {
  satellite: DemoSatellite;
  clock: SimulationClock;
  selected: boolean;
  onSelect: (id: string) => void;
  occluder: RefObject<Mesh>;
}) {
  const marker = useRef<Group>(null);
  useFrame(() => {
    if (marker.current) {
      marker.current.position.set(...satellitePosition(satellite, clock.seconds));
      marker.current.lookAt(0, 0, 0);
    }
  });
  return (
    <group ref={marker}>
      <mesh
        onClick={(e) => {
          e.stopPropagation();
          onSelect(satellite.id);
        }}
      >
        <sphereGeometry args={[selected ? 0.018 : 0.013, 12, 12]} />
        <meshBasicMaterial color={satellite.color} />
      </mesh>
      <mesh rotation={[0, 0, Math.PI / 4]}>
        <boxGeometry args={[0.018, 0.028, 0.018]} />
        <meshBasicMaterial color="#e8f2ff" />
      </mesh>
      <mesh position={[0.036, 0, 0]}>
        <boxGeometry args={[0.043, 0.017, 0.004]} />
        <meshBasicMaterial color="#248bff" />
      </mesh>
      <mesh position={[-0.036, 0, 0]}>
        <boxGeometry args={[0.043, 0.017, 0.004]} />
        <meshBasicMaterial color="#248bff" />
      </mesh>
      <Html center position={[0, 0.07, 0]} zIndexRange={[8, 0]} occlude={[occluder]}>
        <button
          type="button"
          className={`satellite-label ${selected ? "selected" : ""}`}
          onClick={() => onSelect(satellite.id)}
          aria-label={`Select ${satellite.id}, simulated satellite`}
        >
          <span style={{ background: satellite.color }} />
          {satellite.id}
          <small>DEMO</small>
        </button>
      </Html>
    </group>
  );
}
function CommunicationLink({
  clock,
  satellite,
}: {
  clock: SimulationClock;
  satellite: DemoSatellite;
}) {
  const geometry = useMemo(
    () =>
      new BufferGeometry().setAttribute(
        "position",
        new Float32BufferAttribute(new Float32Array(33 * 3), 3),
      ),
    [],
  );
  const line = useMemo(
    () =>
      new ThreeLine(
        geometry,
        new LineBasicMaterial({ color: "#00d9ff", transparent: true, opacity: 0.65 }),
      ),
    [geometry],
  );
  useEffect(
    () => () => {
      geometry.dispose();
      line.material.dispose();
    },
    [geometry, line],
  );
  const beam = useRef<Group>(null),
    pulse = useRef<Mesh>(null),
    station = useRef<Mesh>(null);
  const vectors = useMemo(
    () => ({ start: new Vector3(), end: new Vector3(), mid: new Vector3(), p: new Vector3() }),
    [],
  );
  useFrame(() => {
    const { start, end, mid, p } = vectors;
    start.set(...groundPosition(clock.seconds));
    end.set(...satellitePosition(satellite, clock.seconds));
    if (station.current) station.current.position.copy(start);
    if (!beam.current) return;
    beam.current.visible = elevationDeg(satellite, clock.seconds) > 10;
    if (!beam.current.visible) return;
    mid
      .copy(start)
      .add(end)
      .multiplyScalar(0.5)
      .normalize()
      .multiplyScalar((start.length() + end.length()) * 0.52);
    const attribute = geometry.getAttribute("position");
    const at = (t: number) =>
      p
        .copy(start)
        .multiplyScalar((1 - t) ** 2)
        .addScaledVector(mid, 2 * (1 - t) * t)
        .addScaledVector(end, t * t);
    for (let i = 0; i < 33; i++) {
      at(i / 32);
      attribute.setXYZ(i, p.x, p.y, p.z);
    }
    attribute.needsUpdate = true;
    geometry.computeBoundingSphere();
    pulse.current?.position.copy(at((clock.visibleSeconds * 0.3) % 1));
  });
  return (
    <>
      <mesh ref={station}>
        <sphereGeometry args={[0.009, 12, 12]} />
        <meshBasicMaterial color="#00e0a4" />
      </mesh>
      <group ref={beam}>
        <primitive object={line} />
        <mesh ref={pulse}>
          <sphereGeometry args={[0.008, 10, 10]} />
          <meshBasicMaterial color="#e8faff" />
        </mesh>
      </group>
    </>
  );
}
function OrbitalWorld({
  clock,
  selected,
  onSelect,
  autoRotate,
  paused,
  diagnostics,
}: {
  clock: SimulationClock;
  selected: DemoSatellite;
  onSelect: (id: string) => void;
  autoRotate: boolean;
  paused: RefObject<number>;
  diagnostics: RefObject<HTMLDivElement | null>;
}) {
  const occluder = useRef<Mesh>(null!);
  const world = useRef<Group>(null),
    earth = useRef<Group>(null),
    viewAngle = useRef(0),
    lastReport = useRef(0);
  const sun = useMemo(() => SUN.clone(), []);
  useEffect(() => {
    // Pausing switches to demand rendering; flush the final angle immediately
    // rather than waiting for a telemetry frame that may never be requested.
    if (diagnostics.current) diagnostics.current.dataset["rotation"] = String(viewAngle.current);
  }, [autoRotate, diagnostics]);
  const skip = useRef(false);
  useEffect(() => {
    const change = () => {
      skip.current = true;
    };
    document.addEventListener("visibilitychange", change);
    return () => document.removeEventListener("visibilitychange", change);
  }, []);
  useFrame((_, delta) => {
    if (document.hidden || skip.current) {
      skip.current = false;
      return;
    }
    if (autoRotate && performance.now() > paused.current)
      viewAngle.current = advanceRotation(viewAngle.current, delta);
    if (world.current) world.current.rotation.y = viewAngle.current;
    if (earth.current) earth.current.rotation.y = earthAngle(clock.seconds);
    sun.copy(SUN).applyAxisAngle(AXIS, viewAngle.current);
    if (performance.now() - lastReport.current > 500 && diagnostics.current) {
      diagnostics.current.dataset["rotation"] = String(viewAngle.current);
      diagnostics.current.dataset["frames"] = "active";
      lastReport.current = performance.now();
    }
  });
  return (
    <>
      <ambientLight intensity={0.14} />
      <CosmicBackdrop moving={autoRotate} />
      <group ref={world}>
        <directionalLight position={[-3, 3, -5]} intensity={3.2} />
        {/* Cool fill from camera side — lifts Earth without washing the sky */}
        <directionalLight position={[2.2, 0.6, -2.4]} intensity={0.7} color="#9ec0e8" />
        <group ref={earth}>
          <EarthSurface sun={sun} occluder={occluder} />
        </group>
        {DEMO_SATELLITES.map((s) => (
          <OrbitalRing key={s.id} satellite={s} selected={selected.id === s.id} />
        ))}
        {DEMO_SATELLITES.map((s) => (
          <Spacecraft
            key={s.id}
            satellite={s}
            selected={selected.id === s.id}
            clock={clock}
            onSelect={onSelect}
            occluder={occluder}
          />
        ))}
        <CommunicationLink clock={clock} satellite={selected} />
      </group>
    </>
  );
}
function ViewControls({
  commands,
  clock,
  paused,
  diagnostics,
}: {
  commands: RefObject<ControlsApi | null>;
  clock: SimulationClock;
  paused: RefObject<number>;
  diagnostics: RefObject<HTMLDivElement | null>;
}) {
  const ref = useRef<ComponentRef<typeof OrbitControls>>(null);
  const { camera, invalidate, size } = useThree();
  const fitScale = Math.min(1.55, Math.max(1, (0.96 * size.height) / size.width));
  useEffect(() => {
    const api: ControlsApi = {
      zoom(factor) {
        if (!ref.current) return;
        camera.position
          .sub(ref.current.target)
          .multiplyScalar(factor)
          .clampLength(2.25, 6.5)
          .add(ref.current.target);
        ref.current.update();
        invalidate();
      },
      reset() {
        camera.position.set(...CAMERA).multiplyScalar(fitScale);
        ref.current?.target.set(0, 0, 0);
        ref.current?.update();
        paused.current = 0;
        clock.pauseUntil = 0;
        invalidate();
      },
    };
    commands.current = api;
    api.reset();
    const reset = () => api.reset();
    window.addEventListener("satmod:global-view", reset);
    return () => {
      commands.current = null;
      window.removeEventListener("satmod:global-view", reset);
    };
  }, [camera, commands, invalidate, paused, clock, fitScale]);
  useFrame(() => {
    if (diagnostics.current) {
      diagnostics.current.dataset["cameraDistance"] = String(camera.position.length());
      diagnostics.current.dataset["cameraAzimuth"] = String(Math.atan2(camera.position.x, camera.position.z));
    }
  });
  return (
    <OrbitControls
      ref={ref}
      makeDefault
      enablePan={false}
      enableDamping
      dampingFactor={0.08}
      minDistance={2.25}
      maxDistance={6.5}
      rotateSpeed={0.6}
      zoomSpeed={0.6}
      onStart={() => {
        paused.current = Infinity;
        clock.pauseUntil = Infinity;
      }}
      onEnd={() => {
        paused.current = performance.now() + 1400;
        clock.pauseUntil = paused.current;
      }}
    />
  );
}
function ContextGuard({ onLost }: { onLost: () => void }) {
  const { gl } = useThree();
  useEffect(() => {
    const lost = (e: Event) => {
      e.preventDefault();
      onLost();
    };
    gl.domElement.addEventListener("webglcontextlost", lost);
    return () => gl.domElement.removeEventListener("webglcontextlost", lost);
  }, [gl, onLost]);
  return null;
}
function StaticFallback({ loading = false }: { loading?: boolean }) {
  return (
    <div className="globe-fallback" role="status">
      <img src="/textures/earth-day.jpg" alt="Static equirectangular Earth map" />
      <strong>{loading ? "Preparing orbital viewport…" : "3D view unavailable"}</strong>
      <span>
        {loading
          ? "Loading local Earth textures"
          : "Static geographic fallback · WebGL is unavailable on this device."}
      </span>
    </div>
  );
}
class GlobeBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  override state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  override render() {
    return this.state.failed ? <StaticFallback /> : this.props.children;
  }
}
export default function EarthGlobe({
  clock,
  selected,
  onSelect,
  autoRotate,
  onAutoRotate,
  motionEnabled,
}: GlobeProps) {
  const commands = useRef<ControlsApi | null>(null),
    paused = useRef(0),
    viewport = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(() => !document.hidden),
    [lost, setLost] = useState(false),
    [ready, setReady] = useState(false);
  const onLost = useCallback(() => setLost(true), []);
  useEffect(() => {
    const change = () => setVisible(!document.hidden);
    document.addEventListener("visibilitychange", change);
    return () => document.removeEventListener("visibilitychange", change);
  }, []);
  return (
    <div
      className="earth-viewport"
      ref={viewport}
      data-testid="earth-viewport"
      data-auto-rotate={autoRotate}
      aria-label="Interactive 3D Earth and simulated satellite orbits"
    >
      <div className="globe-caption">
        <span className="panel-eyebrow">ORBITAL VIEW</span>
        <h2>Earth observation</h2>
        <span className="simulation-tag">SIMULATION · 30×</span>
      </div>
      <GlobeBoundary>
        {lost ? (
          <StaticFallback />
        ) : (
          <Suspense fallback={<StaticFallback loading />}>
            <Canvas
              dpr={[1, 1.5]}
              camera={{ position: CAMERA, fov: 38, near: 0.1, far: 70 }}
              gl={{ alpha: false, antialias: true, powerPreference: "high-performance" }}
              frameloop={!visible ? "never" : motionEnabled ? "always" : "demand"}
              fallback={<StaticFallback />}
              onCreated={({ gl }) => {
                gl.setClearColor("#01030a", 1);
                setReady(true);
              }}
            >
              <OrbitalWorld
                clock={clock}
                selected={selected}
                onSelect={onSelect}
                autoRotate={autoRotate}
                paused={paused}
                diagnostics={viewport}
              />
              <ViewControls
                commands={commands}
                clock={clock}
                paused={paused}
                diagnostics={viewport}
              />
              <ContextGuard onLost={onLost} />
            </Canvas>
          </Suspense>
        )}
      </GlobeBoundary>
      <div className="earth-focus-vignette" aria-hidden />
      <div className="globe-controls" aria-label="Globe view controls">
        <button
          disabled={!ready || lost}
          onClick={() => commands.current?.zoom(0.85)}
          aria-label="Zoom in"
          title="Zoom in"
        >
          <Plus size={15} />
        </button>
        <button
          disabled={!ready || lost}
          onClick={() => commands.current?.zoom(1 / 0.85)}
          aria-label="Zoom out"
          title="Zoom out"
        >
          <Minus size={15} />
        </button>
        <button
          onClick={onAutoRotate}
          aria-label="Toggle Earth auto rotation"
          aria-pressed={autoRotate}
          title={autoRotate ? "Pause Earth turntable" : "Enable Earth turntable"}
        >
          {autoRotate ? <Pause size={14} /> : <Play size={14} />}
        </button>
        <button
          disabled={!ready || lost}
          onClick={() => commands.current?.reset()}
          aria-label="Reset globe camera"
          title="Reset camera"
        >
          <RotateCcw size={14} />
        </button>
        <button
          disabled={!document.fullscreenEnabled}
          onClick={() => {
            if (!document.fullscreenElement)
              void viewport.current?.requestFullscreen?.().catch(() => {});
            else void document.exitFullscreen();
          }}
          aria-label="Toggle globe fullscreen"
          title={
            document.fullscreenEnabled ? "Fullscreen" : "Fullscreen is unavailable on this browser"
          }
        >
          <Maximize2 size={14} />
        </button>
      </div>
      <div className="globe-readout">
        <span>
          <Globe2 size={12} /> {autoRotate ? "AUTO ROTATION" : "MANUAL VIEW"}{" "}
          <small>{autoRotate ? "108 s / revolution" : "drag to explore"}</small>
        </span>
        <span className="selected-orbit-dot" style={{ color: selected.color }}>
          ● {selected.id}
          <small>DEMO ORBIT</small>
        </span>
      </div>
      <div className="globe-help">
        Drag to rotate <span>·</span> Scroll to zoom <span>·</span> Select a satellite
      </div>
    </div>
  );
}
