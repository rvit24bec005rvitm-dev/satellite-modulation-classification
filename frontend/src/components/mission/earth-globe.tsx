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
          emissiveIntensity={1.35}
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
      <mesh scale={1.025} name="atmosphere">
        <sphereGeometry args={[1, 64, 48]} />
        <shaderMaterial
          transparent
          side={BackSide}
          depthWrite={false}
          blending={AdditiveBlending}
          vertexShader={
            "varying vec3 n; varying vec3 v; void main(){ vec4 p=modelViewMatrix*vec4(position,1.0); n=normalize(normalMatrix*normal); v=normalize(-p.xyz); gl_Position=projectionMatrix*p; }"
          }
          fragmentShader={
            "varying vec3 n; varying vec3 v; void main(){ float rim=pow(1.0-abs(dot(normalize(n),normalize(v))),3.0); gl_FragColor=vec4(0.12,0.50,1.0,rim*0.55); }"
          }
        />
      </mesh>
    </>
  );
}
function Starfield() {
  const positions = useMemo(() => {
    const data = new Float32Array(600 * 3);
    let seed = 8123;
    for (let i = 0; i < 600; i++) {
      seed = (seed * 16807) % 2147483647;
      const z = 2 * (seed / 2147483647) - 1;
      const angle = i * Math.PI * (3 - Math.sqrt(5));
      const r = 25;
      data[i * 3] = r * Math.sqrt(1 - z * z) * Math.cos(angle);
      data[i * 3 + 1] = r * z;
      data[i * 3 + 2] = r * Math.sqrt(1 - z * z) * Math.sin(angle);
    }
    return data;
  }, []);
  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        color="#aacde9"
        size={0.035}
        transparent
        opacity={0.5}
        sizeAttenuation
        depthWrite={false}
      />
    </points>
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
      <ambientLight intensity={0.17} />
      <Starfield />
      <group ref={world}>
        <directionalLight position={[-3, 3, -5]} intensity={2.7} />
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
    if (diagnostics.current)
      diagnostics.current.dataset["cameraDistance"] = String(camera.position.length());
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
              gl={{ alpha: true, antialias: true, powerPreference: "high-performance" }}
              frameloop={!visible ? "never" : motionEnabled ? "always" : "demand"}
              fallback={<StaticFallback />}
              onCreated={() => setReady(true)}
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
