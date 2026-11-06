import { useEffect, useRef, useState } from "react";
import { SIMULATION_RATE, type SimulationClock } from "@/lib/orbital-simulation";
export function useReducedMotion() {
  const [reduced, setReduced] = useState(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const change = () => setReduced(query.matches);
    query.addEventListener("change", change);
    change();
    return () => query.removeEventListener("change", change);
  }, []);
  return reduced;
}
export function useMissionClock(enabled: boolean) {
  const clock = useRef<SimulationClock>({ seconds: 0, visibleSeconds: 0, enabled, pauseUntil: 0 });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    clock.current.enabled = enabled;
  }, [enabled]);
  useEffect(() => {
    let frame = 0,
      last = performance.now(),
      lastPaint = last;
    const visibility = () => {
      last = performance.now();
    };
    const animate = (now: number) => {
      const delta = (now - last) / 1000;
      last = now;
      if (!document.hidden && clock.current.enabled && now >= clock.current.pauseUntil) {
        clock.current.seconds += delta * SIMULATION_RATE;
        clock.current.visibleSeconds += delta;
      }
      if (now - lastPaint >= 1000) {
        setTick((n) => n + 1);
        lastPaint = now;
      }
      frame = requestAnimationFrame(animate);
    };
    document.addEventListener("visibilitychange", visibility);
    frame = requestAnimationFrame(animate);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener("visibilitychange", visibility);
    };
  }, []);
  return { clock: clock.current, tick };
}
