import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import {
  satmodApi,
  type Capture,
  type ClassificationResult,
  type SignalGenerationRequest,
  type ClassifyRequest,
} from "@/lib/satmod-api";
export const useHealth = () =>
  useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => satmodApi.health(signal),
    refetchInterval: 10000,
    retry: 1,
  });
export const useInfo = () =>
  useQuery({ queryKey: ["info"], queryFn: ({ signal }) => satmodApi.info(signal), retry: 1 });
export const useEvaluation = () =>
  useQuery({
    queryKey: ["evaluation"],
    queryFn: ({ signal }) => satmodApi.evaluation(signal),
    retry: 1,
  });
export const useModels = () =>
  useQuery({ queryKey: ["models"], queryFn: ({ signal }) => satmodApi.models(signal), retry: 1 });
export const useSamples = () =>
  useQuery({ queryKey: ["samples"], queryFn: ({ signal }) => satmodApi.samples(signal), retry: 1 });
export const useClassification = () =>
  useMutation({
    mutationKey: ["classification"],
    mutationFn: (body: ClassifyRequest) => satmodApi.classify(body),
  });
export const useGeneration = () =>
  useMutation({ mutationFn: (body: SignalGenerationRequest) => satmodApi.generate(body) });
export const useAnalysis = () =>
  useMutation({ mutationFn: (capture: Capture) => satmodApi.analyze(capture) });
export const useBenchmark = () =>
  useMutation({
    mutationFn: (v: { modelId: string; sampleId: number; iterations: number }) =>
      satmodApi.benchmark(v.modelId, v.sampleId, v.iterations),
  });
export interface Preferences {
  theme: "dark" | "light";
  charts: "interactive" | "simple";
  defaultSnr: number;
  defaultSps: number;
  defaultModulation: "BPSK" | "QPSK" | "16-QAM";
}
const defaults: Preferences = {
  theme: "dark",
  charts: "interactive",
  defaultSnr: 12,
  defaultSps: 8,
  defaultModulation: "BPSK",
};
const Context = createContext<{
  classification: ClassificationResult | null;
  setClassification: (result: ClassificationResult) => void;
  capture: Capture | null;
  setCapture: (c: Capture) => void;
  preferences: Preferences;
  setPreferences: (p: Preferences) => void;
}>({
  classification: null,
  setClassification: () => {},
  capture: null,
  setCapture: () => {},
  preferences: defaults,
  setPreferences: () => {},
});
export function SignalProvider({ children }: { children: ReactNode }) {
  const [classification, setClassification] = useState<ClassificationResult | null>(null);
  const [capture, setCapture] = useState<Capture | null>(null);
  const [preferences, setPreferences] = useState<Preferences>(() => {
    try {
      const value = JSON.parse(
        localStorage.getItem("satmod-preferences") ?? "null",
      ) as Partial<Preferences> | null;
      return value ? { ...defaults, ...value } : defaults;
    } catch {
      return defaults;
    }
  });
  useEffect(() => {
    document.documentElement.dataset["theme"] = preferences.theme;
  }, [preferences.theme]);
  const save = (p: Preferences) => {
    setPreferences(p);
    localStorage.setItem("satmod-preferences", JSON.stringify(p));
  };
  return (
    <Context.Provider
      value={{
        capture,
        setCapture,
        classification,
        setClassification,
        preferences,
        setPreferences: save,
      }}
    >
      {children}
    </Context.Provider>
  );
}
export const useSignal = () => useContext(Context);
