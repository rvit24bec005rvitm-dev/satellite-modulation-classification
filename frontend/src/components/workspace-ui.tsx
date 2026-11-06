import type React from "react";
import { Button } from "@/components/ui/button";
const cn = (...values: Array<string | false | null | undefined>) =>
  values.filter(Boolean).join(" ");
const PAGE_META: Record<string, { title: string; eyebrow: string; description: string }> = {
  overview: {
    title: "Satellite Signal Intelligence",
    eyebrow: "MISSION OVERVIEW",
    description: "Automatic modulation classification and signal analysis",
  },
  classifier: {
    title: "Live Classifier",
    eyebrow: "SIGNAL WORKSTATION",
    description: "Inspect an I/Q capture and send it to the connected classifier.",
  },
  studio: {
    title: "Satellite Studio",
    eyebrow: "RF SIMULATION LAB",
    description: "Configure a signal generation request and inspect returned samples.",
  },
  evaluation: {
    title: "Model Performance & Evaluation",
    eyebrow: "MODEL EVALUATION",
    description: "Measured evaluation artifacts, reported metrics, and validation coverage.",
  },
  arena: {
    title: "Model Arena",
    eyebrow: "ARCHITECTURE COMPARISON",
    description: "Compare trained model availability without inventing benchmark results.",
  },
  intelligence: {
    title: "Signal Intelligence",
    eyebrow: "SIGNAL DIAGNOSTICS",
    description: "Quality metrics and signal comparisons from an analyzed capture.",
  },
  theory: {
    title: "Satellite Theory",
    eyebrow: "TECHNICAL REFERENCE",
    description: "A concise reference for digital modulation and satellite links.",
  },
  settings: {
    title: "System Settings",
    eyebrow: "PLATFORM CONFIGURATION",
    description: "Inspect service connectivity and model integration status.",
  },
};

export function Eyebrow({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-3 flex items-center gap-2 font-mono text-[10px] tracking-[0.15em] text-primary">
      {children}
    </div>
  );
}

export function SectionHeading({ title, note }: { title: string; note?: string }) {
  return (
    <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
      <h2 className="font-display text-[15px] font-semibold">{title}</h2>
      {note && (
        <span className="max-w-full break-words font-mono text-[9px] text-muted-foreground">
          {note}
        </span>
      )}
    </div>
  );
}

export function Surface({ children, className, ...props }: React.HTMLAttributes<HTMLElement>) {
  return (
    <section className={cn("min-w-0 border border-border/80 bg-card/70", className)} {...props}>
      {children}
    </section>
  );
}

export function PageHeader({ page }: { page: string }) {
  const meta = PAGE_META[page] ?? PAGE_META["overview"]!;
  return (
    <div className="mb-7">
      <Eyebrow>
        <span className="size-1.5 rounded-full bg-primary" />
        {meta.eyebrow}
      </Eyebrow>
      <h1 className="font-display text-2xl font-semibold leading-tight sm:text-[30px]">
        {meta.title}
      </h1>
      <p className="mt-2 max-w-2xl text-[13px] text-muted-foreground">{meta.description}</p>
    </div>
  );
}

export function Message({
  error,
  loading,
}: {
  error?: Error | null | undefined;
  loading?: boolean | undefined;
}) {
  if (error)
    return (
      <div
        role="alert"
        className="my-4 border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive"
      >
        {error.message}
      </div>
    );
  if (loading)
    return (
      <div role="status" className="my-4 animate-pulse p-3 text-sm text-muted-foreground">
        Connecting to signal service…
      </div>
    );
  return null;
}
export function Empty({ text = "Select or generate a capture to begin." }: { text?: string }) {
  return (
    <div className="flex min-h-48 items-center justify-center border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
      {text}
    </div>
  );
}
export function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="mb-4 block min-w-0 text-xs text-muted-foreground">
      <span className="mb-2 block">{label}</span>
      {children}
    </label>
  );
}
export function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <Field label={label}>
      <input
        type="number"
        className="control"
        value={Number.isFinite(value) ? value : ""}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(e.target.valueAsNumber)}
      />
    </Field>
  );
}
export function Stats({ items }: { items: Array<[string, string | number]> }) {
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
      {items.map(([label, value]) => (
        <div key={label} className="min-w-0 border-b border-border py-4">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
          <div className="mt-2 break-words font-mono text-lg text-primary">{value}</div>
        </div>
      ))}
    </div>
  );
}
export const percent = (value: number | undefined) =>
  value === undefined ? "—" : `${(value * 100).toFixed(2)}%`;
export const numeric = (value: number | null | undefined, digits = 3) =>
  value == null ? "—" : value.toLocaleString(undefined, { maximumSignificantDigits: digits });
export function ExportButton({
  onClick,
  disabled = false,
  children = "Export results",
}: {
  onClick: () => void;
  disabled?: boolean;
  children?: React.ReactNode;
}) {
  return (
    <Button variant="outline" disabled={disabled} onClick={onClick}>
      {children}
    </Button>
  );
}
