// The only color in the app lives here: it encodes a measured value (readiness / coverage).
export function band(v: number): { fill: string; text: string } {
  if (v < 0.4) return { fill: "bg-rose-500", text: "text-rose-600 dark:text-rose-400" };
  if (v < 0.7) return { fill: "bg-amber-500", text: "text-amber-600 dark:text-amber-400" };
  return { fill: "bg-emerald-500", text: "text-emerald-600 dark:text-emerald-400" };
}

export function Meter({ value, className = "" }: { value: number; className?: string }) {
  const v = Math.max(0, Math.min(1, value));
  return (
    <div
      className={`h-1.5 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800 ${className}`}
      role="meter"
      aria-valuenow={Math.round(v * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div className={`h-full rounded-full transition-[width] duration-700 ${band(v).fill}`} style={{ width: `${v * 100}%` }} />
    </div>
  );
}

// Weight (1..5) as filled dots — importance is a small discrete scale, so dots, not a bar.
export function WeightDots({ weight }: { weight: number }) {
  return (
    <span className="inline-flex items-center gap-0.5" aria-label={`weight ${weight} of 5`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <span
          key={i}
          className={`h-1.5 w-1.5 rounded-full ${i <= weight ? "bg-zinc-900 dark:bg-zinc-100" : "bg-zinc-300 dark:bg-zinc-700"}`}
        />
      ))}
    </span>
  );
}
