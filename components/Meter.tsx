// The only color that encodes data: readiness/coverage band -> fill + text class.
export function band(v: number): { fill: string; text: string } {
  if (v < 0.4) return { fill: "fill-low", text: "t-low" };
  if (v < 0.7) return { fill: "fill-mid", text: "t-mid" };
  return { fill: "fill-high", text: "t-high" };
}

export function Meter({ value }: { value: number }) {
  const v = Math.max(0, Math.min(1, value));
  return (
    <div className="probbar" role="meter" aria-valuenow={Math.round(v * 100)} aria-valuemin={0} aria-valuemax={100}>
      <div className={band(v).fill} style={{ width: `${v * 100}%` }} />
    </div>
  );
}

// Weight (1..5) as filled dots — importance is a small discrete scale.
export function WeightDots({ weight }: { weight: number }) {
  return (
    <span className="dots" aria-label={`weight ${weight} of 5`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <span key={i} className={`d${i <= weight ? " on" : ""}`} />
      ))}
    </span>
  );
}
