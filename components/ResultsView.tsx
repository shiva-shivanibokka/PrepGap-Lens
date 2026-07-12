import type { ScoreReport } from "@/lib/types";
import { Meter, WeightDots, band } from "./Meter";

const pct = (v: number) => `${Math.round(v * 100)}`;

export function ResultsView({ report }: { report: ScoreReport }) {
  const overall = report.readinessOverall;
  const categories = Object.entries(report.readinessByCategory).sort((a, b) => a[1] - b[1]);

  return (
    <section className="mt-10">
      {/* Signature readout: the readiness gauge */}
      <div className="rounded-2xl border border-zinc-200 p-6 dark:border-zinc-800">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-zinc-500">Readiness</p>
        <div className="mt-2 flex items-end gap-2">
          <span className={`font-mono text-6xl font-semibold leading-none tabular-nums ${band(overall).text}`}>
            {pct(overall)}
          </span>
          <span className="mb-1 font-mono text-lg text-zinc-400">%</span>
        </div>
        <Meter value={overall} className="mt-4 h-2" />

        <div className="mt-6 grid gap-3">
          {categories.map(([name, v]) => (
            <div key={name} className="grid grid-cols-[1fr_auto] items-center gap-x-4 gap-y-1">
              <span className="text-sm text-zinc-700 dark:text-zinc-300">{name}</span>
              <span className={`font-mono text-xs tabular-nums ${band(v).text}`}>{pct(v)}%</span>
              <div className="col-span-2">
                <Meter value={v} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Gaps: what to close first */}
      <div className="mt-8">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-zinc-500">Biggest gaps to close</p>
        <ul className="mt-3 divide-y divide-zinc-200 dark:divide-zinc-800">
          {report.topGaps.map((g) => (
            <li key={g.skillId} className="grid grid-cols-[1fr_auto] items-center gap-x-4 gap-y-1.5 py-3">
              <span className="font-medium">{g.skillName}</span>
              <div className="flex items-center gap-3">
                <WeightDots weight={g.weight} />
                <span className={`font-mono text-xs tabular-nums ${band(g.coverage).text}`}>{pct(g.coverage)}%</span>
              </div>
              <span className="font-mono text-xs uppercase tracking-wider text-zinc-400">{g.category}</span>
              {g.prereqsMissing.length > 0 && (
                <span className="col-start-1 font-mono text-xs text-amber-600 dark:text-amber-400">
                  needs first: {g.prereqsMissing.join(", ")}
                </span>
              )}
              <div className="col-span-2">
                <Meter value={g.coverage} />
              </div>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
