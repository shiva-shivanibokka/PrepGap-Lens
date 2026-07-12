import type { ScoreReport } from "@/lib/types";
import { Meter, WeightDots, band } from "./Meter";

const pct = (v: number) => `${Math.round(v * 100)}`;

export function ResultsView({ report }: { report: ScoreReport }) {
  const overall = report.readinessOverall;
  const categories = Object.entries(report.readinessByCategory).sort((a, b) => a[1] - b[1]);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Your readiness</h2>
        <span className="chip">gap vs. this job</span>
      </div>

      <div className="readout">
        <div className="lbl">Overall readiness</div>
        <div className={`big ${band(overall).text}`}>{pct(overall)}%</div>
      </div>

      <div className="bars">
        {categories.map(([name, v]) => (
          <div key={name} className="bar-row">
            <span className="name">{name}</span>
            <span className={`val ${band(v).text}`}>{pct(v)}%</span>
            <Meter value={v} />
          </div>
        ))}
      </div>

      <p className="section-label" style={{ marginTop: "2rem" }}>
        Biggest gaps to close
      </p>
      <div className="gaps">
        {report.topGaps.map((g) => (
          <div key={g.skillId} className="gap">
            <span className="gname">{g.skillName}</span>
            <span className="gmeta">
              <WeightDots weight={g.weight} />
              <span className={`val ${band(g.coverage).text}`}>{pct(g.coverage)}%</span>
            </span>
            <span className="gcat">{g.category}</span>
            {g.prereqsMissing.length > 0 && <span className="gprereq">needs first: {g.prereqsMissing.join(", ")}</span>}
            <Meter value={g.coverage} />
          </div>
        ))}
      </div>
    </section>
  );
}
