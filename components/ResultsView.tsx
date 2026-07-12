import type { ScoreReport, Skill } from "@/lib/types";
import { Meter, WeightDots, band } from "./Meter";
import { Tip } from "./Tip";

const pct = (v: number) => `${Math.round(v * 100)}`;

export function ResultsView({ report, skills }: { report: ScoreReport; skills: Skill[] }) {
  const overall = report.readinessOverall;
  const categories = Object.entries(report.readinessByCategory).sort((a, b) => a[1] - b[1]);
  const nameById = new Map(skills.map((s) => [s.id, s.name] as const));

  return (
    <section className="panel">
      <div className="panel-head">
        <div className="head-tip">
          <h2>Your readiness</h2>
          <Tip text="How well your resume already covers what this job asks for — overall and by category. Higher is better." />
        </div>
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
        <Tip text="Skills ranked by impact = how much you're missing × how important it is to this job. The dots show importance (1–5); the bar shows your current coverage." />
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
            {g.prereqsMissing.length > 0 && (
              <span className="gprereq">needs first: {g.prereqsMissing.map((id) => nameById.get(id) ?? id).join(", ")}</span>
            )}
            <Meter value={g.coverage} />
          </div>
        ))}
      </div>
    </section>
  );
}
