import type { Skill, Coverage, Gap, ScoreReport } from "./types";

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

export function scoreReadiness(
  skills: Skill[],
  coverage: Coverage,
  opts: { topKGaps?: number; prereqPenalty?: number } = {},
): ScoreReport {
  const topKGaps = opts.topKGaps ?? 6;
  const prereqPenalty = opts.prereqPenalty ?? 0.1;
  const cov = (id: string) => clamp01(coverage[id] ?? 0);

  let totalW = 0;
  let total = 0;
  for (const s of skills) {
    totalW += s.weight;
    total += s.weight * cov(s.id);
  }
  const readinessOverall = totalW > 0 ? total / totalW : 0;

  const readinessByCategory: Record<string, number> = {};
  const categories = [...new Set(skills.map((s) => s.category))];
  for (const c of categories) {
    let cw = 0;
    let ct = 0;
    for (const s of skills.filter((s) => s.category === c)) {
      cw += s.weight;
      ct += s.weight * cov(s.id);
    }
    readinessByCategory[c] = cw > 0 ? ct / cw : 0;
  }

  const gaps: Gap[] = skills.map((s) => {
    let gap = 1 - cov(s.id);
    const prereqsMissing: string[] = [];
    let factor = 0;
    for (const p of s.prereqs) {
      const pc = cov(p);
      if (pc < 0.55) {
        prereqsMissing.push(p);
        factor += (0.55 - pc) / 0.55;
      }
    }
    if (prereqsMissing.length) gap = Math.min(1, gap + prereqPenalty * Math.min(1, factor));
    return {
      skillId: s.id,
      skillName: s.name,
      category: s.category,
      weight: s.weight,
      coverage: cov(s.id),
      gap: clamp01(gap),
      prereqsMissing,
    };
  });

  gaps.sort((a, b) => {
    const ia = a.gap * a.weight;
    const ib = b.gap * b.weight;
    return (
      ib - ia ||
      b.weight - a.weight ||
      b.gap - a.gap ||
      a.skillName.toLowerCase().localeCompare(b.skillName.toLowerCase())
    );
  });

  const coverageBySkill: Coverage = {};
  for (const s of skills) coverageBySkill[s.id] = cov(s.id);

  return {
    readinessOverall: clamp01(readinessOverall),
    readinessByCategory,
    coverageBySkill,
    topGaps: gaps.slice(0, Math.max(0, topKGaps)),
  };
}
