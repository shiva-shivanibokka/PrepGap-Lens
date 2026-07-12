import type { Skill, Resource, ScoreReport, Gap } from "./types";

export type TaskType = "learn" | "practice" | "explain" | "review";

export interface PlanItem {
  day: number;
  order: number;
  skillId: string;
  skillName: string;
  category: string;
  taskType: TaskType;
  minutes: number;
  resourceTitle: string;
  resourceUrl: string;
  resourceType: string;
  credibility: string;
  successCheck: string;
}

export interface DayPlan {
  day: number;
  totalMinutes: number;
  items: PlanItem[];
  notes: string[];
}

export interface DayFeedback {
  status: "done" | "skipped";
  difficulty: "easy" | "ok" | "hard";
  confidence?: number; // 1..5
}

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
const dedupe = (xs: string[]) => [...new Set(xs)];

// Seedable PRNG so a given (seed, day) always yields the same plan (JS Math.random isn't seedable).
function mulberry32(seed: number) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function initMasteryFromReport(report: ScoreReport): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [id, c] of Object.entries(report.coverageBySkill)) out[id] = clamp01(c);
  return out;
}

export function applyDayFeedback(
  mastery: Record<string, number>,
  feedback: Record<string, DayFeedback>,
): { newMastery: Record<string, number>; reinforce: string[]; retry: string[] } {
  const newMastery = { ...mastery };
  const reinforce: string[] = [];
  const retry: string[] = [];

  for (const [id, fb] of Object.entries(feedback)) {
    let m = clamp01(newMastery[id] ?? 0.05);
    if (fb.status === "skipped") {
      retry.push(id);
      continue;
    }
    if (fb.difficulty === "easy") m = Math.min(1, m + 0.12);
    else if (fb.difficulty === "ok") m = Math.min(1, m + 0.08);
    else {
      m = Math.min(1, m + 0.04);
      reinforce.push(id);
    }
    const conf = fb.confidence ?? 3;
    if (conf <= 2) reinforce.push(id);
    else if (conf >= 5) m = Math.min(1, m + 0.02);
    newMastery[id] = clamp01(m);
  }

  return { newMastery, reinforce: dedupe(reinforce), retry: dedupe(retry) };
}

export function chooseResource(skill: Skill, taskType: TaskType, rand: () => number): Resource {
  const resources = [...(skill.resources ?? [])];
  if (!resources.length) {
    return {
      title: `${skill.name} (no resource yet)`,
      type: "text",
      credibility: "good-intuition",
      reason: "Placeholder resource.",
      url: "",
    };
  }
  const credRank: Record<string, number> = { "interview-safe": 0, "good-intuition": 1, misleading: 2 };
  const typeRank = (r: Resource) =>
    taskType === "learn"
      ? r.type === "youtube"
        ? 0
        : ["docs", "blog"].includes(r.type)
          ? 1
          : 2
      : taskType === "practice"
        ? ["docs", "blog"].includes(r.type)
          ? 0
          : 1
        : 0;
  resources.sort(
    (a, b) => (credRank[a.credibility] ?? 9) - (credRank[b.credibility] ?? 9) || typeRank(a) - typeRank(b),
  );
  const top = resources.slice(0, Math.min(2, resources.length));
  return top[Math.floor(rand() * top.length)];
}

function rankGapsByMastery(skills: Skill[], mastery: Record<string, number>): Gap[] {
  const gaps: Gap[] = skills.map((s) => {
    const c = clamp01(mastery[s.id] ?? 0.05);
    return {
      skillId: s.id,
      skillName: s.name,
      category: s.category,
      weight: s.weight,
      coverage: c,
      gap: 1 - c,
      prereqsMissing: s.prereqs.filter((p) => (mastery[p] ?? 0.05) < 0.55),
    };
  });
  gaps.sort(
    (a, b) =>
      b.gap * b.weight - a.gap * a.weight ||
      b.weight - a.weight ||
      b.gap - a.gap ||
      a.skillName.toLowerCase().localeCompare(b.skillName.toLowerCase()),
  );
  return gaps;
}

function pickFocusSkills(candidateIds: string[], mastery: Record<string, number>, maxFocus = 2): string[] {
  const scored = candidateIds.map((sid) => ({ m: mastery[sid] ?? 0.05, sid }));
  scored.sort((a, b) => a.m - b.m); // lowest mastery first
  const focus: string[] = [];
  for (const { m, sid } of scored) {
    if (focus.length >= maxFocus) break;
    if (m >= 0.8) continue;
    focus.push(sid);
  }
  return focus;
}

function pickReviewSkill(skills: Skill[], mastery: Record<string, number>, exclude: Set<string>): string | null {
  const candidates: { d: number; sid: string }[] = [];
  for (const s of skills) {
    if (exclude.has(s.id)) continue;
    const m = mastery[s.id] ?? 0.05;
    if (m >= 0.55 && m <= 0.8) candidates.push({ d: Math.abs(m - 0.68), sid: s.id });
  }
  if (!candidates.length) return null;
  candidates.sort((a, b) => a.d - b.d);
  return candidates[0].sid;
}

const learnCheck = (s: Skill) => `Success: summarize ${s.name} in 3 bullets (focus on interview wording).`;
const practiceCheck = (s: Skill) =>
  s.assessment.length ? `Practice: answer: "${s.assessment[0]}"` : "Practice: solve 2 quick questions and rate confidence.";
const explainCheck = (s: Skill) => `Explain: teach ${s.name} as if to a friend (5 sentences max).`;

export function generateDayPlan(args: {
  day: number;
  skills: Skill[];
  mastery: Record<string, number>;
  minutesPerDay?: number;
  reinforceSkills?: string[];
  retrySkills?: string[];
  seed?: number;
  topGapPool?: number;
}): DayPlan {
  const { day, skills, mastery } = args;
  const minutesPerDay = Math.max(30, Math.floor(args.minutesPerDay ?? 120));
  const reinforceSkills = args.reinforceSkills ?? [];
  const retrySkills = args.retrySkills ?? [];
  const seed = args.seed ?? 7;
  const topGapPool = args.topGapPool ?? 8;
  const rand = mulberry32((seed + day) >>> 0);

  const byId = new Map(skills.map((s) => [s.id, s] as const));
  const notes: string[] = [];

  const gapRanked = rankGapsByMastery(skills, mastery);

  let candidateIds = [
    ...retrySkills,
    ...reinforceSkills,
    ...gapRanked.slice(0, topGapPool).map((g) => g.skillId),
  ];
  candidateIds = dedupe(candidateIds.filter((id) => byId.has(id)));

  if (retrySkills.length) notes.push("You skipped some items yesterday — today starts with smaller retries.");
  if (reinforceSkills.length) notes.push("Some topics felt hard — today includes quick reinforcement blocks.");

  const focusIds = pickFocusSkills(candidateIds, mastery, 2);
  const reviewId = pickReviewSkill(skills, mastery, new Set(focusIds));

  let learnBudget = Math.round(minutesPerDay * 0.45);
  let practiceBudget = Math.round(minutesPerDay * 0.45);
  let explainBudget = minutesPerDay - learnBudget - practiceBudget;

  const items: PlanItem[] = [];
  let order = 1;
  let usedMinutes = 0;

  const addTask = (skill: Skill, taskType: TaskType, minutes: number, successCheck: string) => {
    if (minutes <= 0) return;
    if (usedMinutes + minutes > minutesPerDay) minutes = Math.max(0, minutesPerDay - usedMinutes);
    if (minutes <= 0) return;
    const res = chooseResource(skill, taskType, rand);
    items.push({
      day,
      order,
      skillId: skill.id,
      skillName: skill.name,
      category: skill.category,
      taskType,
      minutes,
      resourceTitle: res.title,
      resourceUrl: res.url,
      resourceType: res.type,
      credibility: res.credibility,
      successCheck,
    });
    usedMinutes += minutes;
    order += 1;
  };

  // 4a) Retry blocks first (short)
  for (const sid of retrySkills.slice(0, 2)) {
    const s = byId.get(sid);
    if (s) addTask(s, "learn", 15, "Finish this short retry and write 3 bullet takeaways.");
  }

  // 4b) Reinforcement blocks (short practice)
  for (const sid of reinforceSkills.slice(0, 2)) {
    const s = byId.get(sid);
    if (s) addTask(s, "practice", 15, practiceCheck(s));
  }

  // 4c) Main focus skills: prereq primer (if needed), then learn + practice + explain
  for (const sid of focusIds) {
    const s = byId.get(sid);
    if (!s) continue;

    const missingPrereqs = s.prereqs.filter((p) => (mastery[p] ?? 0.05) < 0.55);
    if (missingPrereqs.length) {
      const pre = byId.get(missingPrereqs[0]);
      if (pre) {
        addTask(pre, "review", 15, `Quick primer: explain ${pre.name} in 2–3 sentences.`);
        notes.push(`Added prerequisite primer before '${s.name}'.`);
      }
    }

    const learnMinutes = minutesPerDay >= 90 ? 20 : 15;
    addTask(s, "learn", Math.min(learnMinutes, learnBudget), learnCheck(s));
    learnBudget -= Math.min(learnMinutes, learnBudget);

    const practiceMinutes = minutesPerDay >= 120 ? 25 : 20;
    addTask(s, "practice", Math.min(practiceMinutes, practiceBudget), practiceCheck(s));
    practiceBudget -= Math.min(practiceMinutes, practiceBudget);

    addTask(s, "explain", Math.min(10, explainBudget), explainCheck(s));
    explainBudget -= Math.min(10, explainBudget);
  }

  // 4d) Review skill (confidence + spacing)
  if (reviewId && byId.has(reviewId) && usedMinutes < minutesPerDay) {
    const s = byId.get(reviewId)!;
    addTask(s, "review", Math.min(15, minutesPerDay - usedMinutes), "Review: write a tiny cheat-sheet (5 bullets).");
  }

  // 4e) If we still have time, add practice on the biggest remaining gap
  if (usedMinutes < minutesPerDay) {
    const remaining = minutesPerDay - usedMinutes;
    const usedSkills = new Set(items.map((it) => it.skillId));
    for (const g of gapRanked) {
      if (usedSkills.has(g.skillId)) continue;
      const s = byId.get(g.skillId);
      if (!s) continue;
      addTask(s, "practice", Math.min(remaining, 15), practiceCheck(s));
      break;
    }
  }

  if (!items.length) notes.push("No tasks were generated — check your skill map and inputs.");
  else notes.push("Tip: mark tasks as done + difficulty — tomorrow's plan adapts automatically.");

  return { day, totalMinutes: Math.min(minutesPerDay, usedMinutes), items, notes };
}
