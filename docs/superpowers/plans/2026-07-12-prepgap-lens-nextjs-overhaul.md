# PrepGap-Lens Next.js Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn PrepGap-Lens into a live web app where you paste a real job description + your resume and get a readiness score, ranked skill gaps, and an adaptive daily study plan — deployed on Vercel's free tier.

**Architecture:** Single Next.js (App Router, TypeScript) app on Vercel. **BYOK (bring your own key):** the user picks a provider (Google / OpenAI / Groq / Anthropic) and pastes their own API key in the UI. The key is held in React state only — never stored (no DB, no localStorage, no env var) — and sent with each request to one serverless route (`/api/analyze`), which uses it transiently to turn the pasted JD into a structured skill list + resume coverage, then discards it. All scoring and the adaptive day-planner run as deterministic, unit-tested TypeScript in `lib/` (ported faithfully from the existing Python in `core/`). Study-plan progress lives in `localStorage` (the API key never does).

**Tech Stack:** Next.js 15 (App Router) · TypeScript · Tailwind CSS · Vercel AI SDK (`ai` + `@ai-sdk/google` / `@ai-sdk/openai` / `@ai-sdk/groq` / `@ai-sdk/anthropic`) · Zod (structured-output schema) · Vitest (unit tests) · Vercel (hosting).

## Global Constraints

- Node version floor: **20+** (Vercel default; set `"engines": { "node": ">=20" }`).
- Free tier only: no paid Vercel add-ons, no paid DB, no server-side API keys. The app itself has **zero secrets** — every LLM call is made with the user's own BYOK key.
- **BYOK key handling (hard rule):** the user's API key is accepted from the request body, used for exactly one provider call, and then discarded. It is **never** persisted (no DB, no localStorage/sessionStorage, no env var, no cache) and **never logged** (do not `console.log` the request body or the key). It lives in client React state only and is lost on refresh — this is intentional.
- Supported providers: `google`, `openai`, `groq`, `anthropic` — unified behind the Vercel AI SDK. The provider id + model + key all come from the request; no provider is hardcoded as "the" provider.
- All deterministic logic (`lib/score.ts`, `lib/planner.ts`, `lib/credibility.ts`) must be pure and unit-tested — no network calls inside them.
- Ported logic must preserve the behavior of the existing `core/*.py` (same weights, thresholds, gap-priority ordering). The Python files stay in the repo as the reference during the port, then are removed in the final task.
- The model's structured output is produced via `generateObject` against a Zod schema; on validation/parse failure the route returns a 502 with a clear message, never crashes.
- Default models per provider are user-overridable in the UI (a real key + a wrong model id is a common failure; leave the knob). Defaults may drift — treat them as sane starting points, not guarantees.

---

## File Structure

```
app/
  layout.tsx            # root layout, Tailwind styles
  page.tsx              # main screen: paste JD + resume, results
  api/analyze/route.ts  # POST -> Gemini -> validated { skills[] }
lib/
  types.ts              # shared types (Skill, Coverage, Gap, PlanItem, DayPlan...)
  providers.ts          # provider registry + AI SDK model factory (BYOK key in)
  analyze.ts            # Zod schema + prompt + generateObject + coverage split
  score.ts              # readiness % + ranked gaps   (port of core/score.py)
  planner.ts            # adaptive day plan            (port of core/planner.py)
  credibility.ts        # resource credibility ranking (port of core/credibility.py)
  progress.ts           # localStorage load/save for mastery + feedback (never the key)
components/
  KeyBar.tsx            # provider dropdown + API key (password) + model field
  ResultsView.tsx       # readiness bars + gap list
  DayPlanView.tsx       # plan items + done/skip/difficulty controls
lib/__tests__/
  score.test.ts
  planner.test.ts
  analyze.test.ts       # tests the pure schema + coverage-split (no live call)
```

Removed at the end: `app.py`, `core/`, `data/skill_map_ml_intern.json` (the JD is now dynamic, not a hand-authored file).

---

### Task 1: Scaffold Next.js app + deploy a skeleton to Vercel

Gets a live URL early so deployment is proven before feature work.

**Files:**
- Create: `package.json`, `tsconfig.json`, `next.config.ts`, `app/layout.tsx`, `app/page.tsx`, `app/globals.css`, `vitest.config.ts`
- Create: `.env.local.example`

- [ ] **Step 1: Scaffold**

Run:
```bash
npx create-next-app@latest . --typescript --tailwind --app --no-src-dir --eslint --use-npm
npm install ai zod @ai-sdk/google @ai-sdk/openai @ai-sdk/groq @ai-sdk/anthropic
npm install -D vitest
```

- [ ] **Step 2: Add the test script + Node engine**

In `package.json`, add to `"scripts"`: `"test": "vitest run"`, `"test:watch": "vitest"`, and add `"engines": { "node": ">=20" }`.

- [ ] **Step 3: Create `vitest.config.ts`**

```ts
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: { environment: "node", include: ["lib/**/*.test.ts"] },
});
```

- [ ] **Step 4: Minimal landing page**

Replace `app/page.tsx` with a placeholder heading `PrepGap-Lens` so the build is green.

```tsx
export default function Home() {
  return (
    <main className="mx-auto max-w-3xl p-8">
      <h1 className="text-3xl font-bold">PrepGap-Lens</h1>
      <p className="mt-2 text-gray-600">Paste a job description and your resume to see your interview-readiness gaps.</p>
    </main>
  );
}
```

- [ ] **Step 5: Verify build**

Run: `npm run build`
Expected: build completes with no errors. (No `.env` / secrets — this app is BYOK, so there is nothing to configure on the server.)

- [ ] **Step 6: Deploy skeleton to Vercel**

Run: `npx vercel` (link project, accept defaults), then `npx vercel --prod`.
Expected: a live `*.vercel.app` URL showing the heading. No environment variables to set — the app holds no keys.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: scaffold Next.js app and deploy skeleton to Vercel"
```

---

### Task 2: Shared types + readiness scoring (port of `core/score.py`)

**Files:**
- Create: `lib/types.ts`, `lib/score.ts`, `lib/__tests__/score.test.ts`

**Interfaces:**
- Produces: types `Skill`, `Coverage` (`Record<string, number>`), `Gap`, `ScoreReport`; function `scoreReadiness(skills: Skill[], coverage: Coverage, opts?: { topKGaps?: number; prereqPenalty?: number }): ScoreReport`.

- [ ] **Step 1: Write `lib/types.ts`**

```ts
export type Credibility = "interview-safe" | "good-intuition" | "misleading";

export interface Resource {
  title: string;
  type: string;            // "youtube" | "docs" | "blog" | ...
  credibility: Credibility;
  reason: string;
  url: string;
}

export interface Skill {
  id: string;
  name: string;
  category: string;
  weight: number;          // 1..5
  keywords: string[];
  prereqs: string[];       // skill ids
  assessment: string[];
  resources: Resource[];
}

export type Coverage = Record<string, number>; // skillId -> [0,1]

export interface Gap {
  skillId: string;
  skillName: string;
  category: string;
  weight: number;
  coverage: number;
  gap: number;
  prereqsMissing: string[];
}

export interface ScoreReport {
  readinessOverall: number;                 // [0,1]
  readinessByCategory: Record<string, number>;
  coverageBySkill: Coverage;
  topGaps: Gap[];
}
```

- [ ] **Step 2: Write the failing test `lib/__tests__/score.test.ts`**

```ts
import { describe, it, expect } from "vitest";
import { scoreReadiness } from "../score";
import type { Skill } from "../types";

const skills: Skill[] = [
  { id: "a", name: "A", category: "Core", weight: 5, keywords: [], prereqs: [], assessment: [], resources: [] },
  { id: "b", name: "B", category: "Core", weight: 1, keywords: [], prereqs: ["a"], assessment: [], resources: [] },
];

describe("scoreReadiness", () => {
  it("weights overall readiness by skill weight", () => {
    const r = scoreReadiness(skills, { a: 1.0, b: 0.0 });
    // (5*1.0 + 1*0.0) / (5+1) = 0.8333
    expect(r.readinessOverall).toBeCloseTo(0.8333, 3);
  });

  it("ranks gaps by gap*weight and flags missing prereqs", () => {
    const r = scoreReadiness(skills, { a: 0.0, b: 0.0 }, { topKGaps: 2 });
    expect(r.topGaps[0].skillId).toBe("a");     // gap 1.0 * weight 5 = 5 wins
    expect(r.topGaps[1].prereqsMissing).toContain("a"); // b's prereq a is uncovered
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `npm test -- score`
Expected: FAIL — `scoreReadiness` not defined.

- [ ] **Step 4: Write `lib/score.ts`**

Port `core/score.py` faithfully. Preserve: prereq threshold `0.55`, `prereqPenalty` default `0.10`, prereq-gap-factor formula `(0.55 - preCov) / 0.55`, and gap-priority sort `(-(gap*weight), -weight, -gap, name asc)`.

```ts
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

  let totalW = 0, total = 0;
  for (const s of skills) { totalW += s.weight; total += s.weight * cov(s.id); }
  const readinessOverall = totalW > 0 ? total / totalW : 0;

  const readinessByCategory: Record<string, number> = {};
  const categories = [...new Set(skills.map((s) => s.category))];
  for (const c of categories) {
    let cw = 0, ct = 0;
    for (const s of skills.filter((s) => s.category === c)) { cw += s.weight; ct += s.weight * cov(s.id); }
    readinessByCategory[c] = cw > 0 ? ct / cw : 0;
  }

  const gaps: Gap[] = skills.map((s) => {
    let gap = 1 - cov(s.id);
    const prereqsMissing: string[] = [];
    let factor = 0;
    for (const p of s.prereqs) {
      const pc = cov(p);
      if (pc < 0.55) { prereqsMissing.push(p); factor += (0.55 - pc) / 0.55; }
    }
    if (prereqsMissing.length) gap = Math.min(1, gap + prereqPenalty * Math.min(1, factor));
    return { skillId: s.id, skillName: s.name, category: s.category, weight: s.weight, coverage: cov(s.id), gap: clamp01(gap), prereqsMissing };
  });

  gaps.sort((a, b) => {
    const ia = a.gap * a.weight, ib = b.gap * b.weight;
    return ib - ia || b.weight - a.weight || b.gap - a.gap || a.skillName.toLowerCase().localeCompare(b.skillName.toLowerCase());
  });

  const coverageBySkill: Coverage = {};
  for (const s of skills) coverageBySkill[s.id] = cov(s.id);

  return { readinessOverall: clamp01(readinessOverall), readinessByCategory, coverageBySkill, topGaps: gaps.slice(0, Math.max(0, topKGaps)) };
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `npm test -- score`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add lib/types.ts lib/score.ts lib/__tests__/score.test.ts
git commit -m "feat: port readiness scoring to TypeScript with tests"
```

---

### Task 3: Adaptive day planner + credibility (port of `core/planner.py`, `core/credibility.py`)

**Files:**
- Create: `lib/credibility.ts`, `lib/planner.ts`, `lib/__tests__/planner.test.ts`

**Interfaces:**
- Consumes: `Skill`, `Resource`, `ScoreReport` from Task 2.
- Produces:
  - `chooseResource(skill: Skill, taskType: TaskType, seed: number): Resource`
  - `initMasteryFromReport(report: ScoreReport): Record<string, number>`
  - `applyDayFeedback(mastery, feedback): { newMastery; reinforce: string[]; retry: string[] }` where `feedback: Record<string, { status: "done" | "skipped"; difficulty: "easy" | "ok" | "hard"; confidence?: number }>`
  - `generateDayPlan(args: { day; skills: Skill[]; mastery; minutesPerDay?; reinforceSkills?; retrySkills?; seed? }): DayPlan`
  - types `TaskType = "learn" | "practice" | "explain" | "review"`, `PlanItem`, `DayPlan`.

- [ ] **Step 1: Write `lib/credibility.ts`**

Port `core/credibility.py`: `CRED_RANK` (`interview-safe`=0, `good-intuition`=1, `misleading`=2, unknown=9), `credibilityRank`, `bestResources(skill, { preferType?, maxItems? })`, `labelExplanation(label)`.

```ts
import type { Skill, Resource } from "./types";

const CRED_RANK: Record<string, number> = { "interview-safe": 0, "good-intuition": 1, misleading: 2 };
export const credibilityRank = (label: string) => CRED_RANK[label] ?? 9;

export function bestResources(skill: Skill, opts: { preferType?: string; maxItems?: number } = {}): Resource[] {
  const { preferType, maxItems = 3 } = opts;
  return [...(skill.resources ?? [])]
    .sort((a, b) => {
      const ta = preferType ? (a.type === preferType ? 0 : 1) : 0;
      const tb = preferType ? (b.type === preferType ? 0 : 1) : 0;
      return credibilityRank(a.credibility) - credibilityRank(b.credibility) || ta - tb || a.title.toLowerCase().localeCompare(b.title.toLowerCase());
    })
    .slice(0, Math.max(0, maxItems));
}

export function labelExplanation(label: string): string {
  if (label === "interview-safe") return "Accurate + sufficient depth for interviews.";
  if (label === "good-intuition") return "Helpful intuition, may miss edge cases—pair with practice.";
  if (label === "misleading") return "Risky/outdated/oversimplified—use cautiously.";
  return "Unknown credibility.";
}
```

- [ ] **Step 2: Write the failing test `lib/__tests__/planner.test.ts`**

The planner uses a seeded RNG so plans are deterministic — pin that. Port a small seedable PRNG (mulberry32) since JS has no seedable `Math.random`.

```ts
import { describe, it, expect } from "vitest";
import { generateDayPlan, applyDayFeedback, initMasteryFromReport } from "../planner";
import type { Skill } from "../types";

const skills: Skill[] = [
  { id: "a", name: "A", category: "Core", weight: 5, keywords: [], prereqs: [], assessment: ["Explain A."], resources: [
    { title: "A vid", type: "youtube", credibility: "interview-safe", reason: "", url: "http://a" }] },
  { id: "b", name: "B", category: "Core", weight: 3, keywords: [], prereqs: [], assessment: ["Explain B."], resources: [
    { title: "B docs", type: "docs", credibility: "interview-safe", reason: "", url: "http://b" }] },
];

describe("planner", () => {
  it("is deterministic for a given seed and never exceeds the time budget", () => {
    const mastery = { a: 0.1, b: 0.2 };
    const p1 = generateDayPlan({ day: 1, skills, mastery, minutesPerDay: 120, seed: 7 });
    const p2 = generateDayPlan({ day: 1, skills, mastery, minutesPerDay: 120, seed: 7 });
    expect(p1).toEqual(p2);
    expect(p1.totalMinutes).toBeLessThanOrEqual(120);
    expect(p1.items.length).toBeGreaterThan(0);
  });

  it("routes skipped skills to retry and hard skills to reinforce", () => {
    const mastery = { a: 0.3, b: 0.3 };
    const { retry, reinforce } = applyDayFeedback(mastery, {
      a: { status: "skipped", difficulty: "ok" },
      b: { status: "done", difficulty: "hard", confidence: 2 },
    });
    expect(retry).toContain("a");
    expect(reinforce).toContain("b");
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `npm test -- planner`
Expected: FAIL — module not found.

- [ ] **Step 4: Write `lib/planner.ts`**

Port `core/planner.py` faithfully, preserving all constants and behavior:
- feedback deltas: `easy 0.12`, `ok 0.08`, `hard 0.04`; `hard` and `confidence<=2` push to `reinforce`; `confidence>=5` adds `+0.02`; `skipped` → `retry`.
- time buckets: `learn 45%`, `practice 45%`, `explain/review remainder`; retry/reinforce blocks 15m each (max 2); focus = up to 2 lowest-mastery candidates with mastery `< 0.80`; review skill picked from mastery `0.55..0.80` nearest `0.68`; prereq primer (15m) when a prereq mastery `< 0.55`.
- resource choice: credibility first, then task-type type-preference (`learn`→youtube; `practice`→docs/blog), random pick among top 2.
- Replace Python's `random.Random(seed + day)` with a seeded mulberry32 keyed by `seed + day` so output is deterministic and testable.

```ts
import type { Skill, Resource, ScoreReport } from "./types";

export type TaskType = "learn" | "practice" | "explain" | "review";
export interface PlanItem {
  day: number; order: number; skillId: string; skillName: string; category: string;
  taskType: TaskType; minutes: number; resourceTitle: string; resourceUrl: string;
  resourceType: string; credibility: string; successCheck: string;
}
export interface DayPlan { day: number; totalMinutes: number; items: PlanItem[]; notes: string[]; }

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

// seedable PRNG so plans are reproducible (JS Math.random isn't seedable)
function mulberry32(seed: number) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

// NOTE: full body ports core/planner.py; see that file for the reference logic.
// Key exported functions below; implement bodies preserving the constants listed above.
export function initMasteryFromReport(report: ScoreReport): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [id, c] of Object.entries(report.coverageBySkill)) out[id] = clamp01(c);
  return out;
}

export function applyDayFeedback(
  mastery: Record<string, number>,
  feedback: Record<string, { status: "done" | "skipped"; difficulty: "easy" | "ok" | "hard"; confidence?: number }>,
): { newMastery: Record<string, number>; reinforce: string[]; retry: string[] } {
  const newMastery = { ...mastery };
  const reinforce: string[] = [], retry: string[] = [];
  for (const [id, fb] of Object.entries(feedback)) {
    let m = clamp01(newMastery[id] ?? 0.05);
    if (fb.status === "skipped") { retry.push(id); continue; }
    if (fb.difficulty === "easy") m = Math.min(1, m + 0.12);
    else if (fb.difficulty === "ok") m = Math.min(1, m + 0.08);
    else { m = Math.min(1, m + 0.04); reinforce.push(id); }
    const conf = fb.confidence ?? 3;
    if (conf <= 2) reinforce.push(id);
    else if (conf >= 5) m = Math.min(1, m + 0.02);
    newMastery[id] = clamp01(m);
  }
  const dedupe = (xs: string[]) => [...new Set(xs)];
  return { newMastery, reinforce: dedupe(reinforce), retry: dedupe(retry) };
}

export function chooseResource(skill: Skill, taskType: TaskType, rand: () => number): Resource {
  const resources = [...(skill.resources ?? [])];
  if (!resources.length) return { title: `${skill.name} (no resource yet)`, type: "text", credibility: "good-intuition", reason: "Placeholder.", url: "" };
  const credRank: Record<string, number> = { "interview-safe": 0, "good-intuition": 1, misleading: 2 };
  const typeRank = (r: Resource) =>
    taskType === "learn" ? (r.type === "youtube" ? 0 : ["docs", "blog"].includes(r.type) ? 1 : 2)
    : taskType === "practice" ? (["docs", "blog"].includes(r.type) ? 0 : 1) : 0;
  resources.sort((a, b) => (credRank[a.credibility] ?? 9) - (credRank[b.credibility] ?? 9) || typeRank(a) - typeRank(b));
  const top = resources.slice(0, Math.min(2, resources.length));
  return top[Math.floor(rand() * top.length)];
}

export function generateDayPlan(args: {
  day: number; skills: Skill[]; mastery: Record<string, number>;
  minutesPerDay?: number; reinforceSkills?: string[]; retrySkills?: string[]; seed?: number;
}): DayPlan {
  // Port the step-by-step body of core/planner.py:generate_day_plan using
  // rand = mulberry32((args.seed ?? 7) + args.day), the constants above, and
  // the same task ordering (retry -> reinforce -> focus(primer/learn/practice/explain) -> review -> filler).
  // Return { day, totalMinutes, items, notes }.
  throw new Error("implement per core/planner.py reference");
}
```

- [ ] **Step 5: Fill in `generateDayPlan` and run tests to green**

Implement the `generateDayPlan` body per `core/planner.py`. Run: `npm test -- planner`
Expected: PASS (2 tests). Both the determinism test and the feedback-routing test pass.

- [ ] **Step 6: Commit**

```bash
git add lib/credibility.ts lib/planner.ts lib/__tests__/planner.test.ts
git commit -m "feat: port adaptive day planner and credibility ranking to TypeScript"
```

---

### Task 4: BYOK multi-provider analyze route — JD + resume → structured skills + coverage

This is the missing "based on JD" core. One serverless route, one LLM call via the user's own key, structured output validated by Zod through `generateObject`.

**Files:**
- Create: `lib/providers.ts`, `lib/analyze.ts`, `app/api/analyze/route.ts`, `lib/__tests__/analyze.test.ts`

**Interfaces:**
- Produces:
  - `type ProviderId = "google" | "openai" | "groq" | "anthropic"`; `PROVIDERS` registry (label, defaultModel, keysUrl); `getModel(provider, apiKey, model?)`.
  - `AnalyzeSchema` (Zod) + `splitCoverage(parsed): { skills: Skill[]; coverage: Record<string, number> }` (pure) + `runAnalyze({ jd, resume, provider, apiKey, model })`.
  - POST `/api/analyze` body `{ jd, resume, provider, apiKey, model? }` → `{ skills: Skill[]; report: ScoreReport }`.

- [ ] **Step 1: Write `lib/providers.ts` (BYOK model factory)**

```ts
import type { LanguageModel } from "ai";

export type ProviderId = "google" | "openai" | "groq" | "anthropic";

export const PROVIDERS: Record<ProviderId, { label: string; defaultModel: string; keysUrl: string }> = {
  google:    { label: "Google (Gemini)", defaultModel: "gemini-2.0-flash",        keysUrl: "https://aistudio.google.com/app/apikey" },
  openai:    { label: "OpenAI",          defaultModel: "gpt-4o-mini",             keysUrl: "https://platform.openai.com/api-keys" },
  groq:      { label: "Groq",            defaultModel: "llama-3.3-70b-versatile", keysUrl: "https://console.groq.com/keys" },
  anthropic: { label: "Anthropic",       defaultModel: "claude-3-5-haiku-latest", keysUrl: "https://console.anthropic.com/settings/keys" },
};

// BYOK: the user's key comes in per call and is only used to build this model.
// Dynamic import so only the chosen provider's SDK loads.
export async function getModel(provider: ProviderId, apiKey: string, model?: string): Promise<LanguageModel> {
  const id = model?.trim() || PROVIDERS[provider].defaultModel;
  switch (provider) {
    case "google":    { const { createGoogleGenerativeAI } = await import("@ai-sdk/google"); return createGoogleGenerativeAI({ apiKey })(id); }
    case "openai":    { const { createOpenAI } = await import("@ai-sdk/openai");             return createOpenAI({ apiKey })(id); }
    case "groq":      { const { createGroq } = await import("@ai-sdk/groq");                 return createGroq({ apiKey })(id); }
    case "anthropic": { const { createAnthropic } = await import("@ai-sdk/anthropic");       return createAnthropic({ apiKey })(id); }
    default: throw new Error(`Unknown provider: ${provider}`);
  }
}
```

- [ ] **Step 2: Write `lib/analyze.ts` (schema + prompt + coverage split + runAnalyze)**

```ts
import { generateObject } from "ai";
import { z } from "zod";
import type { Skill } from "./types";
import { getModel, type ProviderId } from "./providers";

export const AnalyzeSchema = z.object({
  skills: z.array(z.object({
    id: z.string(),
    name: z.string(),
    category: z.string(),
    weight: z.number().int().min(1).max(5),
    keywords: z.array(z.string()).default([]),
    prereqs: z.array(z.string()).default([]),
    assessment: z.array(z.string()).default([]),
    coverage: z.number().min(0).max(1),            // how well the resume covers this skill
    resources: z.array(z.object({
      title: z.string(), type: z.string(),
      credibility: z.enum(["interview-safe", "good-intuition", "misleading"]),
      reason: z.string(), url: z.string(),
    })).default([]),
  })).min(1),
});
export type AnalyzeParsed = z.infer<typeof AnalyzeSchema>;

// pure + tested: drop per-skill coverage out of the Skill shape into its own map
export function splitCoverage(parsed: AnalyzeParsed): { skills: Skill[]; coverage: Record<string, number> } {
  const skills: Skill[] = parsed.skills.map(({ coverage, ...s }) => s);
  const coverage: Record<string, number> = {};
  for (const s of parsed.skills) coverage[s.id] = s.coverage;
  return { skills, coverage };
}

export function buildPrompt(jd: string, resume: string): string {
  return `You are an interview-prep analyst. Read the JOB DESCRIPTION and the CANDIDATE RESUME, then produce a structured skill gap analysis.
Rules: output 8-15 skills the JOB actually requires. weight = importance to THIS job (5=core, 1=nice-to-have). prereqs must only reference ids you also output. coverage = how convincingly the RESUME demonstrates that skill (0=absent, 1=clearly proven). assessment = one likely interview question per skill. Suggest 1-2 real, well-known resources per skill; mark anything you are unsure about as "good-intuition", never invent a specific URL you are not confident exists.
JOB DESCRIPTION:\n${jd}\n\nRESUME:\n${resume}`;
}

// ponytail: LLM-suggested resource URLs can still be wrong. Acceptable for MVP;
// upgrade path = a curated resource table keyed by skill id, LLM only picks from it.
export async function runAnalyze(input: {
  jd: string; resume: string; provider: ProviderId; apiKey: string; model?: string;
}): Promise<{ skills: Skill[]; coverage: Record<string, number> }> {
  const model = await getModel(input.provider, input.apiKey, input.model);
  const { object } = await generateObject({ model, schema: AnalyzeSchema, prompt: buildPrompt(input.jd, input.resume) });
  return splitCoverage(object);
}
```

- [ ] **Step 3: Write the failing test `lib/__tests__/analyze.test.ts`** (pure functions only — no live call)

```ts
import { describe, it, expect } from "vitest";
import { AnalyzeSchema, splitCoverage } from "../analyze";

const parsed = AnalyzeSchema.parse({
  skills: [
    { id: "python", name: "Python", category: "Languages", weight: 5, keywords: ["python"], prereqs: [], assessment: ["FizzBuzz?"], coverage: 0.9, resources: [] },
    { id: "ml", name: "ML", category: "Core", weight: 4, keywords: ["scikit"], prereqs: ["python"], assessment: ["Overfitting?"], coverage: 0.4, resources: [] },
  ],
});

describe("analyze", () => {
  it("splits coverage out of the skill objects", () => {
    const { skills, coverage } = splitCoverage(parsed);
    expect(skills).toHaveLength(2);
    expect(coverage.python).toBe(0.9);
    expect(coverage.ml).toBe(0.4);
    expect(skills[0]).not.toHaveProperty("coverage");
  });

  it("rejects an out-of-range weight", () => {
    expect(() => AnalyzeSchema.parse({ skills: [{ id: "x", name: "X", category: "C", weight: 9, coverage: 0.5 }] })).toThrow();
  });
});
```

- [ ] **Step 4: Run tests**

Run: `npm test -- analyze`
Expected: PASS (2 tests). (`splitCoverage` and schema exist from Steps 1–2.)

- [ ] **Step 5: Write `app/api/analyze/route.ts`**

```ts
import { NextResponse } from "next/server";
import { runAnalyze } from "@/lib/analyze";
import { scoreReadiness } from "@/lib/score";
import type { ProviderId } from "@/lib/providers";

export const runtime = "nodejs";
export const maxDuration = 30;

export async function POST(req: Request) {
  // BYOK: never log the body — it carries the user's API key.
  try {
    const { jd, resume, provider, apiKey, model } = await req.json();
    if (!jd?.trim() || !resume?.trim()) return NextResponse.json({ error: "Provide both a job description and a resume." }, { status: 400 });
    if (!apiKey?.trim()) return NextResponse.json({ error: "Enter your provider API key." }, { status: 400 });
    if (!["google", "openai", "groq", "anthropic"].includes(provider)) return NextResponse.json({ error: "Pick a provider." }, { status: 400 });

    const { skills, coverage } = await runAnalyze({ jd, resume, provider: provider as ProviderId, apiKey, model });
    const report = scoreReadiness(skills, coverage, { topKGaps: 6 });
    return NextResponse.json({ skills, report });
  } catch (err) {
    // provider auth errors, bad model id, or schema/parse failures land here
    const msg = err instanceof Error ? err.message : "Analysis failed";
    return NextResponse.json({ error: `Could not analyze: ${msg}` }, { status: 502 });
  }
}
```

- [ ] **Step 6: Smoke-test the route locally** (needs a real key you own)

Run: `npm run dev`, then in another shell (swap in your own provider/key):
```bash
curl -s -X POST localhost:3000/api/analyze -H 'content-type: application/json' \
  -d '{"provider":"google","apiKey":"YOUR_KEY","jd":"Python ML engineer, scikit-learn, model evaluation.","resume":"Built classification models in Python; evaluated with F1 and ROC-AUC."}' | head -c 400
```
Expected: JSON with `skills` and `report.readinessOverall`. A bad key returns the 502 message.

- [ ] **Step 7: Commit**

```bash
git add lib/providers.ts lib/analyze.ts app/api/analyze/route.ts lib/__tests__/analyze.test.ts
git commit -m "feat: BYOK multi-provider /api/analyze (JD + resume -> skills + readiness)"
```

---

### Task 5: Main UI — BYOK key bar, paste JD + resume, show readiness + gaps

**Files:**
- Modify: `app/page.tsx`
- Create: `components/KeyBar.tsx`, `components/ResultsView.tsx`

**Interfaces:**
- Consumes: `PROVIDERS`, `ProviderId` (Task 4); POST `/api/analyze` → `{ skills, report }`.
- Produces: client state `provider`, `apiKey`, `model`, `skills`, `report` used by Task 6.

- [ ] **Step 1: Build `components/KeyBar.tsx`**

A controlled row: a provider `<select>` populated from `PROVIDERS` (label per option), an `<input type="password">` for the API key, and an optional `<input>` for model (placeholder = `PROVIDERS[provider].defaultModel`). Props: `{ provider, apiKey, model, onChange }`. Include a small "Get a key ↗" link to `PROVIDERS[provider].keysUrl` and a one-line reassurance: *"Your key stays in this browser tab, is used only for this request, and is never stored or logged."* The key value must live in the parent's React state only — do **not** write it to localStorage/sessionStorage.

- [ ] **Step 2: Build `components/ResultsView.tsx`**

Render `report.readinessOverall` as a big percentage, a bar per `readinessByCategory` entry, and a list of `report.topGaps` (skill name, category, `weight`, coverage %, and a "prereqs missing" note when non-empty). Plain Tailwind — no chart library.

- [ ] **Step 3: Build `app/page.tsx`** (mark `"use client"`)

Render `<KeyBar>`, two `<textarea>`s (JD, resume), and an Analyze button. Keep `provider` (default `"google"`), `apiKey`, `model`, `jd`, `resume`, `skills`, `report` in `useState`. On submit: `POST /api/analyze` with `{ jd, resume, provider, apiKey, model }`, show a loading state, render `<ResultsView>` on success and the returned `error` string on failure.

- [ ] **Step 4: Manual verification**

Run: `npm run dev`, pick a provider, paste your own key + a real JD + your resume, click Analyze.
Expected: readiness %, category bars, and a ranked gap list appear. Empty JD/resume or empty key shows the 400 message; a bad key/model shows the 502 message. Refreshing the page clears the key field (confirms it isn't persisted).

- [ ] **Step 5: Commit**

```bash
git add app/page.tsx components/KeyBar.tsx components/ResultsView.tsx
git commit -m "feat: BYOK key bar + JD/resume input UI with readiness and gap results"
```

---

### Task 6: Adaptive day-plan UI + localStorage progress

**Files:**
- Create: `lib/progress.ts`, `components/DayPlanView.tsx`
- Modify: `app/page.tsx`

**Interfaces:**
- Consumes: `skills`, `report` (Task 5); `generateDayPlan`, `applyDayFeedback`, `initMasteryFromReport` (Task 3).
- Produces: persisted `{ mastery, day }` in `localStorage`.

- [ ] **Step 1: Write `lib/progress.ts`**

```ts
const KEY = "prepgap.progress.v1";
export type Progress = { mastery: Record<string, number>; day: number };

export function loadProgress(): Progress | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(KEY);
  return raw ? (JSON.parse(raw) as Progress) : null;
}
export function saveProgress(p: Progress): void {
  if (typeof window !== "undefined") window.localStorage.setItem(KEY, JSON.stringify(p));
}
export function clearProgress(): void {
  if (typeof window !== "undefined") window.localStorage.removeItem(KEY);
}
```

- [ ] **Step 2: Build `components/DayPlanView.tsx`**

Show `DayPlan.items` (order, task type, skill, minutes, credibility badge, resource link, success check) and `notes`. Each item gets a status control (done/skip) + difficulty (easy/ok/hard). A "Generate next day" button calls `applyDayFeedback` → `generateDayPlan(day+1, ...)`, persists via `saveProgress`, and re-renders.

- [ ] **Step 3: Wire into `app/page.tsx`**

After a successful analyze: `initMasteryFromReport(report)` (or `loadProgress()` if present), `generateDayPlan({ day: 1, skills, mastery })`, render `<DayPlanView>`. On mount, if `loadProgress()` exists, offer a "Resume plan" affordance.

- [ ] **Step 4: Manual verification**

Run: `npm run dev`. Analyze, mark day 1 items, click "Generate next day".
Expected: day 2 differs (retries for skipped, reinforcement for hard), progress survives a page refresh, and "Reset" clears it.

- [ ] **Step 5: Commit**

```bash
git add lib/progress.ts components/DayPlanView.tsx app/page.tsx
git commit -m "feat: adaptive day-plan UI with localStorage progress"
```

---

### Task 7: README, cleanup, and production deploy

**Files:**
- Modify: `README.md`
- Delete: `app.py`, `core/`, `data/skill_map_ml_intern.json`

- [ ] **Step 1: Remove the retired Python reference**

```bash
git rm -r app.py core data/skill_map_ml_intern.json
```

- [ ] **Step 2: Write a real `README.md`**

Include: one-line pitch, live Vercel link, a screenshot/GIF, "How it works" (paste JD + resume + your own key → chosen provider → skills → readiness → adaptive plan), local setup (`npm install`, `npm run dev` — no env/secrets needed), the BYOK model (supported providers + where to get a free key, e.g. Google AI Studio and Groq both have free tiers), an explicit **privacy note** ("your API key is held in your browser tab only, sent over HTTPS for a single request, and never stored or logged server-side"), `npm test`, and a short "Design notes" paragraph on the deterministic scoring/planner + the LLM boundary.

- [ ] **Step 3: Full check**

Run: `npm test && npm run build`
Expected: all tests pass, build succeeds.

- [ ] **Step 4: Deploy to production**

Run: `npx vercel --prod`. Confirm the live URL analyzes a real JD end to end. Confirm `GEMINI_API_KEY` is set for the Production environment.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs: production README; remove retired Python CLI"
```

---

## Self-Review

- **Spec coverage:** JD ingestion (Task 4) ✓ · resume-vs-JD gap (Tasks 2, 4) ✓ · readiness score (Task 2) ✓ · adaptive plan (Tasks 3, 6) ✓ · credibility-tagged resources (Tasks 3, 4) ✓ · **BYOK, 4 providers, key never stored (Tasks 4, 5)** ✓ · Vercel free-tier deploy (Tasks 1, 7) ✓ · frontend (Tasks 5, 6) ✓ · tests on core logic (Tasks 2, 3, 4) ✓.
- **BYOK check:** key enters via request body, used for one `generateObject` call, never persisted or logged (route comment forbids logging the body); client holds it in React state only, lost on refresh.
- **Known ceiling (marked, not hidden):** LLM-suggested resource URLs may be inaccurate — flagged with a `ponytail:` note in `lib/analyze.ts`; upgrade path is a curated resource table.
- **Type consistency:** `Skill`, `Coverage`, `ScoreReport`, `Gap`, `DayPlan`, `PlanItem` defined once in `lib/types.ts`/`lib/planner.ts` and reused; `generateDayPlan`/`applyDayFeedback`/`initMasteryFromReport` signatures match across Tasks 3 and 6; `ProviderId`/`PROVIDERS` defined once in `lib/providers.ts`, reused in Tasks 4 and 5.
- **Free-tier check:** Vercel Hobby (Next.js + Node serverless), no server keys (BYOK), no DB (localStorage). No paid services.
