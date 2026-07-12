import { describe, it, expect } from "vitest";
import { generateDayPlan, applyDayFeedback } from "../planner";
import type { Skill } from "../types";

const skills: Skill[] = [
  {
    id: "a", name: "A", category: "Core", weight: 5, keywords: [], prereqs: [], assessment: ["Explain A."],
    resources: [{ title: "A vid", type: "youtube", credibility: "interview-safe", reason: "", url: "http://a" }],
  },
  {
    id: "b", name: "B", category: "Core", weight: 3, keywords: [], prereqs: [], assessment: ["Explain B."],
    resources: [{ title: "B docs", type: "docs", credibility: "interview-safe", reason: "", url: "http://b" }],
  },
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
