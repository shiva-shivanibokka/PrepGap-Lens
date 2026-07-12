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
    expect(r.topGaps[0].skillId).toBe("a"); // gap 1.0 * weight 5 = 5 wins
    expect(r.topGaps[1].prereqsMissing).toContain("a"); // b's prereq a is uncovered
  });
});
