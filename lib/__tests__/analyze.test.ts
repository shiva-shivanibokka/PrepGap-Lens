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
