import type { Skill, Resource } from "./types";

const CRED_RANK: Record<string, number> = { "interview-safe": 0, "good-intuition": 1, misleading: 2 };
export const credibilityRank = (label: string) => CRED_RANK[label] ?? 9;

export function bestResources(
  skill: Skill,
  opts: { preferType?: string; maxItems?: number } = {},
): Resource[] {
  const { preferType, maxItems = 3 } = opts;
  return [...(skill.resources ?? [])]
    .sort((a, b) => {
      const ta = preferType ? (a.type === preferType ? 0 : 1) : 0;
      const tb = preferType ? (b.type === preferType ? 0 : 1) : 0;
      return (
        credibilityRank(a.credibility) - credibilityRank(b.credibility) ||
        ta - tb ||
        a.title.toLowerCase().localeCompare(b.title.toLowerCase())
      );
    })
    .slice(0, Math.max(0, maxItems));
}

export function labelExplanation(label: string): string {
  if (label === "interview-safe") return "Accurate + sufficient depth for interviews.";
  if (label === "good-intuition") return "Helpful intuition, may miss edge cases—pair with practice.";
  if (label === "misleading") return "Risky/outdated/oversimplified—use cautiously.";
  return "Unknown credibility.";
}

export function flagMisleadingResources(skill: Skill): Resource[] {
  return (skill.resources ?? []).filter((r) => r.credibility === "misleading");
}
