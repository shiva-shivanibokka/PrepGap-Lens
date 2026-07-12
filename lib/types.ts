export type Credibility = "interview-safe" | "good-intuition" | "misleading";

export interface Resource {
  title: string;
  type: string; // "youtube" | "docs" | "blog" | ...
  credibility: Credibility;
  reason: string;
  url: string;
}

export interface Skill {
  id: string;
  name: string;
  category: string;
  weight: number; // 1..5
  keywords: string[];
  prereqs: string[]; // skill ids
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
  readinessOverall: number; // [0,1]
  readinessByCategory: Record<string, number>;
  coverageBySkill: Coverage;
  topGaps: Gap[];
}
