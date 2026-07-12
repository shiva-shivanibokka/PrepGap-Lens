import type { Skill, ScoreReport } from "./types";

// Only non-secret working state is persisted. The API key is NEVER written here.
const KEY = "prepgap.progress.v1";

export interface Progress {
  skills: Skill[];
  report: ScoreReport;
  mastery: Record<string, number>;
  day: number;
}

export function loadProgress(): Progress | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as Progress;
  } catch {
    return null;
  }
}

export function saveProgress(p: Progress): void {
  if (typeof window !== "undefined") window.localStorage.setItem(KEY, JSON.stringify(p));
}

export function clearProgress(): void {
  if (typeof window !== "undefined") window.localStorage.removeItem(KEY);
}
