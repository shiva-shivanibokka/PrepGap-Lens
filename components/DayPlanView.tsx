"use client";

import { useState } from "react";
import type { DayPlan, DayFeedback } from "@/lib/planner";

const DIFFICULTIES: DayFeedback["difficulty"][] = ["easy", "ok", "hard"];

const credChip = (c: string) =>
  c === "interview-safe"
    ? "border-emerald-300 text-emerald-700 dark:border-emerald-800 dark:text-emerald-400"
    : c === "misleading"
      ? "border-rose-300 text-rose-700 dark:border-rose-800 dark:text-rose-400"
      : "border-amber-300 text-amber-700 dark:border-amber-800 dark:text-amber-400";

export function DayPlanView({
  plan,
  onNextDay,
  busy,
}: {
  plan: DayPlan;
  onNextDay: (feedback: Record<string, DayFeedback>) => void;
  busy: boolean;
}) {
  // feedback keyed by skillId, mirroring applyDayFeedback's input
  const [feedback, setFeedback] = useState<Record<string, DayFeedback>>({});

  const setStatus = (skillId: string, status: DayFeedback["status"]) =>
    setFeedback((f) => ({ ...f, [skillId]: { ...f[skillId], status, difficulty: f[skillId]?.difficulty ?? "ok" } }));
  const setDifficulty = (skillId: string, difficulty: DayFeedback["difficulty"]) =>
    setFeedback((f) => ({ ...f, [skillId]: { ...f[skillId], difficulty, status: f[skillId]?.status ?? "done" } }));

  return (
    <section className="mt-10 rounded-2xl border border-zinc-200 p-6 dark:border-zinc-800">
      <div className="flex items-baseline justify-between">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-zinc-500">
          Day {String(plan.day).padStart(2, "0")} · {plan.totalMinutes} min
        </p>
      </div>

      <ol className="mt-4 space-y-4">
        {plan.items.map((it) => {
          const fb = feedback[it.skillId];
          return (
            <li key={`${it.order}-${it.skillId}`} className="border-t border-zinc-100 pt-4 first:border-t-0 first:pt-0 dark:border-zinc-900">
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                <span className="font-mono text-xs text-zinc-400 tabular-nums">{String(it.order).padStart(2, "0")}</span>
                <span className="font-mono text-[0.7rem] uppercase tracking-wider text-zinc-500">{it.taskType}</span>
                <span className="font-medium">{it.skillName}</span>
                <span className="font-mono text-xs text-zinc-400">{it.minutes}m</span>
                <span className={`rounded-full border px-2 py-0.5 font-mono text-[0.65rem] uppercase tracking-wider ${credChip(it.credibility)}`}>
                  {it.credibility}
                </span>
              </div>

              <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">{it.successCheck}</p>
              {it.resourceUrl && (
                <a
                  href={it.resourceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-1 inline-block text-sm text-zinc-900 underline decoration-zinc-300 underline-offset-2 dark:text-zinc-100 dark:decoration-zinc-700"
                >
                  {it.resourceTitle}
                </a>
              )}

              <div className="mt-2 flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => setStatus(it.skillId, "done")}
                  className={`rounded-full border px-3 py-1 text-xs ${fb?.status === "done" ? "border-emerald-500 bg-emerald-500 text-white" : "border-zinc-300 dark:border-zinc-700"}`}
                >
                  Done
                </button>
                <button
                  type="button"
                  onClick={() => setStatus(it.skillId, "skipped")}
                  className={`rounded-full border px-3 py-1 text-xs ${fb?.status === "skipped" ? "border-zinc-900 bg-zinc-900 text-white dark:border-zinc-100 dark:bg-zinc-100 dark:text-zinc-900" : "border-zinc-300 dark:border-zinc-700"}`}
                >
                  Skip
                </button>
                <span className="mx-1 h-4 w-px bg-zinc-200 dark:bg-zinc-800" />
                {DIFFICULTIES.map((d) => (
                  <button
                    key={d}
                    type="button"
                    onClick={() => setDifficulty(it.skillId, d)}
                    className={`rounded-full border px-3 py-1 font-mono text-xs ${fb?.difficulty === d && fb?.status === "done" ? "border-zinc-900 bg-zinc-900 text-white dark:border-zinc-100 dark:bg-zinc-100 dark:text-zinc-900" : "border-zinc-300 text-zinc-500 dark:border-zinc-700"}`}
                  >
                    {d}
                  </button>
                ))}
              </div>
            </li>
          );
        })}
      </ol>

      {plan.notes.length > 0 && (
        <ul className="mt-5 space-y-1 border-t border-zinc-100 pt-4 dark:border-zinc-900">
          {plan.notes.map((n, i) => (
            <li key={i} className="text-xs text-zinc-500">
              {n}
            </li>
          ))}
        </ul>
      )}

      <button
        type="button"
        disabled={busy}
        onClick={() => onNextDay(feedback)}
        className="mt-5 rounded-full bg-zinc-900 px-5 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900"
      >
        {busy ? "Building…" : `Generate day ${plan.day + 1}`}
      </button>
    </section>
  );
}
