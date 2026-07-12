"use client";

import { useState } from "react";
import type { DayPlan, DayFeedback } from "@/lib/planner";

const DIFFICULTIES: DayFeedback["difficulty"][] = ["easy", "ok", "hard"];

const credClass = (c: string) => (c === "interview-safe" ? "safe" : c === "misleading" ? "bad" : "good");

export function DayPlanView({
  plan,
  onNextDay,
}: {
  plan: DayPlan;
  onNextDay: (feedback: Record<string, DayFeedback>) => void;
}) {
  const [feedback, setFeedback] = useState<Record<string, DayFeedback>>({});

  const setStatus = (skillId: string, status: DayFeedback["status"]) =>
    setFeedback((f) => ({ ...f, [skillId]: { ...f[skillId], status, difficulty: f[skillId]?.difficulty ?? "ok" } }));
  const setDifficulty = (skillId: string, difficulty: DayFeedback["difficulty"]) =>
    setFeedback((f) => ({ ...f, [skillId]: { ...f[skillId], difficulty, status: f[skillId]?.status ?? "done" } }));

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>
          Day {String(plan.day).padStart(2, "0")}
        </h2>
        <span className="chip">{plan.totalMinutes} min · adaptive</span>
      </div>

      <div>
        {plan.items.map((it) => {
          const fb = feedback[it.skillId];
          return (
            <div key={`${it.order}-${it.skillId}`} className="plan-item">
              <div className="plan-line">
                <span className="ord">{String(it.order).padStart(2, "0")}</span>
                <span className="ttype">{it.taskType}</span>
                <span className="sname">{it.skillName}</span>
                <span className="mins">{it.minutes}m</span>
                <span className={`cred ${credClass(it.credibility)}`}>{it.credibility}</span>
              </div>

              <p className="plan-check">{it.successCheck}</p>
              {it.resourceUrl && (
                <a href={it.resourceUrl} target="_blank" rel="noopener noreferrer">
                  {it.resourceTitle}
                </a>
              )}

              <div className="fb seg">
                <button type="button" aria-pressed={fb?.status === "done"} onClick={() => setStatus(it.skillId, "done")}>
                  Done
                </button>
                <button type="button" aria-pressed={fb?.status === "skipped"} onClick={() => setStatus(it.skillId, "skipped")}>
                  Skip
                </button>
                <span className="divider" />
                {DIFFICULTIES.map((d) => (
                  <button
                    key={d}
                    type="button"
                    aria-pressed={fb?.difficulty === d && fb?.status === "done"}
                    onClick={() => setDifficulty(it.skillId, d)}
                  >
                    {d}
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {plan.notes.length > 0 && (
        <ul className="notes">
          {plan.notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      )}

      <div className="row-actions">
        <button type="button" className="btn" onClick={() => onNextDay(feedback)}>
          Generate day {plan.day + 1}
        </button>
      </div>
    </section>
  );
}
