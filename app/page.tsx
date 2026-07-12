"use client";

import { useEffect, useState } from "react";
import { KeyBar, type KeyState } from "@/components/KeyBar";
import { ResultsView } from "@/components/ResultsView";
import { DayPlanView } from "@/components/DayPlanView";
import type { Skill, ScoreReport } from "@/lib/types";
import {
  generateDayPlan,
  applyDayFeedback,
  initMasteryFromReport,
  type DayPlan,
  type DayFeedback,
} from "@/lib/planner";
import { loadProgress, saveProgress, clearProgress, type Progress } from "@/lib/progress";

export default function Home() {
  const [key, setKey] = useState<KeyState>({ provider: "google", apiKey: "", model: "" });
  const [jd, setJd] = useState("");
  const [resume, setResume] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [skills, setSkills] = useState<Skill[] | null>(null);
  const [report, setReport] = useState<ScoreReport | null>(null);
  const [mastery, setMastery] = useState<Record<string, number>>({});
  const [day, setDay] = useState(1);
  const [plan, setPlan] = useState<DayPlan | null>(null);

  const [saved, setSaved] = useState<Progress | null>(null);

  useEffect(() => setSaved(loadProgress()), []);

  const canAnalyze = key.apiKey.trim() && jd.trim() && resume.trim() && !loading;

  async function analyze() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ jd, resume, provider: key.provider, apiKey: key.apiKey, model: key.model }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error ?? "Something went wrong.");
        return;
      }
      const s: Skill[] = data.skills;
      const r: ScoreReport = data.report;
      const m = initMasteryFromReport(r);
      const p = generateDayPlan({ day: 1, skills: s, mastery: m });
      setSkills(s);
      setReport(r);
      setMastery(m);
      setDay(1);
      setPlan(p);
      setSaved(null);
      saveProgress({ skills: s, report: r, mastery: m, day: 1 });
    } catch {
      setError("Could not reach the analyzer. Check your connection and try again.");
    } finally {
      setLoading(false);
    }
  }

  function nextDay(feedback: Record<string, DayFeedback>) {
    if (!skills) return;
    const { newMastery, reinforce, retry } = applyDayFeedback(mastery, feedback);
    const nd = day + 1;
    const p = generateDayPlan({ day: nd, skills, mastery: newMastery, reinforceSkills: reinforce, retrySkills: retry });
    setMastery(newMastery);
    setDay(nd);
    setPlan(p);
    if (report) saveProgress({ skills, report, mastery: newMastery, day: nd });
  }

  function resume_() {
    if (!saved) return;
    setSkills(saved.skills);
    setReport(saved.report);
    setMastery(saved.mastery);
    setDay(saved.day);
    setPlan(generateDayPlan({ day: saved.day, skills: saved.skills, mastery: saved.mastery }));
    setSaved(null);
  }

  function reset() {
    clearProgress();
    setSkills(null);
    setReport(null);
    setPlan(null);
    setMastery({});
    setDay(1);
    setSaved(null);
  }

  return (
    <main className="mx-auto w-full max-w-3xl px-6 py-14">
      <header>
        <p className="font-mono text-xs uppercase tracking-[0.25em] text-zinc-500">Interview-readiness lens</p>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">PrepGap-Lens</h1>
        <p className="mt-3 max-w-xl text-zinc-600 dark:text-zinc-400">
          Paste a job description and your resume. See how ready you are, which gaps cost you the most, and get an
          adaptive day-by-day plan to close them.
        </p>
      </header>

      {saved && !report && (
        <div className="mt-8 flex items-center justify-between rounded-xl border border-zinc-200 bg-zinc-50/60 px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900/40">
          <span className="text-sm text-zinc-600 dark:text-zinc-400">
            You have a saved plan on day {saved.day}.
          </span>
          <div className="flex gap-2">
            <button onClick={resume_} className="rounded-full bg-zinc-900 px-4 py-1.5 text-sm text-white dark:bg-zinc-100 dark:text-zinc-900">
              Resume
            </button>
            <button onClick={reset} className="rounded-full border border-zinc-300 px-4 py-1.5 text-sm dark:border-zinc-700">
              Discard
            </button>
          </div>
        </div>
      )}

      <div className="mt-8">
        <KeyBar value={key} onChange={(patch) => setKey((k) => ({ ...k, ...patch }))} />
      </div>

      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        <label className="grid gap-1.5">
          <span className="font-mono text-xs uppercase tracking-wider text-zinc-500">Job description</span>
          <textarea
            value={jd}
            onChange={(e) => setJd(e.target.value)}
            rows={10}
            placeholder="Paste the full job posting…"
            className="resize-y rounded-xl border border-zinc-300 bg-white p-3 text-sm dark:border-zinc-700 dark:bg-zinc-950"
          />
        </label>
        <label className="grid gap-1.5">
          <span className="font-mono text-xs uppercase tracking-wider text-zinc-500">Your resume</span>
          <textarea
            value={resume}
            onChange={(e) => setResume(e.target.value)}
            rows={10}
            placeholder="Paste your resume or skills summary…"
            className="resize-y rounded-xl border border-zinc-300 bg-white p-3 text-sm dark:border-zinc-700 dark:bg-zinc-950"
          />
        </label>
      </div>

      <div className="mt-5 flex items-center gap-4">
        <button
          onClick={analyze}
          disabled={!canAnalyze}
          className="rounded-full bg-zinc-900 px-6 py-2.5 text-sm font-medium text-white transition-opacity disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900"
        >
          {loading ? "Analyzing…" : "Analyze gap"}
        </button>
        {report && (
          <button onClick={reset} className="text-sm text-zinc-500 underline underline-offset-2 hover:text-zinc-900 dark:hover:text-zinc-100">
            Start over
          </button>
        )}
      </div>

      {error && (
        <p className="mt-4 rounded-xl border border-rose-300 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-300">
          {error}
        </p>
      )}

      {report && <ResultsView report={report} />}
      {plan && <DayPlanView plan={plan} onNextDay={nextDay} busy={loading} />}
    </main>
  );
}
