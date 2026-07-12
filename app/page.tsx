"use client";

import { useEffect, useRef, useState } from "react";
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

type JdMode = "paste" | "link";
type ResumeMode = "paste" | "upload";

export default function Home() {
  const [key, setKey] = useState<KeyState>({ provider: "google", apiKey: "", model: "" });

  const [jdMode, setJdMode] = useState<JdMode>("paste");
  const [jd, setJd] = useState("");
  const [jdUrl, setJdUrl] = useState("");
  const [fetchingJd, setFetchingJd] = useState(false);

  const [resumeMode, setResumeMode] = useState<ResumeMode>("paste");
  const [resume, setResume] = useState("");
  const [resumeFile, setResumeFile] = useState<string | null>(null);
  const [parsingResume, setParsingResume] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [report, setReport] = useState<ScoreReport | null>(null);
  const [skills, setSkills] = useState<Skill[] | null>(null);
  const [mastery, setMastery] = useState<Record<string, number>>({});
  const [day, setDay] = useState(1);
  const [plan, setPlan] = useState<DayPlan | null>(null);

  const [saved, setSaved] = useState<Progress | null>(null);
  useEffect(() => setSaved(loadProgress()), []);

  const canAnalyze = Boolean(key.apiKey.trim() && jd.trim() && resume.trim()) && !loading;

  async function fetchJd() {
    setFetchingJd(true);
    setError(null);
    try {
      const res = await fetch("/api/fetch-jd", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ url: jdUrl }),
      });
      const data = await res.json();
      if (!res.ok) setError(data.error ?? "Couldn't fetch that link.");
      else setJd(data.text);
    } catch {
      setError("Couldn't reach that link.");
    } finally {
      setFetchingJd(false);
    }
  }

  async function onFile(file: File) {
    setParsingResume(true);
    setError(null);
    setResumeFile(file.name);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch("/api/parse-resume", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error ?? "Couldn't parse that file.");
        setResumeFile(null);
      } else {
        setResume(data.text);
      }
    } catch {
      setError("Couldn't upload that file.");
      setResumeFile(null);
    } finally {
      setParsingResume(false);
    }
  }

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
      setSkills(s);
      setReport(r);
      setMastery(m);
      setDay(1);
      setPlan(generateDayPlan({ day: 1, skills: s, mastery: m }));
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
    setMastery(newMastery);
    setDay(nd);
    setPlan(generateDayPlan({ day: nd, skills, mastery: newMastery, reinforceSkills: reinforce, retrySkills: retry }));
    if (report) saveProgress({ skills, report, mastery: newMastery, day: nd });
  }

  function resumePlan() {
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
    <main className="wrap">
      <header className="hero">
        <h1>PrepGap-Lens</h1>
        <p>
          Paste a job description and your resume. See how ready you are, which gaps cost you the most, and get an
          adaptive day-by-day plan to close them — powered by your own API key.
        </p>
        <span className="live">
          <b>●</b> BYOK · your key never leaves this request
        </span>
      </header>

      {saved && !report && (
        <section className="panel">
          <div className="panel-head">
            <h2>Resume your plan</h2>
            <span className="chip">saved · day {saved.day}</span>
          </div>
          <div className="row-actions" style={{ marginTop: 0 }}>
            <button className="btn" onClick={resumePlan}>
              Resume day {saved.day}
            </button>
            <button className="btn btn-ghost" onClick={reset}>
              Discard
            </button>
          </div>
        </section>
      )}

      <section className="panel">
        <div className="panel-head">
          <h2>Set up your analysis</h2>
          <span className="chip">Google · OpenAI · Groq · Anthropic</span>
        </div>

        <KeyBar value={key} onChange={(patch) => setKey((k) => ({ ...k, ...patch }))} />

        {/* JD row */}
        <div className="io-row">
          <div className="io-head">
            <span className="section-label" style={{ margin: 0 }}>
              Job description
            </span>
            <div className="seg">
              <button aria-pressed={jdMode === "paste"} onClick={() => setJdMode("paste")}>
                Paste
              </button>
              <button aria-pressed={jdMode === "link"} onClick={() => setJdMode("link")}>
                From link
              </button>
            </div>
          </div>

          {jdMode === "link" && (
            <div className="link-row">
              <input
                type="url"
                value={jdUrl}
                onChange={(e) => setJdUrl(e.target.value)}
                placeholder="https://company.com/careers/the-role"
              />
              <button className="btn btn-ghost btn-sm" onClick={fetchJd} disabled={fetchingJd || !jdUrl.trim()}>
                {fetchingJd ? "Fetching…" : "Fetch"}
              </button>
            </div>
          )}

          <textarea
            value={jd}
            onChange={(e) => setJd(e.target.value)}
            rows={8}
            placeholder={jdMode === "link" ? "Fetched text appears here — edit if needed." : "Paste the full job posting…"}
          />
        </div>

        {/* Resume row */}
        <div className="io-row">
          <div className="io-head">
            <span className="section-label" style={{ margin: 0 }}>
              Your resume
            </span>
            <div className="seg">
              <button aria-pressed={resumeMode === "paste"} onClick={() => setResumeMode("paste")}>
                Paste
              </button>
              <button aria-pressed={resumeMode === "upload"} onClick={() => setResumeMode("upload")}>
                Upload
              </button>
            </div>
          </div>

          {resumeMode === "upload" && (
            <>
              <label className="dropzone" onClick={() => fileInput.current?.click()}>
                <span>
                  {parsingResume ? (
                    "Reading your file…"
                  ) : (
                    <>
                      <b>Choose a file</b> — PDF or Word (.docx)
                    </>
                  )}
                </span>
                <input
                  ref={fileInput}
                  type="file"
                  accept=".pdf,.docx"
                  hidden
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) onFile(f);
                  }}
                />
              </label>
              {resumeFile && <p className="filemeta">✓ {resumeFile} · {resume.length} chars extracted</p>}
            </>
          )}

          <textarea
            value={resume}
            onChange={(e) => setResume(e.target.value)}
            rows={8}
            placeholder={resumeMode === "upload" ? "Extracted text appears here — edit if needed." : "Paste your resume or skills summary…"}
          />
        </div>

        <div className="row-actions">
          <button className="btn" onClick={analyze} disabled={!canAnalyze}>
            {loading ? "Analyzing…" : "Analyze gap"}
          </button>
          {report && (
            <button className="btn btn-ghost" onClick={reset}>
              Start over
            </button>
          )}
        </div>

        {error && <p className="error">{error}</p>}
      </section>

      {report && <ResultsView report={report} />}
      {plan && <DayPlanView plan={plan} onNextDay={nextDay} />}

      <p className="footer">Built by Shivani Bokka · BYOK · deployed on Vercel</p>
    </main>
  );
}
