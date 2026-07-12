"use client";

import { useEffect, useRef, useState } from "react";
import { KeyBar, type KeyState } from "@/components/KeyBar";
import { ResultsView } from "@/components/ResultsView";
import { DayPlanView } from "@/components/DayPlanView";
import { Tip } from "@/components/Tip";
import { defaultModel } from "@/lib/providers";
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
  const [key, setKey] = useState<KeyState>({ provider: "google", apiKey: "", model: defaultModel("google") });

  const [jdMode, setJdMode] = useState<JdMode>("paste");
  const [jd, setJd] = useState("");
  const [jdUrl, setJdUrl] = useState("");
  const [jdLoaded, setJdLoaded] = useState(false);
  const [fetchingJd, setFetchingJd] = useState(false);

  const [resumeMode, setResumeMode] = useState<ResumeMode>("paste");
  const [resume, setResume] = useState("");
  const [resumeFile, setResumeFile] = useState<string | null>(null);
  const [resumeLoaded, setResumeLoaded] = useState(false);
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

  function analyze() {
    // Validate on click (not by disabling the button) so the user always sees *why* it won't run.
    if (!key.apiKey.trim()) return setError("Add your provider API key above.");
    if (!jd.trim())
      return setError(
        jdMode === "link"
          ? "No job description loaded yet — click Fetch, or switch to Paste."
          : "Add the job description — paste it, or use the From-link option.",
      );
    if (!resume.trim())
      return setError(
        resumeMode === "upload"
          ? "No resume loaded yet — choose a file, or switch to Paste."
          : "Add your resume — paste it, or upload a PDF/Word file.",
      );
    void runAnalyze();
  }

  function updateKey(patch: Partial<KeyState>) {
    setKey((k) => {
      const next = { ...k, ...patch };
      // switching provider resets the model to that provider's default
      if (patch.provider) next.model = defaultModel(patch.provider);
      return next;
    });
  }

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
      if (!res.ok) {
        // couldn't read the link — fall back to paste so the user can drop it in
        setJd("");
        setJdLoaded(false);
        setJdMode("paste");
        setError(data.error ?? "Couldn't fetch that link — please paste the job description instead.");
      } else {
        setJd(data.text);
        setJdLoaded(true);
      }
    } catch {
      setJdMode("paste");
      setError("Couldn't reach that link — please paste the job description instead.");
    } finally {
      setFetchingJd(false);
    }
  }

  async function onFile(file: File) {
    setParsingResume(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch("/api/parse-resume", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) {
        setResume("");
        setResumeFile(null);
        setResumeLoaded(false);
        setResumeMode("paste");
        setError(data.error ?? "Couldn't read that file — please paste your resume instead.");
      } else {
        setResume(data.text);
        setResumeFile(file.name);
        setResumeLoaded(true);
      }
    } catch {
      setResumeMode("paste");
      setError("Couldn't upload that file — please paste your resume instead.");
    } finally {
      setParsingResume(false);
    }
  }

  async function runAnalyze() {
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
          <div className="head-tip">
            <h2>Set up your analysis</h2>
            <Tip text="Choose a provider + model, add your API key, then give the job description and your resume." />
          </div>
          <span className="chip">Google · OpenAI · Groq · Anthropic</span>
        </div>

        <KeyBar value={key} onChange={updateKey} />

        {/* JD row */}
        <div className="io-row">
          <div className="io-head">
            <span className="section-label" style={{ margin: 0 }}>
              Job description
              <Tip text="The posting you're preparing for. Paste the text, or fetch it from a public link." />
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

          {jdMode === "paste" ? (
            <textarea value={jd} onChange={(e) => setJd(e.target.value)} rows={8} placeholder="Paste the full job posting…" />
          ) : (
            <>
              <div className="link-row">
                <input
                  type="url"
                  value={jdUrl}
                  onChange={(e) => {
                    setJdUrl(e.target.value);
                    setJdLoaded(false);
                  }}
                  placeholder="https://company.com/careers/the-role"
                />
                <button className="btn btn-ghost btn-sm" onClick={fetchJd} disabled={fetchingJd || !jdUrl.trim()}>
                  {fetchingJd ? "Fetching…" : "Fetch"}
                </button>
              </div>
              {jdLoaded && <p className="load-msg">✓ Job description loaded from the link.</p>}
            </>
          )}
        </div>

        {/* Resume row */}
        <div className="io-row">
          <div className="io-head">
            <span className="section-label" style={{ margin: 0 }}>
              Your resume
              <Tip text="Your resume or a skills summary. Paste it, or upload a PDF or Word (.docx) file." />
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

          {resumeMode === "paste" ? (
            <textarea
              value={resume}
              onChange={(e) => setResume(e.target.value)}
              rows={8}
              placeholder="Paste your resume or skills summary…"
            />
          ) : (
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
              {resumeLoaded && resumeFile && <p className="load-msg">✓ {resumeFile} read successfully.</p>}
            </>
          )}
        </div>

        <div className="row-actions">
          <button className="btn" onClick={analyze} disabled={loading}>
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

      {report && skills && <ResultsView report={report} skills={skills} />}
      {plan && <DayPlanView plan={plan} onNextDay={nextDay} />}

      <p className="footer">Built by Shivani Bokka · BYOK · deployed on Vercel</p>
    </main>
  );
}
