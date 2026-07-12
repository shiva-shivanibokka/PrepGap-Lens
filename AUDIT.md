# Repo Audit Report — PrepGap-Lens

**Date:** 2026-07-12
**Stack detected:** Next.js 16 (App Router) · TypeScript · React 19 · Vercel AI SDK · Vitest. Node serverless runtime for API routes.
**Scope:** All of `lib/`, `app/`, `components/`. Excluded: generated (`.next/`), `node_modules/`, scaffolding docs (`AGENTS.md`, `CLAUDE.md`), and the plan under `docs/`.

## Summary

- Total findings: 8
- Auto-fixed (trivial-safe): 2
- Needs review (see PLAN.md): 6
- Critical: 0 | Major: 2 | Minor: 4 | Notes: 2

## Production-readiness scorecard

| Category | Status | Notes |
|---|---|---|
| Correctness | ⚠️ | One UX-correctness bug: missing prerequisites render as raw skill IDs. |
| Silent failures | ⚠️ | API routes catch-and-return generic messages with no server-side logging. |
| Security | ❌ | `fetch-jd` fetches an arbitrary user URL (SSRF) and reads the response unbounded. |
| Concurrency | ✅ | Stateless serverless; no shared mutable state. localStorage is single-user. |
| Performance | ⚠️ | Unbounded response read in `fetch-jd`; otherwise no N+1 / hot-path issues. |
| Architecture | ✅ | Clean layering: pure `lib/`, thin routes, presentational components. |
| Production-readiness | ⚠️ | No logging on failure paths; no outbound-fetch timeout. |
| Test coverage | ⚠️ | Core logic (score/planner/analyze) tested; `htmlToText` and route logic untested. |

## Auto-fixed (trivial-safe)

- **`lib/credibility.ts` (whole file)** — deleted. Zero importers anywhere in the repo (verified via grep); the planner has its own inline credibility ranking in `chooseResource`. Dead code.
- **`lib/planner.ts:` `chooseResource`** — removed the unused `export`. It's only called inside `planner.ts`; no external caller or test references it.

Both verified: `npx vitest run` (6/6 pass) and `npm run build` succeed after removal.

## Findings requiring review

### Security (pass 8)

**F1 — `app/api/fetch-jd/route.ts` · Server-Side Request Forgery (SSRF) · Major**
The route does `fetch(new URL(userInput))` against any URL the client sends, with `redirect: "follow"`. A user can point it at `http://localhost:*`, private ranges (`10.x`, `192.168.x`, `169.254.169.254` cloud-metadata), or use your deployment as an anonymizing proxy. On Vercel the blast radius is smaller (no internal network to pivot into), but it's still an open fetch proxy and a real class of vulnerability worth closing on a portfolio piece reviewers will read.
*Fix:* resolve the hostname and reject non-public IPs before fetching; also cap redirects. Full code in PLAN.md Task 1.

**F2 — `app/api/fetch-jd/route.ts:38` · Unbounded response read · Major**
`await res.text()` buffers the entire response into memory with no cap. A link that returns a multi-GB body (or a slow drip) can OOM or hang the function up to its 20s limit. 
*Fix:* enforce a byte cap while reading and reject oversized bodies. PLAN.md Task 2.

### Correctness (pass 3 / 6)

**F3 — `components/ResultsView.tsx:` gap `needs first:` line · Minor**
`g.prereqsMissing.join(", ")` renders raw skill **IDs** (`model_evaluation`, snake_case) instead of human names, because `Gap.prereqsMissing` carries IDs. Users see machine identifiers in the one place the app is supposed to be most legible. 
*Fix:* resolve IDs → names (map built from `skills`). PLAN.md Task 4.

### Production-readiness (pass 12)

**F4 — `app/api/fetch-jd/route.ts`, `app/api/parse-resume/route.ts` · No outbound timeout · Minor**
The `fetch-jd` outbound `fetch` has no timeout; a slow host ties the function up to `maxDuration`. 
*Fix:* `AbortSignal.timeout(10000)`. PLAN.md Task 3.

**F5 — both parsing routes · Errors swallowed without logging · Minor / Note**
`catch { return NextResponse.json({ error: "…generic…" }) }` returns a friendly message (good) but logs nothing server-side, so real parse/fetch failures are invisible in Vercel logs when debugging. 
*Fix:* `console.error` the real error (server-only; no secrets involved here). PLAN.md Task 5.

### Test coverage (pass 13)

**F6 — `app/api/fetch-jd/route.ts:` `htmlToText` · Minor**
`htmlToText` is pure, regex-heavy, and untested — exactly the kind of logic that silently rots (entity decoding, tag stripping, whitespace collapsing). It's currently inlined in the route, so it isn't importable. 
*Fix:* extract to `lib/html.ts` and add a Vitest. PLAN.md Task 6 (paired with Task 1/2 since they touch the same file).

## Notes (not bugs)

- **N1 — `app/api/analyze/route.ts`** returns the provider's raw `err.message` to the client. Acceptable and even helpful for BYOK (users debug their own key/model), but be aware it surfaces provider-internal text. No change recommended.
- **N2 — Paste-mode inputs are unbounded.** A user can paste very large JD/resume text straight to the LLM. Cost is on their own key (BYOK), so this is informational, not a fix.

## Clean areas

- `lib/score.ts`, `lib/planner.ts`, `lib/types.ts`, `lib/analyze.ts` (schema + `splitCoverage`), `lib/progress.ts` — logic is correct, pure, and (for the first three) unit-tested. No control-flow, boundary, or contract bugs found.
- `app/api/analyze/route.ts` — input validation present, BYOK key never logged, errors handled. Clean.
- Concurrency & architecture — no findings.
