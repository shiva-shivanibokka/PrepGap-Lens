# PrepGap-Lens

**Paste a job description and your resume. See how ready you are, which gaps cost you the most, and get an adaptive day-by-day plan to close them.**

Most interview-prep tools hand you a generic checklist. PrepGap-Lens measures the gap between *your* resume and *this specific* job posting, then turns that gap into a concrete study schedule that adapts as you make progress.

🔗 **Live demo:** _add your Vercel URL here after deploying_

<!-- add a screenshot/GIF here: docs/screenshot.png -->

---

## How it works

1. **You bring your own key (BYOK).** Pick a provider — Google, OpenAI, Groq, or Anthropic — and paste your own API key. The key stays in your browser tab, is sent over HTTPS for a single request, and is **never stored or logged**. Refreshing the page clears it.
2. **The JD becomes a skill map.** One LLM call turns the pasted job description into a structured, weighted skill list and scores how well your resume covers each skill.
3. **Deterministic scoring.** Your overall readiness, per-category readiness, and ranked gaps are computed in plain, unit-tested TypeScript — no LLM guesswork in the numbers.
4. **An adaptive plan.** A day plan focuses on your highest-impact gaps, respects prerequisites, and balances learn / practice / explain / review blocks. Mark items done/skipped with a difficulty rating, and the next day re-weights: skipped topics come back as short retries, hard topics get reinforcement.

Progress is saved to `localStorage` (your API key never is).

## Bring your own key

| Provider | Where to get a key | Free tier |
| --- | --- | --- |
| Google (Gemini) | https://aistudio.google.com/app/apikey | Yes |
| Groq | https://console.groq.com/keys | Yes |
| OpenAI | https://platform.openai.com/api-keys | Paid |
| Anthropic | https://console.anthropic.com/settings/keys | Paid |

Google AI Studio and Groq both have free tiers, so you can run the whole app at no cost. You can override the default model per provider in the UI.

**Privacy:** the app itself holds zero secrets. Your key is accepted by the `/api/analyze` serverless route, used for exactly one provider call, and discarded — it is never written to a database, cache, cookie, or log.

## Run locally

```bash
npm install
npm run dev
```

Open http://localhost:3000. No `.env` or secrets to configure — it's BYOK, so you enter a key in the UI.

```bash
npm test        # unit tests for scoring, planner, and analyze parsing
npm run build   # production build + typecheck
```

## Deploy (Vercel free tier)

```bash
npx vercel          # link + preview
npx vercel --prod   # production
```

There are **no environment variables to set** — the app has no server-side keys.

## Design notes

- **The LLM boundary is deliberate.** The model does only the fuzzy work it's good at: reading a JD and a resume into structured data (`lib/analyze.ts`, validated with a Zod schema via the AI SDK's `generateObject`). Everything downstream — readiness math (`lib/score.ts`) and the adaptive planner (`lib/planner.ts`) — is pure, deterministic, and unit-tested, so the numbers are reproducible and explainable.
- **BYOK, four providers, one interface.** All providers are unified behind the Vercel AI SDK, so switching between Gemini / GPT / Llama-on-Groq / Claude is a dropdown, not four code paths.
- **The UI is a diagnostic readout.** Chrome is monochrome; the only color on the page encodes your data — the readiness meter runs rose → amber → emerald by score, and skill weight shows as filled dots.

## Tech stack

Next.js 16 (App Router) · TypeScript · Tailwind CSS · Vercel AI SDK (`ai` + `@ai-sdk/google` / `@ai-sdk/openai` / `@ai-sdk/groq` / `@ai-sdk/anthropic`) · Zod · Vitest · deployed on Vercel.
