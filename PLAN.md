# Fix Plan — PrepGap-Lens

Generated from repo-bug-audit on 2026-07-12. 6 tasks, ordered so the shared-file security fixes land together and the pure helper is extracted+tested first.

All tasks touch `app/api/fetch-jd/route.ts` except Task 5. Apply 1→4 in order (same file). Task 5 and 6 are independent.

---

## Task 1: Extract `htmlToText` to a tested module

- **File:** `lib/html.ts` (new), `lib/__tests__/html.test.ts` (new), `app/api/fetch-jd/route.ts` (edit import)
- **Category:** Test coverage (pass 13)
- **Severity:** Minor
- **Finding:** `htmlToText` is inlined in the route, regex-heavy, and untested.
- **Why it matters:** Entity-decode / tag-strip logic silently degrades JD quality if a regex breaks; no test would catch it.
- **Proposed change:** Move the function verbatim into `lib/html.ts` and export it:
  ```ts
  // lib/html.ts
  export function htmlToText(html: string): string {
    return html
      .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
      .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
      .replace(/<\/(p|div|li|h[1-6]|br|tr|section)>/gi, "\n")
      .replace(/<[^>]+>/g, " ")
      .replace(/&nbsp;/gi, " ").replace(/&amp;/gi, "&").replace(/&lt;/gi, "<").replace(/&gt;/gi, ">")
      .replace(/&#39;|&rsquo;|&apos;/gi, "'").replace(/&quot;|&ldquo;|&rdquo;/gi, '"')
      .replace(/[ \t]+/g, " ").replace(/\n\s*\n\s*\n+/g, "\n\n").trim();
  }
  ```
  Then in the route: `import { htmlToText } from "@/lib/html";` and delete the inline copy.
  ```ts
  // lib/__tests__/html.test.ts
  import { describe, it, expect } from "vitest";
  import { htmlToText } from "../html";
  describe("htmlToText", () => {
    it("strips scripts/styles and tags, decodes entities, keeps block breaks", () => {
      const out = htmlToText(`<style>x{}</style><h1>Senior</h1><p>Python &amp; SQL</p><script>bad()</script>`);
      expect(out).toContain("Senior");
      expect(out).toContain("Python & SQL");
      expect(out).not.toMatch(/bad\(\)|x\{\}|</);
    });
  });
  ```
- **Verification:** `npx vitest run html` → PASS; `npm run build` succeeds.

## Task 2: Block SSRF — reject private/loopback/link-local hosts before fetching

- **File:** `app/api/fetch-jd/route.ts`
- **Category:** Security (pass 8)
- **Severity:** Major
- **Finding:** `fetch(new URL(userInput))` reaches any address, including `localhost`, RFC-1918 ranges, and `169.254.169.254`.
- **Why it matters:** Open fetch proxy / SSRF — a reviewer running the code could pivot to internal or metadata endpoints; at minimum it's an anonymizing proxy on your deployment.
- **Proposed change:** Resolve the host and validate every resolved IP is public before fetching.
  ```ts
  import { lookup } from "node:dns/promises";
  import { isIP } from "node:net";

  function isPrivateIp(ip: string): boolean {
    if (ip.startsWith("127.") || ip === "::1" || ip.startsWith("::ffff:127.")) return true;
    if (ip.startsWith("10.") || ip.startsWith("192.168.")) return true;
    if (ip.startsWith("169.254.") || ip.startsWith("fe80:")) return true; // link-local + cloud metadata
    if (/^172\.(1[6-9]|2\d|3[01])\./.test(ip)) return true; // 172.16.0.0/12
    if (ip === "0.0.0.0" || ip.startsWith("fc") || ip.startsWith("fd")) return true; // unique-local IPv6
    return false;
  }

  async function assertPublicHost(host: string): Promise<void> {
    const literal = isIP(host);
    const ips = literal ? [host] : (await lookup(host, { all: true })).map((a) => a.address);
    if (!ips.length || ips.some(isPrivateIp)) throw new Error("blocked host");
  }
  ```
  In `POST`, after validating protocol and before `fetch`:
  ```ts
  try {
    await assertPublicHost(target.hostname);
  } catch {
    return NextResponse.json({ error: "That address isn't allowed. Paste the description instead." }, { status: 400 });
  }
  ```
- **Verification:** `curl -s -X POST localhost:3000/api/fetch-jd -H 'content-type: application/json' -d '{"url":"http://169.254.169.254/"}'` → 400 "isn't allowed"; a normal public careers URL still returns text.
- **Depends on:** Task 1 (same file; apply after).

## Task 3: Cap the response body size while reading

- **File:** `app/api/fetch-jd/route.ts`
- **Category:** Security / Performance (pass 8/10)
- **Severity:** Major
- **Finding:** `await res.text()` buffers an unbounded response.
- **Why it matters:** A large or slow-drip body can OOM the function or hold it open to `maxDuration`.
- **Proposed change:** Reject on `content-length` when present, and stream-read with a hard byte cap otherwise.
  ```ts
  const MAX_BYTES = 2_000_000; // 2 MB is plenty for a job page
  const len = Number(res.headers.get("content-length") ?? 0);
  if (len > MAX_BYTES) return NextResponse.json({ error: "That page is too large to read." }, { status: 413 });

  const reader = res.body?.getReader();
  if (!reader) return NextResponse.json({ error: "Couldn't read that link." }, { status: 502 });
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.length;
    if (total > MAX_BYTES) { await reader.cancel(); return NextResponse.json({ error: "That page is too large to read." }, { status: 413 }); }
    chunks.push(value);
  }
  const html = new TextDecoder().decode(Buffer.concat(chunks));
  const text = htmlToText(html);
  ```
  (Replaces the current `const text = htmlToText(await res.text());` line.)
- **Verification:** normal URL still returns text; build passes. Optionally test against a known-large asset URL → 413.
- **Depends on:** Task 2.

## Task 4: Add an outbound-fetch timeout

- **File:** `app/api/fetch-jd/route.ts`
- **Category:** Production-readiness (pass 12)
- **Severity:** Minor
- **Finding:** outbound `fetch` has no timeout.
- **Why it matters:** a slow host ties the function up to its 20s ceiling.
- **Proposed change:** add `signal: AbortSignal.timeout(10_000)` to the `fetch` options; in the outer `catch`, return the existing generic "couldn't fetch" message (an `AbortError` lands there naturally).
  ```ts
  const res = await fetch(target, {
    headers: { "user-agent": "Mozilla/5.0 (compatible; PrepGapLens/1.0)", accept: "text/html" },
    redirect: "follow",
    signal: AbortSignal.timeout(10_000),
  });
  ```
- **Verification:** `npm run build`; normal fetch still works.
- **Depends on:** Task 3.

## Task 5: Show prerequisite skill names, not raw IDs

- **File:** `components/ResultsView.tsx`
- **Category:** Correctness / UX (pass 3/6)
- **Severity:** Minor
- **Finding:** `g.prereqsMissing.join(", ")` prints snake_case skill IDs.
- **Why it matters:** the app's most legible surface shows machine identifiers (`model_evaluation`) instead of names.
- **Proposed change:** pass `skills` into `ResultsView` and resolve IDs → names.
  - In `app/page.tsx`, change `{report && <ResultsView report={report} />}` to `{report && skills && <ResultsView report={report} skills={skills} />}`.
  - In `components/ResultsView.tsx`:
  ```tsx
  import type { ScoreReport, Skill } from "@/lib/types";
  export function ResultsView({ report, skills }: { report: ScoreReport; skills: Skill[] }) {
    const nameById = new Map(skills.map((s) => [s.id, s.name] as const));
    // ...
    // in the gap row:
    {g.prereqsMissing.length > 0 && (
      <span className="gprereq">needs first: {g.prereqsMissing.map((id) => nameById.get(id) ?? id).join(", ")}</span>
    )}
  ```
- **Verification:** run the app, analyze a JD whose skills have prereqs, confirm the "needs first" line shows names.

## Task 6: Log the real error on parse/fetch failure

- **File:** `app/api/fetch-jd/route.ts`, `app/api/parse-resume/route.ts`
- **Category:** Production-readiness (pass 12)
- **Severity:** Minor
- **Finding:** `catch { return NextResponse.json({ error: "generic" }) }` logs nothing.
- **Why it matters:** real failures are invisible in Vercel logs when a user reports "it won't read my file/link." No secrets are involved in these two routes, so logging the error is safe.
- **Proposed change:** in each route's outer `catch (err)`, add `console.error("[fetch-jd] ", err);` / `console.error("[parse-resume] ", err);` before returning. (Do **not** add this to `analyze/route.ts` — that request body carries the API key.)
- **Verification:** `npm run build`; trigger a bad file/link locally and confirm the error prints to the dev server console.
