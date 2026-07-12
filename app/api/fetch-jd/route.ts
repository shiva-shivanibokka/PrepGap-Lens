import { NextResponse } from "next/server";
import { lookup } from "node:dns/promises";
import { isIP } from "node:net";
import { htmlToText } from "@/lib/html";
import { isPrivateIp } from "@/lib/net";

export const runtime = "nodejs";
export const maxDuration = 20;

const MAX_BYTES = 2_000_000; // 2 MB is plenty for a job page

async function assertPublicHost(host: string): Promise<void> {
  const ips = isIP(host) ? [host] : (await lookup(host, { all: true })).map((a) => a.address);
  if (!ips.length || ips.some(isPrivateIp)) throw new Error("blocked host");
}

// Follow redirects manually so every hop's host is re-validated *before* the request
// fires — otherwise a public URL could 302 to a private/metadata address (SSRF bypass).
async function safeFetch(start: URL, maxHops = 3): Promise<Response> {
  let url = start;
  for (let hop = 0; hop <= maxHops; hop++) {
    await assertPublicHost(url.hostname);
    const res = await fetch(url, {
      headers: { "user-agent": "Mozilla/5.0 (compatible; PrepGapLens/1.0)", accept: "text/html" },
      redirect: "manual",
      signal: AbortSignal.timeout(10_000),
    });
    const location = res.status >= 300 && res.status < 400 ? res.headers.get("location") : null;
    if (!location) return res;
    url = new URL(location, url); // resolve relative redirects against the current URL
    if (!/^https?:$/.test(url.protocol)) throw new Error("blocked redirect scheme");
  }
  throw new Error("too many redirects");
}

async function readCapped(res: Response): Promise<string> {
  const declared = Number(res.headers.get("content-length") ?? 0);
  if (declared > MAX_BYTES) throw new Error("too large");
  const reader = res.body?.getReader();
  if (!reader) throw new Error("no body");
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.length;
    if (total > MAX_BYTES) {
      await reader.cancel();
      throw new Error("too large");
    }
    chunks.push(value);
  }
  return new TextDecoder().decode(Buffer.concat(chunks));
}

export async function POST(req: Request) {
  try {
    const { url } = await req.json();
    if (!url?.trim()) return NextResponse.json({ error: "Enter a URL." }, { status: 400 });

    let target: URL;
    try {
      target = new URL(url.trim());
    } catch {
      return NextResponse.json({ error: "That doesn't look like a valid URL." }, { status: 400 });
    }
    if (!/^https?:$/.test(target.protocol))
      return NextResponse.json({ error: "Only http(s) links are supported." }, { status: 400 });

    let res: Response;
    try {
      res = await safeFetch(target);
    } catch (e) {
      if (e instanceof Error && (e.message === "blocked host" || e.message.startsWith("blocked")))
        return NextResponse.json({ error: "That address isn't allowed. Paste the description instead." }, { status: 400 });
      throw e; // timeouts / DNS / too-many-redirects → outer catch (logged, generic message)
    }
    if (!res.ok) return NextResponse.json({ error: `The page returned ${res.status}.` }, { status: 502 });

    let text: string;
    try {
      text = htmlToText(await readCapped(res));
    } catch (e) {
      if (e instanceof Error && e.message === "too large")
        return NextResponse.json({ error: "That page is too large to read. Paste it instead." }, { status: 413 });
      throw e; // other read errors → outer catch
    }

    if (text.length < 80)
      return NextResponse.json(
        { error: "Couldn't read enough text from that link (it may need a login or JavaScript). Paste it instead." },
        { status: 422 },
      );

    return NextResponse.json({ text: text.slice(0, 20000) });
  } catch (err) {
    console.error("[fetch-jd]", err);
    return NextResponse.json({ error: "Couldn't fetch that link. Paste the description instead." }, { status: 502 });
  }
}
