import { NextResponse } from "next/server";
import { lookup } from "node:dns/promises";
import { isIP } from "node:net";
import { htmlToText } from "@/lib/html";

export const runtime = "nodejs";
export const maxDuration = 20;

const MAX_BYTES = 2_000_000; // 2 MB is plenty for a job page

// Reject loopback / private / link-local / cloud-metadata addresses (SSRF guard).
function isPrivateIp(ip: string): boolean {
  const v = ip.replace(/^::ffff:/, "");
  if (v.startsWith("127.") || v === "::1") return true;
  if (v.startsWith("10.") || v.startsWith("192.168.")) return true;
  if (v.startsWith("169.254.") || v.toLowerCase().startsWith("fe80:")) return true; // link-local + metadata
  if (/^172\.(1[6-9]|2\d|3[01])\./.test(v)) return true; // 172.16.0.0/12
  if (v === "0.0.0.0" || /^f[cd][0-9a-f]{2}:/i.test(v)) return true; // unspecified + unique-local IPv6
  return false;
}

async function assertPublicHost(host: string): Promise<void> {
  const ips = isIP(host) ? [host] : (await lookup(host, { all: true })).map((a) => a.address);
  if (!ips.length || ips.some(isPrivateIp)) throw new Error("blocked host");
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

    try {
      await assertPublicHost(target.hostname);
    } catch {
      return NextResponse.json({ error: "That address isn't allowed. Paste the description instead." }, { status: 400 });
    }

    const res = await fetch(target, {
      headers: { "user-agent": "Mozilla/5.0 (compatible; PrepGapLens/1.0)", accept: "text/html" },
      redirect: "follow",
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) return NextResponse.json({ error: `The page returned ${res.status}.` }, { status: 502 });

    let text: string;
    try {
      text = htmlToText(await readCapped(res));
    } catch {
      return NextResponse.json({ error: "That page is too large to read. Paste it instead." }, { status: 413 });
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
