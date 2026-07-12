import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 20;

// Strip HTML to readable text. ponytail: naive tag-strip — good enough for static job
// pages; JS-rendered / bot-blocked postings won't extract, so the UI keeps paste as fallback.
function htmlToText(html: string): string {
  return html
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
    .replace(/<\/(p|div|li|h[1-6]|br|tr|section)>/gi, "\n")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&#39;|&rsquo;|&apos;/gi, "'")
    .replace(/&quot;|&ldquo;|&rdquo;/gi, '"')
    .replace(/[ \t]+/g, " ")
    .replace(/\n\s*\n\s*\n+/g, "\n\n")
    .trim();
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

    const res = await fetch(target, {
      headers: { "user-agent": "Mozilla/5.0 (compatible; PrepGapLens/1.0)", accept: "text/html" },
      redirect: "follow",
    });
    if (!res.ok) return NextResponse.json({ error: `The page returned ${res.status}.` }, { status: 502 });

    const text = htmlToText(await res.text());
    if (text.length < 80)
      return NextResponse.json(
        { error: "Couldn't read enough text from that link (it may need a login or JavaScript). Paste it instead." },
        { status: 422 },
      );

    return NextResponse.json({ text: text.slice(0, 20000) });
  } catch {
    return NextResponse.json({ error: "Couldn't fetch that link. Paste the description instead." }, { status: 502 });
  }
}
