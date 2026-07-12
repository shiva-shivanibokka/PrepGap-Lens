import { NextResponse } from "next/server";
import { runAnalyze } from "@/lib/analyze";
import { scoreReadiness } from "@/lib/score";
import { PROVIDER_IDS, type ProviderId } from "@/lib/providers";

export const runtime = "nodejs";
export const maxDuration = 30;

export async function POST(req: Request) {
  // BYOK: never log the body — it carries the user's API key.
  try {
    const { jd, resume, provider, apiKey, model } = await req.json();
    if (!jd?.trim() || !resume?.trim())
      return NextResponse.json({ error: "Provide both a job description and a resume." }, { status: 400 });
    if (!apiKey?.trim()) return NextResponse.json({ error: "Enter your provider API key." }, { status: 400 });
    if (!PROVIDER_IDS.includes(provider)) return NextResponse.json({ error: "Pick a provider." }, { status: 400 });

    const { skills, coverage } = await runAnalyze({ jd, resume, provider: provider as ProviderId, apiKey, model });
    const report = scoreReadiness(skills, coverage, { topKGaps: 6 });
    return NextResponse.json({ skills, report });
  } catch (err) {
    // provider auth errors, bad model id, or schema/parse failures land here
    const msg = err instanceof Error ? err.message : "Analysis failed";
    return NextResponse.json({ error: `Could not analyze: ${msg}` }, { status: 502 });
  }
}
