import { generateObject } from "ai";
import { z } from "zod";
import type { Skill } from "./types";
import { getModel, type ProviderId } from "./providers";

export const AnalyzeSchema = z.object({
  skills: z
    .array(
      z.object({
        id: z.string(),
        name: z.string(),
        category: z.string(),
        weight: z.number().int().min(1).max(5),
        keywords: z.array(z.string()).default([]),
        prereqs: z.array(z.string()).default([]),
        assessment: z.array(z.string()).default([]),
        coverage: z.number().min(0).max(1), // how well the resume covers this skill
        resources: z
          .array(
            z.object({
              title: z.string(),
              type: z.string(),
              credibility: z.enum(["interview-safe", "good-intuition", "misleading"]),
              reason: z.string(),
              url: z.string(),
            }),
          )
          .default([]),
      }),
    )
    .min(1),
});
export type AnalyzeParsed = z.infer<typeof AnalyzeSchema>;

// pure + tested: drop per-skill coverage out of the Skill shape into its own map
export function splitCoverage(parsed: AnalyzeParsed): { skills: Skill[]; coverage: Record<string, number> } {
  const skills: Skill[] = parsed.skills.map(({ coverage, ...s }) => s);
  const coverage: Record<string, number> = {};
  for (const s of parsed.skills) coverage[s.id] = s.coverage;
  return { skills, coverage };
}

export function buildPrompt(jd: string, resume: string): string {
  return `You are an interview-prep analyst. Read the JOB DESCRIPTION and the CANDIDATE RESUME, then produce a structured skill gap analysis.
Rules: output 8-15 skills the JOB actually requires. weight = importance to THIS job (5=core, 1=nice-to-have). prereqs must only reference ids you also output. coverage = how convincingly the RESUME demonstrates that skill (0=absent, 1=clearly proven). assessment = one likely interview question per skill. Suggest 1-2 real, well-known resources per skill; mark anything you are unsure about as "good-intuition", and never invent a specific URL you are not confident exists.

JOB DESCRIPTION:
${jd}

RESUME:
${resume}`;
}

// ponytail: LLM-suggested resource URLs can still be wrong. Acceptable for MVP;
// upgrade path = a curated resource table keyed by skill id, LLM only picks from it.
export async function runAnalyze(input: {
  jd: string;
  resume: string;
  provider: ProviderId;
  apiKey: string;
  model?: string;
}): Promise<{ skills: Skill[]; coverage: Record<string, number> }> {
  const model = await getModel(input.provider, input.apiKey, input.model);
  const { object } = await generateObject({
    model,
    schema: AnalyzeSchema,
    prompt: buildPrompt(input.jd, input.resume),
    // Many Groq models (e.g. llama-3.3-70b) reject the `json_schema` response format
    // generateObject uses by default. Disabling structured outputs makes the SDK fall
    // back to tool-calling, which those models do support.
    ...(input.provider === "groq" ? { providerOptions: { groq: { structuredOutputs: false } } } : {}),
  });
  return splitCoverage(object);
}
