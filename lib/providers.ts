export type ProviderId = "google" | "openai" | "groq" | "anthropic";

export const PROVIDERS: Record<ProviderId, { label: string; defaultModel: string; keysUrl: string }> = {
  google: { label: "Google (Gemini)", defaultModel: "gemini-2.0-flash", keysUrl: "https://aistudio.google.com/app/apikey" },
  openai: { label: "OpenAI", defaultModel: "gpt-4o-mini", keysUrl: "https://platform.openai.com/api-keys" },
  groq: { label: "Groq", defaultModel: "llama-3.3-70b-versatile", keysUrl: "https://console.groq.com/keys" },
  anthropic: { label: "Anthropic", defaultModel: "claude-3-5-haiku-latest", keysUrl: "https://console.anthropic.com/settings/keys" },
};

export const PROVIDER_IDS = Object.keys(PROVIDERS) as ProviderId[];

// BYOK: the user's key comes in per call and is only used to build the model.
// Dynamic import so only the chosen provider's SDK loads.
export async function getModel(provider: ProviderId, apiKey: string, model?: string) {
  const id = model?.trim() || PROVIDERS[provider].defaultModel;
  switch (provider) {
    case "google": {
      const { createGoogleGenerativeAI } = await import("@ai-sdk/google");
      return createGoogleGenerativeAI({ apiKey })(id);
    }
    case "openai": {
      const { createOpenAI } = await import("@ai-sdk/openai");
      return createOpenAI({ apiKey })(id);
    }
    case "groq": {
      const { createGroq } = await import("@ai-sdk/groq");
      return createGroq({ apiKey })(id);
    }
    case "anthropic": {
      const { createAnthropic } = await import("@ai-sdk/anthropic");
      return createAnthropic({ apiKey })(id);
    }
    default:
      throw new Error(`Unknown provider: ${provider}`);
  }
}
