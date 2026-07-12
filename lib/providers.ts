export type ProviderId = "google" | "openai" | "groq" | "anthropic";

export interface ProviderMeta {
  label: string;
  models: string[]; // first entry is the default
  keysUrl: string;
}

export const PROVIDERS: Record<ProviderId, ProviderMeta> = {
  google: {
    label: "Google (Gemini)",
    models: ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"],
    keysUrl: "https://aistudio.google.com/app/apikey",
  },
  openai: {
    label: "OpenAI",
    models: ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
    keysUrl: "https://platform.openai.com/api-keys",
  },
  groq: {
    label: "Groq",
    models: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
    keysUrl: "https://console.groq.com/keys",
  },
  anthropic: {
    label: "Anthropic",
    models: ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest"],
    keysUrl: "https://console.anthropic.com/settings/keys",
  },
};

export const PROVIDER_IDS = Object.keys(PROVIDERS) as ProviderId[];

export const defaultModel = (p: ProviderId) => PROVIDERS[p].models[0];

// BYOK: the user's key comes in per call and is only used to build the model.
// Dynamic import so only the chosen provider's SDK loads.
export async function getModel(provider: ProviderId, apiKey: string, model?: string) {
  const id = model?.trim() || defaultModel(provider);
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
