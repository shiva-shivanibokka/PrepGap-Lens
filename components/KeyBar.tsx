import { PROVIDERS, PROVIDER_IDS, type ProviderId } from "@/lib/providers";

export interface KeyState {
  provider: ProviderId;
  apiKey: string;
  model: string;
}

export function KeyBar({ value, onChange }: { value: KeyState; onChange: (patch: Partial<KeyState>) => void }) {
  const meta = PROVIDERS[value.provider];
  return (
    <div className="rounded-xl border border-zinc-200 bg-zinc-50/60 p-4 dark:border-zinc-800 dark:bg-zinc-900/40">
      <div className="flex items-center justify-between">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-zinc-500">Your key, your call</p>
        <a
          href={meta.keysUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="font-mono text-xs text-zinc-500 underline decoration-dotted underline-offset-2 hover:text-zinc-900 dark:hover:text-zinc-100"
        >
          Get a key ↗
        </a>
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-[minmax(0,10rem)_1fr_minmax(0,12rem)]">
        <label className="grid gap-1">
          <span className="font-mono text-[0.7rem] uppercase tracking-wider text-zinc-500">Provider</span>
          <select
            value={value.provider}
            onChange={(e) => onChange({ provider: e.target.value as ProviderId })}
            className="h-10 rounded-lg border border-zinc-300 bg-white px-3 text-sm dark:border-zinc-700 dark:bg-zinc-950"
          >
            {PROVIDER_IDS.map((id) => (
              <option key={id} value={id}>
                {PROVIDERS[id].label}
              </option>
            ))}
          </select>
        </label>

        <label className="grid gap-1">
          <span className="font-mono text-[0.7rem] uppercase tracking-wider text-zinc-500">API key</span>
          <input
            type="password"
            value={value.apiKey}
            autoComplete="off"
            spellCheck={false}
            onChange={(e) => onChange({ apiKey: e.target.value })}
            placeholder="sk-… (kept in this tab only)"
            className="h-10 rounded-lg border border-zinc-300 bg-white px-3 font-mono text-sm dark:border-zinc-700 dark:bg-zinc-950"
          />
        </label>

        <label className="grid gap-1">
          <span className="font-mono text-[0.7rem] uppercase tracking-wider text-zinc-500">Model (optional)</span>
          <input
            type="text"
            value={value.model}
            spellCheck={false}
            onChange={(e) => onChange({ model: e.target.value })}
            placeholder={meta.defaultModel}
            className="h-10 rounded-lg border border-zinc-300 bg-white px-3 font-mono text-sm dark:border-zinc-700 dark:bg-zinc-950"
          />
        </label>
      </div>

      <p className="mt-3 text-xs text-zinc-500">
        Your key stays in this browser tab, is sent over HTTPS for a single request, and is never stored or logged. Refreshing the page clears it.
      </p>
    </div>
  );
}
