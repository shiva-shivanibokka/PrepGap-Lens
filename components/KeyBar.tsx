import { PROVIDERS, PROVIDER_IDS, type ProviderId } from "@/lib/providers";

export interface KeyState {
  provider: ProviderId;
  apiKey: string;
  model: string;
}

export function KeyBar({ value, onChange }: { value: KeyState; onChange: (patch: Partial<KeyState>) => void }) {
  const meta = PROVIDERS[value.provider];
  return (
    <div>
      <div className="keybar-head">
        <span className="section-label" style={{ margin: 0 }}>
          Your key, your call
        </span>
        <a href={meta.keysUrl} target="_blank" rel="noopener noreferrer" className="chip">
          Get a key ↗
        </a>
      </div>

      <div className="keybar">
        <div className="field">
          <label htmlFor="provider">Provider</label>
          <select id="provider" value={value.provider} onChange={(e) => onChange({ provider: e.target.value as ProviderId })}>
            {PROVIDER_IDS.map((id) => (
              <option key={id} value={id}>
                {PROVIDERS[id].label}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="apiKey">API key</label>
          <input
            id="apiKey"
            className="mono"
            type="password"
            value={value.apiKey}
            autoComplete="off"
            spellCheck={false}
            onChange={(e) => onChange({ apiKey: e.target.value })}
            placeholder="kept in this tab only"
          />
        </div>

        <div className="field">
          <label htmlFor="model">Model (optional)</label>
          <input
            id="model"
            className="mono"
            type="text"
            value={value.model}
            spellCheck={false}
            onChange={(e) => onChange({ model: e.target.value })}
            placeholder={meta.defaultModel}
          />
        </div>
      </div>

      <p className="keybar-note">
        Your key stays in this browser tab, is sent over HTTPS for a single request, and is never stored or logged.
        Refreshing the page clears it.
      </p>
    </div>
  );
}
