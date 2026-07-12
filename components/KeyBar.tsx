import { PROVIDERS, PROVIDER_IDS, type ProviderId } from "@/lib/providers";
import { Tip } from "./Tip";

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

      {/* provider + model side by side */}
      <div className="keybar-top">
        <div className="field">
          <span className="lbl head-tip">
            Provider
            <Tip text="Which AI company's model runs the analysis. Google and Groq have free tiers." />
          </span>
          <select value={value.provider} onChange={(e) => onChange({ provider: e.target.value as ProviderId })}>
            {PROVIDER_IDS.map((id) => (
              <option key={id} value={id}>
                {PROVIDERS[id].label}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <span className="lbl head-tip">
            Model
            <Tip text="The specific model to use. Smaller models (flash, mini, haiku) are cheaper and faster; larger ones are more accurate." />
          </span>
          <select value={value.model} onChange={(e) => onChange({ model: e.target.value })}>
            {meta.models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* api key last, full width */}
      <div className="field" style={{ marginTop: "1rem" }}>
        <span className="lbl head-tip">
          API key
          <Tip text="Your own key for the chosen provider. It's used for a single request and never stored or logged; refreshing the page clears it." />
        </span>
        <input
          className="mono"
          type="password"
          value={value.apiKey}
          autoComplete="off"
          spellCheck={false}
          onChange={(e) => onChange({ apiKey: e.target.value })}
          placeholder="kept in this tab only"
        />
      </div>

      <p className="keybar-note">
        Your key stays in this browser tab, is sent over HTTPS for a single request, and is never stored or logged.
        Refreshing the page clears it.
      </p>
    </div>
  );
}
