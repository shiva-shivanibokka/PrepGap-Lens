// A small accessible "?" help tooltip. Hover or keyboard-focus to reveal.
export function Tip({ text }: { text: string }) {
  return (
    <span className="tip" tabIndex={0} role="note" aria-label={text}>
      <span className="q" aria-hidden="true">?</span>
      <span className="pop">{text}</span>
    </span>
  );
}
