import { describe, it, expect } from "vitest";
import { htmlToText } from "../html";

describe("htmlToText", () => {
  it("strips scripts/styles and tags, decodes entities, keeps block breaks", () => {
    const out = htmlToText(`<style>x{color:red}</style><h1>Senior Engineer</h1><p>Python &amp; SQL</p><script>bad()</script>`);
    expect(out).toContain("Senior Engineer");
    expect(out).toContain("Python & SQL");
    expect(out).not.toMatch(/bad\(\)|color:red|</);
  });

  it("collapses whitespace and trims", () => {
    expect(htmlToText("<p>  a   b  </p>\n\n\n\n<p>c</p>")).toBe("a b\n\nc");
  });
});
