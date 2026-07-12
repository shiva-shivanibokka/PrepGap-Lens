import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 20;

export async function POST(req: Request) {
  try {
    const form = await req.formData();
    const file = form.get("file");
    if (!(file instanceof File)) return NextResponse.json({ error: "No file uploaded." }, { status: 400 });

    const name = file.name.toLowerCase();
    const buf = Buffer.from(await file.arrayBuffer());

    let text = "";
    if (name.endsWith(".pdf")) {
      const { extractText, getDocumentProxy } = await import("unpdf");
      const pdf = await getDocumentProxy(new Uint8Array(buf));
      const res = await extractText(pdf, { mergePages: true });
      text = Array.isArray(res.text) ? res.text.join("\n") : res.text;
    } else if (name.endsWith(".docx")) {
      const mammoth = (await import("mammoth")).default;
      const res = await mammoth.extractRawText({ buffer: buf });
      text = res.value;
    } else {
      return NextResponse.json({ error: "Upload a PDF or Word (.docx) file, or paste your resume instead." }, { status: 415 });
    }

    text = text.replace(/[ \t]+/g, " ").replace(/\n\s*\n\s*\n+/g, "\n\n").trim();
    if (text.length < 40)
      return NextResponse.json({ error: "Couldn't read text from that file. Paste your resume instead." }, { status: 422 });

    return NextResponse.json({ text: text.slice(0, 20000) });
  } catch (err) {
    console.error("[parse-resume]", err);
    return NextResponse.json({ error: "Couldn't parse that file. Paste your resume instead." }, { status: 502 });
  }
}
