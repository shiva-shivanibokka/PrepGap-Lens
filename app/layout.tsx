import type { Metadata } from "next";
import { Space_Grotesk, Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const display = Space_Grotesk({ subsets: ["latin"], variable: "--font-display", weight: ["500", "600", "700"] });
const body = Inter({ subsets: ["latin"], variable: "--font-body" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", weight: ["500", "700"] });

export const metadata: Metadata = {
  title: "PrepGap-Lens — measure your interview-readiness gap",
  description:
    "Paste a job description and your resume. PrepGap-Lens scores your readiness, ranks your biggest gaps, and builds an adaptive daily study plan. Bring your own API key.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body>
        <div className="aurora" aria-hidden="true">
          <span className="blob b1" />
          <span className="blob b2" />
          <span className="blob b3" />
          <span className="blob b4" />
        </div>
        {children}
      </body>
    </html>
  );
}
