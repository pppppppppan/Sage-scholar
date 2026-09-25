"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { EvidenceRef } from "@/lib/api";

export default function InlineAnswer({ answer, evidence }: { answer: string; evidence: EvidenceRef[] }) {
  return (
    <div className="w-full max-w-3xl rounded-2xl border border-foreground/10 bg-surface p-6 shadow-sm">
      <article className="report-prose">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
      </article>
      {evidence.length > 0 && (
        <div className="mt-6 border-t border-foreground/10 pt-4">
          <p className="mb-3 font-mono text-[11px] uppercase tracking-widest text-muted-foreground">Evidence</p>
          <div className="flex flex-col gap-3">
            {evidence.map((item, index) => (
              <details key={item.evidence_id} className="rounded-lg border border-foreground/10 px-3 py-2 text-sm">
                <summary className="cursor-pointer text-foreground/70">[{index + 1}] {item.source_id}</summary>
                <p className="mt-2 whitespace-pre-wrap text-foreground/65">{item.text}</p>
              </details>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
