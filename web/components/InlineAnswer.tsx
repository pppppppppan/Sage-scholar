"use client";

import MarkdownContent from "@/components/MarkdownContent";
import Image from "next/image";
import { API_BASE } from "@/lib/api";
import type { EvidenceRef } from "@/lib/api";

export default function InlineAnswer({ answer, evidence }: { answer: string; evidence: EvidenceRef[] }) {
  return (
    <div className="w-full max-w-3xl rounded-2xl border border-foreground/10 bg-surface p-6 shadow-sm">
      <article className="report-prose">
        <MarkdownContent>{answer}</MarkdownContent>
      </article>
      {evidence.length > 0 && (
        <div className="mt-6 border-t border-foreground/10 pt-4">
          <p className="mb-3 font-mono text-[11px] uppercase tracking-widest text-muted-foreground">Evidence</p>
          <div className="flex flex-col gap-3">
            {evidence.map((item, index) => (
              <details key={item.evidence_id} className="rounded-lg border border-foreground/10 px-3 py-2 text-sm">
                <summary className="cursor-pointer text-foreground/70">[{index + 1}] {String(item.metadata.title || item.source_id)}{item.section ? ` · ${item.section}` : ""}{item.page !== null ? ` · p.${item.page}` : ""}</summary>
                <p className="mt-2 whitespace-pre-wrap text-foreground/65">{item.text}</p>
                {item.metadata.kind === "visual_source" && (
                  <a href={`${API_BASE}/api/evidence/${encodeURIComponent(item.evidence_id)}/image`} target="_blank" rel="noreferrer">
                    <Image unoptimized loading="lazy" className="mt-3 h-auto max-w-full rounded"
                      src={`${API_BASE}/api/evidence/${encodeURIComponent(item.evidence_id)}/image`}
                      width={Number(item.metadata.width) || 800} height={Number(item.metadata.height) || 1000}
                      alt={`论文第 ${item.page} 页：${item.section || "原图"}`} />
                  </a>
                )}
              </details>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
