"use client";

import { useEffect, useRef, useState } from "react";
import InlineAnswer from "@/components/InlineAnswer";
import LibraryDrawer from "@/components/LibraryDrawer";
import QueryInput from "@/components/QueryInput";
import ReportView from "@/components/ReportView";
import ResearchProgress from "@/components/ResearchProgress";
import StreamingText from "@/components/StreamingText";
import {
  submitQuery,
  type ChatMessage,
  type EvidenceRef,
  type LibraryDoc,
} from "@/lib/api";

type Stage = "input" | "chat" | "researching" | "report";

interface ChatTurn {
  id: string;
  query: string;
  answer: string | null;
  evidence: EvidenceRef[];
  status: "pending" | "done" | "error" | "researching";
  kind: "inline" | "research";
}

export default function Home() {
  const [selectedDocuments, setSelectedDocuments] = useState<LibraryDoc[]>([]);
  const [stage, setStage] = useState<Stage>("input");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [sending, setSending] = useState(false);
  const [activeQuery, setActiveQuery] = useState("");
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [report, setReport] = useState("");
  const [stats, setStats] = useState<{ total_calls: number; total_tokens: number } | null>(null);
  const [exiting, setExiting] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (stage === "chat") {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [stage, turns]);

  function conversationHistory(): ChatMessage[] {
    return turns
      .filter((turn) => turn.status === "done" && turn.answer)
      .flatMap<ChatMessage>((turn) => [
        { role: "user", content: turn.query },
        { role: "assistant", content: turn.answer!.slice(0, 20_000) },
      ])
      .slice(-20);
  }

  async function handleQuerySubmit(value: string) {
    if (sending || stage === "researching") return;

    const history = conversationHistory();
    const turnId = crypto.randomUUID();
    const nextTurn: ChatTurn = {
      id: turnId,
      query: value,
      answer: null,
      evidence: [],
      status: "pending",
      kind: "inline",
    };

    setTurns((previous) => [...previous, nextTurn]);
    setActiveQuery(value);
    setActiveTurnId(turnId);
    setReport("");
    setStats(null);
    setStage("chat");
    setSending(true);

    try {
      const response = await submitQuery(value, history, selectedDocuments.map((doc) => doc.document_id));
      if (response.status === "accepted" && response.run_id) {
        setTurns((previous) =>
          previous.map((turn) =>
            turn.id === turnId
              ? { ...turn, kind: "research", status: "researching" }
              : turn,
          ),
        );
        setRunId(response.run_id);
        setStage("researching");
        return;
      }

      const answer = response.answer || response.message || "当前请求暂时无法处理。";
      setTurns((previous) =>
        previous.map((turn) =>
          turn.id === turnId
            ? {
                ...turn,
                answer,
                evidence: response.evidence || [],
                status: "done",
              }
            : turn,
        ),
      );
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "请求失败";
      setTurns((previous) =>
        previous.map((turn) =>
          turn.id === turnId
            ? { ...turn, answer: `错误：${message}`, status: "error" }
            : turn,
        ),
      );
    } finally {
      setSending(false);
    }
  }

  function handleResearchReport(value: string, usage: { total_calls: number; total_tokens: number } | null) {
    setReport(value);
    if (usage) setStats(usage);
    if (activeTurnId) {
      setTurns((previous) =>
        previous.map((turn) =>
          turn.id === activeTurnId
            ? { ...turn, answer: value, status: "done", kind: "research" }
            : turn,
        ),
      );
    }
    setStage("report");
  }

  function handleNewConversation() {
    setExiting(true);
    setTimeout(() => {
      setStage("input");
      setTurns([]);
      setSelectedDocuments([]);
      setSending(false);
      setActiveQuery("");
      setActiveTurnId(null);
      setRunId(null);
      setReport("");
      setStats(null);
      setExiting(false);
    }, 250);
  }

  const isReport = stage === "report";

  return (
    <>
      <LibraryDrawer selectedDocuments={selectedDocuments} onSelectionChange={setSelectedDocuments} />
      {selectedDocuments.length > 0 && (
        <div className="mx-auto flex max-w-3xl items-center gap-3 px-4 py-2 text-sm text-muted-foreground">
          <span>本次资料：{selectedDocuments.map((doc) => doc.title).join("、")}</span>
          <button onClick={() => setSelectedDocuments([])} className="shrink-0 underline">清除选择</button>
        </div>
      )}
      {stage === "input" && (
        <>
          <div className="flex flex-1 flex-col items-center px-4">
            <HeroSection onSubmit={handleQuerySubmit} />
          </div>
        </>
      )}

      {stage !== "input" && (
        <div className={`relative flex min-h-0 w-full flex-1 overflow-hidden ${exiting ? "back-exit" : ""}`}>
          <div className="transition-[flex] duration-700" style={{ flex: isReport ? "0 0 0px" : "1 1 0px" }} />
          <div className="chat-scrollbar min-h-0 w-full max-w-3xl shrink-0 overflow-y-scroll px-4" data-scroll-container>
            <div
              className="sticky top-0 z-10 -mx-4 flex h-[58px] items-start px-4"
              style={{ background: "linear-gradient(to bottom, var(--background) 62%, transparent)" }}
            >
              <button onClick={handleNewConversation} className="mt-4 text-[13px] text-muted-foreground hover:text-foreground">
                ＋ 新对话
              </button>
            </div>

            {stage === "chat" ? (
              <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 pb-4">
                {turns.map((turn) => (
                  <div key={turn.id} className="flex flex-col gap-3 bubble-enter">
                    <div className="ml-auto max-w-[85%] rounded-2xl bg-foreground/[0.06] px-5 py-3 text-sm text-foreground/80">
                      {turn.query}
                    </div>
                    {turn.status === "pending" ? (
                      <div className="w-full rounded-2xl border border-foreground/10 bg-surface px-5 py-4 text-sm text-muted-foreground">
                        <StreamingText text="正在回答" mode="dots" />
                      </div>
                    ) : turn.kind === "research" ? (
                      <div className="w-full rounded-2xl border border-foreground/10 bg-surface px-5 py-4 text-sm text-foreground/70">
                        深度调研已完成，报告内容已加入本轮对话上下文。
                      </div>
                    ) : (
                      <InlineAnswer answer={turn.answer || ""} evidence={turn.evidence} />
                    )}
                  </div>
                ))}
                <div ref={chatEndRef} />
                <div className="sticky bottom-0 bg-gradient-to-t from-background from-80% to-transparent pb-5 pt-4">
                  <QueryInput onSubmit={handleQuerySubmit} disabled={sending} />
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-4 pb-8">
                <div className="w-full max-w-3xl rounded-2xl bg-foreground/[0.04] px-5 py-3 text-sm text-foreground/75">
                  {activeQuery}
                </div>
                {runId && (
                  <ResearchProgress runId={runId} onReport={handleResearchReport} />
                )}
                {isReport && (
                  <div className="sticky bottom-0 w-full max-w-2xl bg-gradient-to-t from-background from-80% to-transparent pb-2 pt-5">
                    <QueryInput onSubmit={handleQuerySubmit} />
                  </div>
                )}
              </div>
            )}
          </div>

          <div
            data-report-panel
            className={`thin-scroll min-h-0 min-w-0 flex-1 overflow-y-auto transition-opacity duration-500 ${isReport ? "border-l border-foreground/8 bg-surface/40 opacity-100" : "pointer-events-none opacity-0"}`}
          >
            {report && <ReportView report={report} stats={stats} />}
          </div>
        </div>
      )}
    </>
  );
}

function HeroSection({ onSubmit }: { onSubmit: (query: string) => void }) {
  const [showSubtitle, setShowSubtitle] = useState(false);
  return (
    <div className="flex w-full flex-1 flex-col items-center">
      <div className="flex flex-1 flex-col items-center justify-center">
        <h1 className="w-full text-center text-4xl font-medium">
          <StreamingText text="SAGE Scholar" mode="typewriter" speed="slow" pause={350} onComplete={() => setShowSubtitle(true)} />
        </h1>
        <div className={`w-full overflow-hidden transition-all duration-1000 ${showSubtitle ? "mt-3 max-h-16 opacity-100" : "mt-0 max-h-0 opacity-0"}`}>
          <p className="text-center text-lg tracking-wide text-foreground/65">简单问题直接回答，复杂问题深入研究。</p>
        </div>
      </div>
      <div className="flex w-full justify-center pb-8">
        <QueryInput onSubmit={onSubmit} />
      </div>
    </div>
  );
}
