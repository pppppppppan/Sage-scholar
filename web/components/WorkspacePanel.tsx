"use client";
import { useState } from "react";
import { listRuns, type SavedRun, type ChatMessage } from "@/lib/api";
import MemoryPanel from "./MemoryPanel";

export default function WorkspacePanel({ messages, workspaceId, sessionId }: {
  messages: ChatMessage[]; workspaceId: string; sessionId: string;
}) {
  const [open, setOpen] = useState(false);
  const [runs, setRuns] = useState<SavedRun[]>([]);
  const [error, setError] = useState("");
  return <>
    <button className="fixed right-5 bottom-5 z-40 rounded-lg bg-surface border border-border px-3 py-2 text-sm" onClick={() => {
      setOpen(true); setError("");
      void listRuns().then(data => setRuns(data.filter(run => run.execution?.workspace_id === workspaceId)))
        .catch(e => setError(e instanceof Error ? e.message : "加载失败"));
    }}>记忆管理</button>
    {open && <aside className="fixed right-0 top-0 bottom-0 z-50 w-[min(440px,100vw)] overflow-y-auto bg-surface border-l border-border p-5 shadow-xl">
      <div className="flex justify-between"><h2>记忆管理</h2><button onClick={() => setOpen(false)}>关闭</button></div>
      {error && <p role="alert" className="my-3 text-error text-sm">{error}</p>}
      <MemoryPanel key={`${workspaceId}:${sessionId}`} messages={messages} workspaceId={workspaceId} sessionId={sessionId} runs={runs} />
    </aside>}
  </>;
}
