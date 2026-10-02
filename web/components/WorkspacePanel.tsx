"use client";
import { useState } from "react";
import { listRuns, runAction, type SavedRun, type ChatMessage } from "@/lib/api";
import MemoryPanel from "./MemoryPanel";

export default function WorkspacePanel({ onRun, messages, workspaceId, sessionId }: {
  onRun: (run: SavedRun) => void; messages: ChatMessage[]; workspaceId: string; sessionId: string;
}) {
  const [open, setOpen] = useState(false);
  const [runs, setRuns] = useState<SavedRun[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function refresh() {
    setRuns(await listRuns());
  }
  async function perform(work: () => Promise<void>) {
    setError(""); setBusy(true);
    try { await work(); } catch (e) { setError(e instanceof Error ? e.message : "操作失败"); }
    finally { setBusy(false); }
  }
  return <>
    <button className="fixed right-5 bottom-5 z-40 rounded-lg bg-surface border border-border px-3 py-2 text-sm" onClick={() => {
      setOpen(true); void perform(refresh);
    }}>任务与记忆</button>
    {open && <aside className="fixed right-0 top-0 bottom-0 z-50 w-[min(440px,100vw)] overflow-y-auto bg-surface border-l border-border p-5 shadow-xl">
      <div className="flex justify-between"><h2>任务与记忆</h2><button onClick={() => setOpen(false)}>关闭</button></div>
      {error && <p role="alert" className="my-3 text-error text-sm">{error}</p>}
      <MemoryPanel key={`${workspaceId}:${sessionId}`} messages={messages} workspaceId={workspaceId} sessionId={sessionId} runs={runs} />
      <div className="mt-6 mb-2 flex justify-between"><h3 className="font-medium">历史任务</h3><button disabled={busy} onClick={() => void perform(refresh)}>刷新</button></div>
      {runs.map(run => <div className="border-t border-border py-3 text-sm" key={run.run_id}>
        <p className="line-clamp-2">{run.query}</p>
        <p className="text-muted-foreground">{run.workflow} · {run.status} · 工具调用 {run.consumed_budget.tool_calls || 0}</p>
        {(run.reason || run.error) && <p className="text-xs mt-1">{run.reason || run.error}</p>}
        <div className="flex gap-3 mt-2">
          <button onClick={() => { onRun(run); setOpen(false); }}>查看</button>
          {["interrupted", "failed"].includes(run.status) && <>
            <button disabled={busy || run.restart_required} onClick={() => void perform(async () => { onRun(await runAction(run.run_id, "resume")); setOpen(false); })}>继续原任务</button>
            <button disabled={busy} onClick={() => void perform(async () => { onRun(await runAction(run.run_id, "restart")); setOpen(false); })}>重新开始</button>
          </>}
          {["pending", "running"].includes(run.status) && <button disabled={busy} onClick={() => void perform(async () => { await runAction(run.run_id, "cancel"); await refresh(); })}>取消</button>}
        </div>
      </div>)}
    </aside>}
  </>;
}
