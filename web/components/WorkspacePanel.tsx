"use client";
import { useState } from "react";
import { listMemories, saveMemory, deleteMemory, listRuns, runAction, type SavedMemory, type SavedRun } from "@/lib/api";

export default function WorkspacePanel({ onRun }: { onRun: (run: SavedRun) => void }) {
  const [open, setOpen] = useState(false);
  const [memories, setMemories] = useState<SavedMemory[]>([]);
  const [runs, setRuns] = useState<SavedRun[]>([]);
  const [content, setContent] = useState("");
  const [kind, setKind] = useState<SavedMemory["type"]>("preference");
  const [editing, setEditing] = useState<string>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function refresh() {
    const [m, r] = await Promise.all([listMemories(), listRuns()]);
    setMemories(m); setRuns(r);
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
      <h3 className="mt-5 mb-2 font-medium">工作区记忆</h3>
      <p className="text-xs text-muted-foreground mb-2">仅保存你明确确认的偏好和笔记，不作为论文证据。当前指令优先。</p>
      <textarea aria-label="记忆内容" className="w-full rounded border border-border p-2 text-sm" value={content} onChange={e => setContent(e.target.value)} />
      <div className="flex gap-3 text-sm my-2">
        <select aria-label="记忆类型" value={kind} onChange={e => setKind(e.target.value as SavedMemory["type"])}><option value="preference">偏好</option><option value="note">笔记</option></select>
        <button disabled={busy || !content.trim()} onClick={() => void perform(async () => {
          await saveMemory(content, kind, editing); setContent(""); setEditing(undefined); await refresh();
        })}>{editing ? "确认修改" : "确认记住"}</button>
        {editing && <button onClick={() => { setEditing(undefined); setContent(""); }}>取消修改</button>}
      </div>
      {memories.map(m => <div className="border-t border-border py-3 text-sm" key={m.memory_id}>
        <p className="whitespace-pre-wrap">{m.content}</p><div className="flex gap-3 mt-1 text-muted-foreground">
          <button onClick={() => { setEditing(m.memory_id); setContent(m.content); setKind(m.type); }}>修改</button>
          <button disabled={busy} onClick={() => void perform(async () => { await deleteMemory(m.memory_id); await refresh(); })}>删除</button>
        </div>
      </div>)}
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
