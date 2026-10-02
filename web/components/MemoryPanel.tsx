"use client";
import { useEffect, useState } from "react";
import { listMemories, listMemoryCandidates, saveMemory, deleteMemory, extractMemories, decideMemoryCandidate,
  type SavedMemory, type MemoryCandidate, type ChatMessage, type SavedRun } from "@/lib/api";

const labels = { user: "个人通用", workspace: "当前工作区", session: "当前会话" };
const types = { preference: "偏好", note: "明确笔记", decision: "研究决策", background: "课题背景", research_note: "研究线索" };

export default function MemoryPanel({ workspaceId, sessionId, messages, runs }: {
  workspaceId: string; sessionId: string; messages: ChatMessage[]; runs: SavedRun[];
}) {
  const [memories, setMemories] = useState<SavedMemory[]>([]);
  const [candidates, setCandidates] = useState<MemoryCandidate[]>([]);
  const [content, setContent] = useState("");
  const [scope, setScope] = useState<SavedMemory["scope"]>("workspace");
  const [kind, setKind] = useState<SavedMemory["type"]>("preference");
  const [editing, setEditing] = useState<string>();
  const [researchRun, setResearchRun] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let active = true;
    Promise.all([listMemories(workspaceId, sessionId), listMemoryCandidates(workspaceId, sessionId)])
      .then(([m, c]) => { if (active) { setMemories(m); setCandidates(c); } })
      .catch(e => { if (active) setError(e instanceof Error ? e.message : "加载失败"); });
    return () => { active = false; };
  }, [workspaceId, sessionId]);

  async function refresh() {
    const [m, c] = await Promise.all([listMemories(workspaceId, sessionId), listMemoryCandidates(workspaceId, sessionId)]);
    setMemories(m); setCandidates(c);
  }
  async function perform(work: () => Promise<unknown>) {
    setBusy(true); setError("");
    try { await work(); await refresh(); }
    catch (e) { setError(e instanceof Error ? e.message : "操作失败"); }
    finally { setBusy(false); }
  }
  return <section>
    <h3 className="mt-5 mb-2 font-medium">研究记忆 · {workspaceId}</h3>
    <p className="text-xs text-muted-foreground mb-2">只召回相关记忆。候选需确认，当前指令优先；记忆不是论文证据。</p>
    {error && <p role="alert" className="text-error text-sm my-2">{error}</p>}
    <textarea aria-label="记忆内容" className="w-full border border-border rounded p-2 text-sm" value={content} onChange={e => setContent(e.target.value)} />
    <div className="flex flex-wrap gap-2 my-2 text-sm">
      <select aria-label="记忆类型" value={kind} onChange={e => setKind(e.target.value as SavedMemory["type"])}>
        {Object.entries(types).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
      </select>
      <select aria-label="记忆范围" value={scope} onChange={e => setScope(e.target.value as SavedMemory["scope"])}>
        {Object.entries(labels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
      </select>
      <button disabled={busy || !content.trim()} onClick={() => void perform(async () => {
        await saveMemory(content, kind, editing, scope, workspaceId, sessionId); setContent(""); setEditing(undefined);
      })}>{editing ? "确认修改" : "确认记住"}</button>
      {editing && <button onClick={() => { setEditing(undefined); setContent(""); }}>取消修改</button>}
    </div>
    <div className="my-3 flex flex-wrap gap-2 text-sm">
      <button disabled={busy || !messages.length} onClick={() => void perform(() => extractMemories({ messages, workspace_id: workspaceId, session_id: sessionId }))}>从对话提取候选</button>
      <select aria-label="提取来源任务" className="max-w-full" value={researchRun} onChange={e => setResearchRun(e.target.value)}>
        <option value="">选择已完成研究</option>
        {runs.filter(r => ["completed", "partial"].includes(r.status) && (!r.execution || r.execution.workspace_id === workspaceId))
          .map(r => <option key={r.run_id} value={r.run_id}>{r.query.slice(0, 35)}</option>)}
      </select>
      <button disabled={busy || !researchRun} onClick={() => void perform(() => extractMemories({ run_id: researchRun }))}>提取研究线索</button>
      <button disabled={busy} onClick={() => void perform(refresh)}>刷新记忆</button>
    </div>
    {candidates.map(c => <div key={c.candidate_id} className="border border-border rounded p-2 my-2 text-sm">
      <p className="font-medium">待确认 · {labels[c.scope]} · {c.proposal.action}</p>
      <p className="whitespace-pre-wrap">{c.content}</p>
      <p className="text-xs text-muted-foreground">{c.proposal.reason || c.rationale}</p>
      <details className="text-xs my-1"><summary>查看出处与替换目标</summary>
        <p>{c.source.kind} · {c.source.reference}</p><blockquote>{c.source.quote}</blockquote>
        {c.proposal.targets.map(id => <p key={id}>已有记忆：{memories.find(m => m.memory_id === id)?.content || id}</p>)}
      </details>
      <div className="flex gap-3"><button disabled={busy} onClick={() => void perform(() => decideMemoryCandidate(c.candidate_id, true, c.revision))}>确认保存／合并</button>
        <button disabled={busy} onClick={() => void perform(() => decideMemoryCandidate(c.candidate_id, false))}>丢弃</button></div>
    </div>)}
    {memories.map(m => <div className="border-t border-border py-3 text-sm" key={m.memory_id}>
      <p className="whitespace-pre-wrap">{m.content}</p>
      <p className="text-xs text-muted-foreground">{labels[m.scope]} · {types[m.memory_type]} · {m.status === "active" ? "有效" : "已被替代"} · v{m.version}</p>
      <details className="text-xs"><summary>来源与版本</summary>
        <p>{m.source.kind} · {m.source.reference}</p><p>{m.source.quote}</p>
        <p>创建：{new Date(m.created_at).toLocaleString()}<br />更新：{new Date(m.updated_at).toLocaleString()}</p>
        <p>替代：{m.supersedes.join(", ") || "无"}<br />被替代为：{m.superseded_by || "无"}</p>
      </details>
      <div className="flex gap-3 mt-1 text-muted-foreground">
        {m.status === "active" && <button disabled={busy} onClick={() => { setEditing(m.memory_id); setContent(m.content); setKind(m.memory_type); setScope(m.scope); }}>修改</button>}
        <button disabled={busy} onClick={() => void perform(() => deleteMemory(m.memory_id))}>删除</button>
      </div>
    </div>)}
  </section>;
}
