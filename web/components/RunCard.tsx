"use client";
import { useEffect, useState } from "react";
import { runAction, auditReport, startRunEvents, type SavedRun, type ResearchEvent } from "@/lib/api";
import { isActive, runLabels } from "@/lib/conversations";
import InlineAnswer from "./InlineAnswer";
import ReportView from "./ReportView";
import HumanInputCard from "./HumanInputCard";

export default function RunCard({ run: saved, refresh, archived = false }: { run: SavedRun; refresh: () => void; archived?: boolean }) {
  const [stream, setStream] = useState<{ id: string; text: string } | null>(null);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const active = isActive(saved);
  useEffect(() => {
    if (!active) return;
    let live = true;
    const sync = () => { if (live) refresh(); };
    let streamed = "";
    const stop = startRunEvents(saved.run_id, (event: ResearchEvent) => {
      if (!live) return;
      if (event.type === "answer_delta") { streamed += event.text; setStream({ id: saved.run_id, text: streamed }); }
      if (["initializing_tools", "tool_unavailable", "research_started", "audit_started"].includes(event.type) && "message" in event) setNotice(event.message);
      if (["run_finished", "run_cancelled", "run_interrupted", "waiting_for_user", "error", "cancel_requested"].includes(event.type)) sync();
      if (event.type === "connection_error") { setNotice(event.message); sync(); }
    }, saved.event_cursor);
    const timer = setInterval(sync, 2000); // Authoritative state also heals a lost terminal event.
    return () => { live = false; stop(); clearInterval(timer); };
    // Only attach once per activation. Switching views closes the stream, never cancels the Run.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [saved.run_id, active]);
  async function action(work: () => Promise<unknown>) {
    setBusy(true); setError("");
    try { await work(); refresh(); } catch (e) { setError(e instanceof Error ? e.message : "操作失败"); }
    finally { setBusy(false); }
  }
  const answer = active && stream?.id === saved.run_id ? stream.text : saved.result?.answer;
  return <div className="flex flex-col gap-3" data-run-id={saved.run_id}>
    <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
      <span>{runLabels[saved.status]}</span><span>{saved.workflow}</span>
      {!!saved.consumed_budget.tool_calls && <span>工具 {saved.consumed_budget.tool_calls} 次</span>}
      {!!saved.consumed_budget.prompt_tokens && <span>累计 token {(saved.consumed_budget.prompt_tokens || 0) + (saved.consumed_budget.completion_tokens || 0)}</span>}
      {saved.consumed_budget.cache_observed_input_tokens > 0 && <span title="仅统计提供商返回缓存字段的请求，命中的输入仍计入 Run token 上限">
        输入缓存 {(100 * saved.consumed_budget.cached_input_tokens / saved.consumed_budget.cache_observed_input_tokens).toFixed(1)}%
      </span>}
      {active && <button disabled={busy || saved.status === "cancel_requested"} className="ml-auto rounded-lg border border-border px-3 py-1 text-foreground" onClick={() => void action(() => runAction(saved.run_id, "cancel"))}>■ {saved.status === "cancel_requested" ? "正在停止…" : "停止生成"}</button>}
    </div>
    {active && <p className="text-sm text-muted-foreground">{saved.progress?.message || notice || "正在处理…"}</p>}
    {!!saved.progress?.tasks?.length && <details className="rounded-lg border border-border p-3 text-sm"><summary>研究进度（{saved.progress.tasks.filter(t => t.status === "completed").length}/{saved.progress.tasks.length}）</summary>
      {saved.progress.tasks.map(t => <p key={t.task_id} className="mt-2">{t.status === "completed" ? "✓" : "·"} {t.objective} · {t.status}</p>)}</details>}
    {answer && <InlineAnswer answer={answer} evidence={saved.result?.evidence || []} />}
    {saved.result?.report && <div className="rounded-2xl border border-border bg-surface"><ReportView report={saved.result.report} stats={null} /></div>}
    {saved.status === "waiting_for_user" && saved.interrupts?.map(question => <HumanInputCard key={question.interrupt_id} run={saved} question={question} onChange={refresh} />)}
    {(saved.reason || saved.error) && <p className="whitespace-pre-wrap text-sm text-muted-foreground">{saved.reason || saved.error}</p>}
    {!!saved.remaining_gaps.length && <details className="text-sm"><summary>尚未覆盖的内容</summary><ul className="list-disc pl-5">{saved.remaining_gaps.map((gap,i) => <li key={i}>{gap}</li>)}</ul></details>}
    {error && <p role="alert" className="text-error text-sm">{error}</p>}
    {!active && !archived && saved.status !== "waiting_for_user" && <div className="flex gap-4 text-sm">
      {["partial", "interrupted", "failed", "cancelled"].includes(saved.status) && <>
        {saved.workflow === "deep_research" && <button disabled={busy || saved.restart_required} onClick={() => void action(() => runAction(saved.run_id, "resume"))}>继续研究</button>}
        {saved.workflow === "citation_audit" && ["interrupted", "cancelled", "failed"].includes(saved.status) && <button disabled={busy || saved.restart_required} onClick={() => void action(() => runAction(saved.run_id, "resume"))}>继续核验</button>}
        <button disabled={busy} onClick={() => void action(() => runAction(saved.run_id, "restart"))}>重新执行</button>
      </>}
      {saved.workflow === "deep_research" && saved.result?.report && <button disabled={busy} onClick={() => void action(() => auditReport(saved.run_id))}>核验报告引用</button>}
    </div>}
  </div>;
}
