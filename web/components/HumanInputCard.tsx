"use client";
import { useState } from "react";
import { answerRun } from "@/lib/conversations";
import { listDocs, runAction, type SavedRun, type HumanQuestion, type RoutingSettings, type LibraryDoc } from "@/lib/api";

export default function HumanInputCard({ run, question, onChange }: { run: SavedRun; question: HumanQuestion; onChange: () => void }) {
  const [answer, setAnswer] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [applySettings, setApplySettings] = useState(false);
  const [settings, setSettings] = useState<RoutingSettings>(question.routing_settings ?? { source_scope: null, allow_web: null, document_ids: [] });
  const [documents, setDocuments] = useState<LibraryDoc[]>([]);
  const [loadingDocuments, setLoadingDocuments] = useState(false);
  const settingsChanged = JSON.stringify(settings) !== JSON.stringify(question.routing_settings);
  async function refreshDocuments() {
    setLoadingDocuments(true);
    try { setDocuments(await listDocs()); }
    catch (e) { setError(e instanceof Error ? e.message : "文献库加载失败"); }
    finally { setLoadingDocuments(false); }
  }
  async function send(action: "answer" | "skip" | "cancel") {
    if (busy) return;
    setBusy(true); setError("");
    try {
      if (action === "cancel") await runAction(run.run_id, "cancel");
      else await answerRun(run, question.interrupt_id, answer, action === "skip", action === "answer" && applySettings ? settings : undefined);
      onChange();
    } catch (e) { setError(e instanceof Error ? e.message : "提交失败"); onChange(); }
    finally { setBusy(false); }
  }
  return <section className="rounded-xl border border-accent/40 bg-surface p-5" aria-label="等待用户回答">
    <p className="font-medium">{question.question}</p>
    {question.reason && <p className="mt-2 text-sm text-muted-foreground">{question.reason}</p>}
    <div className="my-3 flex flex-col gap-2">{question.options.map(option => <label key={option} className="flex items-center gap-2 text-sm">
      <input type="radio" name={question.interrupt_id} checked={answer === option} onChange={() => setAnswer(option)} disabled={busy} />{option}
    </label>)}</div>
    <textarea aria-label="补充回答" value={answer} onChange={e => setAnswer(e.target.value)} disabled={busy} placeholder="选择上面的选项，或在这里补充…" className="w-full rounded-lg border border-border bg-transparent p-2" />
    {question.routing_settings && <details open={question.kind === "source_policy"} className="mt-3 text-sm">
      <summary>本次任务的资料设置</summary>
      <p className="my-2 text-muted-foreground">仅对本次任务生效。文字回答不会自动改变联网权限；修改后勾选确认并提交。</p>
      <label className="block my-2">资料范围
        <select aria-label="资料范围" className="ml-2 border border-border bg-surface p-1" disabled={busy} value={settings.source_scope ?? ""}
          onChange={e => setSettings({ ...settings, source_scope: (e.target.value || null) as RoutingSettings["source_scope"] })}>
          <option value="">自动理解任务</option><option value="conversation">仅对话内容</option>
          <option value="selected_documents">仅选中文献</option><option value="workspace">整个本地文献库</option>
          <option value="public">仅公网</option><option value="mixed">本地与公网</option>
        </select>
      </label>
      <label className="block my-2">联网权限
        <select aria-label="联网权限" className="ml-2 border border-border bg-surface p-1" disabled={busy} value={settings.allow_web === null ? "" : String(settings.allow_web)}
          onChange={e => setSettings({ ...settings, allow_web: e.target.value === "" ? null : e.target.value === "true" })}>
          <option value="">按资料范围决定</option><option value="false">禁止联网</option><option value="true">允许联网</option>
        </select>
      </label>
      <button type="button" className="underline" disabled={busy || loadingDocuments} onClick={() => void refreshDocuments()}>选择或刷新文献</button>
      {documents.map(doc => <label key={doc.document_id} className="flex gap-2 my-1">
        <input type="checkbox" disabled={busy} checked={settings.document_ids.includes(doc.document_id)} onChange={e => setSettings({ ...settings,
          document_ids: e.target.checked ? [...settings.document_ids, doc.document_id] : settings.document_ids.filter(id => id !== doc.document_id) })} />{doc.title}
      </label>)}
      {settings.document_ids.length > 0 && <button type="button" disabled={busy} className="block underline my-2" onClick={() => setSettings({ ...settings, document_ids: [] })}>清除选中文献（{settings.document_ids.length} 篇）</button>}
      <label className="flex gap-2 my-3"><input type="checkbox" disabled={busy} checked={applySettings} onChange={e => setApplySettings(e.target.checked)} />我确认将上述资料设置用于本次任务</label>
    </details>}
    {error && <p role="alert" className="text-sm text-error">{error}</p>}
    <div className="mt-3 flex gap-4 text-sm"><button disabled={busy || (applySettings ? !settingsChanged : !answer.trim() || question.kind === "source_policy")} onClick={() => void send("answer")}>提交并继续</button>
      {question.allow_skip && <button disabled={busy} onClick={() => void send("skip")}>跳过</button>}
      <button disabled={busy} onClick={() => void send("cancel")}>停止任务</button></div>
  </section>;
}
