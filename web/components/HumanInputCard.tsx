"use client";
import { useState } from "react";
import { answerRun } from "@/lib/conversations";
import { runAction, type SavedRun, type HumanQuestion } from "@/lib/api";

export default function HumanInputCard({ run, question, onChange }: { run: SavedRun; question: HumanQuestion; onChange: () => void }) {
  const [answer, setAnswer] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function send(action: "answer" | "skip" | "cancel") {
    if (busy) return;
    setBusy(true); setError("");
    try {
      if (action === "cancel") await runAction(run.run_id, "cancel");
      else await answerRun(run, question.interrupt_id, answer, action === "skip");
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
    {error && <p role="alert" className="text-sm text-error">{error}</p>}
    <div className="mt-3 flex gap-4 text-sm"><button disabled={busy || !answer.trim()} onClick={() => void send("answer")}>提交并继续</button>
      {question.allow_skip && <button disabled={busy} onClick={() => void send("skip")}>跳过</button>}
      <button disabled={busy} onClick={() => void send("cancel")}>停止任务</button></div>
  </section>;
}
