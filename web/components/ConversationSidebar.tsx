"use client";
import { useEffect, useState } from "react";
import { deleteConversation, editConversation, listConversations, runLabels, type Conversation } from "@/lib/conversations";

export default function ConversationSidebar({ active, workspace, revision, onSelect, onNew, onDeleted, onChange }: {
  active: string | null; workspace: string; revision: number; onSelect: (id: string) => void;
  onNew: () => void; onDeleted: (id: string) => void; onChange: () => void;
}) {
  const [items, setItems] = useState<Conversation[]>([]);
  const [search, setSearch] = useState("");
  const [archived, setArchived] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let live = true;
    const refresh = () => listConversations(workspace, search, archived).then(data => { if (live) setItems(data); }).catch(e => { if (live) setError(String(e.message)); });
    const timer = setTimeout(refresh, 180);
    const poll = setInterval(refresh, 5000);
    return () => { live = false; clearTimeout(timer); clearInterval(poll); };
  }, [workspace, search, archived, revision]);
  async function action(work: () => Promise<void>) {
    setBusy(true); setError("");
    try { await work(); onChange(); } catch (e) { setError(e instanceof Error ? e.message : "操作失败"); }
    finally { setBusy(false); }
  }
  return <aside aria-label="历史对话" className="flex h-full w-64 shrink-0 flex-col border-r border-border bg-surface/60 p-4">
    <button onClick={onNew} className="mb-4 rounded-lg border border-border p-2 text-left">＋ 新对话</button>
    <input aria-label="搜索对话" placeholder="搜索历史对话…" value={search} onChange={e => setSearch(e.target.value)} className="rounded-lg border border-border bg-transparent p-2 text-sm" />
    <label className="my-3 flex items-center gap-2 text-xs text-muted-foreground"><input type="checkbox" checked={archived} onChange={e => setArchived(e.target.checked)} />已归档</label>
    {error && <p role="alert" className="text-xs text-error">{error}</p>}
    <nav className="thin-scroll min-h-0 flex-1 overflow-y-auto">
      {!items.length && <p className="py-5 text-sm text-muted-foreground">暂无对话</p>}
      {items.map((c, i) => <div key={c.conversation_id}>
        {(i === 0 || items[i-1].updated_at.slice(0,10) !== c.updated_at.slice(0,10)) && <p className="mt-4 mb-2 text-xs text-muted-foreground">{new Date(c.updated_at).toLocaleDateString()}</p>}
        <div className={`group rounded-xl p-2 ${active === c.conversation_id ? "bg-foreground/8" : "hover:bg-foreground/4"}`}>
          <button onClick={() => onSelect(c.conversation_id)} className="w-full text-left">
            <span className="block truncate text-sm">{c.title}</span>
            <span className="text-xs text-muted-foreground">{runLabels[c.status]}</span>
          </button>
          <div className="mt-2 flex gap-3 text-xs text-muted-foreground">
            <button disabled={busy} onClick={() => { const title = window.prompt("重命名对话", c.title); if (title?.trim()) void action(() => editConversation(c.conversation_id, { title })); }}>重命名</button>
            <button disabled={busy} onClick={() => void action(() => editConversation(c.conversation_id, { archived: !c.archived }))}>{c.archived ? "恢复" : "归档"}</button>
            <button disabled={busy} onClick={() => {
              if (window.confirm(`删除“${c.title}”？关联任务会先停止，对话、运行记录与检查点将删除。文献库、已保存证据和长期记忆仍保留；未完成的自动记忆抽取将取消。`))
                void action(async () => { await deleteConversation(c.conversation_id); onDeleted(c.conversation_id); });
            }}>删除</button>
          </div>
        </div>
      </div>)}
    </nav>
  </aside>;
}
