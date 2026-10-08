"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import ConversationSidebar from "@/components/ConversationSidebar";
import RunCard from "@/components/RunCard";
import WorkspacePanel from "@/components/WorkspacePanel";
import LibraryDrawer from "@/components/LibraryDrawer";
import QueryInput from "@/components/QueryInput";
import { submitQuery, listDocs, runAction, type LibraryDoc } from "@/lib/api";
import { conversationUrl, createConversation, getConversation, isActive, type ConversationDetail } from "@/lib/conversations";

export default function Home() {
  const [id, setId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ConversationDetail | null>(null);
  const [workspace, setWorkspace] = useState("default");
  const [workspaceDraft, setWorkspaceDraft] = useState("default");
  const [sidebar, setSidebar] = useState(true);
  const [selected, setSelected] = useState<LibraryDoc[]>([]);
  const [revision, setRevision] = useState(0);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const activeId = useRef<string | null>(null);
  const epoch = useRef(0);
  const scroll = useRef<HTMLDivElement>(null);
  const follow = useRef(true);
  const pendingSubmission = useRef<{ query: string; id: string; conversation: string } | null>(null);
  const refresh = useCallback(async () => {
    const current = activeId.current;
    if (!current) return;
    const version = ++epoch.current;
    try {
      const data = await getConversation(current);
      if (activeId.current !== current || version !== epoch.current) return;
      setDetail(data); setWorkspace(data.conversation.workspace_id); setWorkspaceDraft(data.conversation.workspace_id);
      if (follow.current) requestAnimationFrame(() => scroll.current?.scrollTo({ top: scroll.current.scrollHeight }));
    } catch (e) { if (activeId.current === current) setError(e instanceof Error ? e.message : "加载失败"); }
  }, []);
  const navigate = useCallback((next: string | null, push = true) => {
    activeId.current = next; epoch.current++; follow.current = true;
    setId(next); setDetail(null); setSelected([]); setError(""); pendingSubmission.current = null;
    if (push) window.history.pushState(null, "", conversationUrl(next));
    if (next) localStorage.setItem("sage-conversation", next); else localStorage.removeItem("sage-conversation");
    void refresh();
    if (next) {
      void Promise.all([getConversation(next), listDocs()]).then(([data, docs]) => {
        if (activeId.current !== next) return;
        const ids = data.runs.at(-1)?.execution?.document_ids || [];
        setSelected(docs.filter(doc => ids.includes(doc.document_id)));
      }).catch(() => {});
    }
  }, [refresh]);
  useEffect(() => {
    const url = new URL(window.location.href);
    const restored = url.searchParams.get("conversation") || localStorage.getItem("sage-conversation");
    if (restored) window.history.replaceState(null, "", conversationUrl(restored));
    const timer = setTimeout(() => navigate(restored, false), 0);
    const back = () => navigate(new URL(window.location.href).searchParams.get("conversation"), false);
    window.addEventListener("popstate", back);
    return () => { clearTimeout(timer); window.removeEventListener("popstate", back); };
  }, [navigate]);
  const active = detail?.runs.find(r => isActive(r) || r.status === "waiting_for_user");
  async function newConversation() {
    try { const c = await createConversation(workspace); navigate(c.conversation_id); setRevision(v => v + 1); }
    catch (e) { setError(e instanceof Error ? e.message : "新建失败"); }
  }
  async function submit(query: string) {
    if (sending || active) return;
    setSending(true); setError("");
    try {
      let current = activeId.current;
      if (!current) { const c = await createConversation(workspace); current = c.conversation_id; navigate(current); }
      const pending = pendingSubmission.current;
      const requestId = pending?.query === query && pending.conversation === current ? pending.id : crypto.randomUUID();
      pendingSubmission.current = { id: requestId, query, conversation: current };
      const response = await submitQuery(query, [], selected.map(d => d.document_id), {
        workspace_id: workspace, conversation_id: current, request_id: requestId,
      });
      pendingSubmission.current = null;
      if (response.status !== "accepted") setError(response.message || "当前功能尚未实现");
      if (activeId.current === current) await refresh();
      setRevision(v => v + 1);
    } catch (e) { setError(e instanceof Error ? e.message : "发送失败，请重试；相同请求不会重复执行。"); }
    finally { setSending(false); }
  }
  function changed() { void refresh(); setRevision(v => v + 1); }
  return <div className="flex min-h-0 w-full flex-1 overflow-hidden">
    {sidebar && <ConversationSidebar active={id} workspace={workspace} revision={revision} onSelect={navigate}
      onNew={() => void newConversation()} onChange={changed} onDeleted={deleted => { if (deleted === activeId.current) navigate(null); }} />}
    <main className="flex min-w-0 flex-1 flex-col">
      <header className="flex h-14 shrink-0 items-center gap-4 border-b border-border px-4 pr-24">
        <button aria-label="展开或收起历史对话" aria-expanded={sidebar} onClick={() => setSidebar(value => !value)}>☰</button>
        <button onClick={() => navigate(null)}>首页</button><button onClick={() => void newConversation()}>＋ 新对话</button>
        <span className="truncate text-sm text-muted-foreground">{detail?.conversation.title || "SAGE Scholar"}</span>
      </header>
      <div ref={scroll} data-scroll-container className="chat-scrollbar min-h-0 flex-1 overflow-y-scroll px-4" onScroll={() => {
        const node = scroll.current; if (node) follow.current = node.scrollHeight - node.scrollTop - node.clientHeight < 120;
      }}>
        <div className="mx-auto flex max-w-3xl flex-col gap-6 py-8">
          {!id && <div className="py-20 text-center"><h1 className="text-4xl font-medium">SAGE Scholar</h1><p className="mt-4 text-muted-foreground">简单问题直接回答，复杂问题深入研究。</p></div>}
          {id && !detail && <p className="text-muted-foreground">正在恢复对话…</p>}
          {detail?.messages.map(message => message.role === "user" ? <div key={message.message_id} className="ml-auto max-w-[85%] whitespace-pre-wrap rounded-2xl bg-foreground/6 px-5 py-3 text-sm">{message.content}</div> : (() => {
            const run = detail.runs.find(r => r.run_id === message.run_id);
            return run ? <RunCard key={message.message_id} run={run} refresh={changed} archived={detail.conversation.archived} /> : null;
          })())}
        </div>
      </div>
      <footer className="mx-auto flex w-full max-w-3xl shrink-0 flex-col items-center gap-2 px-4 pt-2 pb-6">
        {error && <p role="alert" className="w-full whitespace-pre-wrap text-sm text-error">{error}</p>}
        {!!selected.length && <p className="w-full text-xs text-muted-foreground">本次资料：{selected.map(d => d.title).join("、")} <button onClick={() => setSelected([])}>清除</button></p>}
        {detail?.conversation.archived ? <p className="py-6 text-sm text-muted-foreground">对话已归档。在左侧恢复后可继续。</p> : <QueryInput key={id || "new"} onSubmit={submit} disabled={sending || !!active} stopping={active?.status === "cancel_requested"}
          onStop={active ? () => { void runAction(active.run_id, "cancel").then(changed).catch(e => setError(e.message)); } : undefined} />}
        {active?.status === "waiting_for_user" && <p className="text-xs text-muted-foreground">请在上方卡片中回答，或停止任务后发送新消息。</p>}
        <div className="flex gap-2 self-start text-xs text-muted-foreground"><input aria-label="工作区" className="w-28 bg-transparent" value={workspaceDraft} onChange={e => setWorkspaceDraft(e.target.value)} />
          <button disabled={!workspaceDraft.trim() || workspaceDraft.trim() === workspace} onClick={() => { setWorkspace(workspaceDraft.trim()); navigate(null); }}>切换工作区</button></div>
      </footer>
    </main>
    <LibraryDrawer selectedDocuments={selected} onSelectionChange={setSelected} />
    <WorkspacePanel messages={detail?.messages.filter(m => m.content) || []} workspaceId={workspace} sessionId={id || ""} />
  </div>;
}
