import { API_BASE, expectJson, type SavedRun, type ChatMessage } from "./api";
export interface Conversation { conversation_id: string; workspace_id: string; title: string; archived: boolean; status: SavedRun["status"]; created_at: string; updated_at: string }
export interface ConversationMessage extends ChatMessage { message_id: string; run_id: string; kind: string }
export interface ConversationDetail { conversation: Conversation; messages: ConversationMessage[]; runs: SavedRun[] }
export async function listConversations(workspace: string, search = "", archived = false): Promise<Conversation[]> {
  const params = new URLSearchParams({ workspace_id: workspace, search, archived: String(archived) });
  return (await expectJson<{ conversations: Conversation[] }>(await fetch(`${API_BASE}/api/conversations?${params}`))).conversations;
}
export async function createConversation(workspace: string): Promise<Conversation> {
  return (await expectJson<{ conversation: Conversation }>(await fetch(`${API_BASE}/api/conversations`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ workspace_id: workspace }),
  }))).conversation;
}
export async function getConversation(id: string): Promise<ConversationDetail> {
  return expectJson(await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(id)}`));
}
export async function editConversation(id: string, data: { title?: string; archived?: boolean }): Promise<void> {
  await expectJson(await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(id)}`, {
    method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
  }));
}
export async function deleteConversation(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!response.ok) await expectJson(response);
}
export async function answerRun(run: SavedRun, interruptId: string, answer: string, skip = false): Promise<SavedRun> {
  return (await expectJson<{ run: SavedRun }>(await fetch(`${API_BASE}/api/runs/${encodeURIComponent(run.run_id)}/respond`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
      run_id: run.run_id, conversation_id: run.execution?.conversation_id, interrupt_id: interruptId, answer, skip,
    }),
  }))).run;
}
export const runLabels: Record<SavedRun["status"], string> = {
  pending: "排队中", running: "运行中", waiting_for_user: "等待你的回答", cancel_requested: "正在停止…",
  completed: "已完成", partial: "部分完成", failed: "失败", cancelled: "已停止", interrupted: "已中断",
};
export function isActive(run: SavedRun) { return ["pending", "running", "cancel_requested"].includes(run.status); }
export function conversationUrl(id: string | null) { return id ? `/?conversation=${encodeURIComponent(id)}` : "/"; }
