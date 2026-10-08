const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export type Workflow =
  | "direct"
  | "qa"
  | "search"
  | "deep_research"
  | "citation_audit"
  | "paper2code";

export interface EvidenceRef {
  evidence_id: string;
  source_id: string;
  document_id: string | null;
  original_text: string;
  location: Record<string, unknown>;
  source_version: string;
  text: string;
  page: number | null;
  section: string | null;
  table: string | null;
  parser_version: string;
  score: number | null;
  metadata: Record<string, unknown>;
}

export interface QueryResponse {
  request_id: string;
  route: {
    workflow: Workflow;
    source_scope: string;
    document_ids: string[];
    allow_web: boolean;
    execution_mode: "inline" | "run";
    needs_clarification: boolean;
    clarification_message: string | null;
    reason_code: string;
  };
  status: "inline" | "accepted" | "needs_clarification" | "not_implemented";
  answer: string | null;
  evidence: EvidenceRef[];
  citations: Record<string, string>;
  citation_issues: string[];
  reading_strategy: "targeted" | "overview" | null;
  run_id: string | null;
  message: string | null;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

async function expectJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.text();
    let message = `请求失败 (${response.status})`;
    try {
      const data = JSON.parse(body);
      if (typeof data.detail === "string") message = data.detail;
    } catch {
      // Proxy error pages are not suitable for displaying in the document drawer.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function submitQuery(
  query: string,
  history: ChatMessage[] = [],
  documentIds: string[] = [],
  memoryContext: { workspace_id?: string; session_id?: string } = {},
): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE}/api/queries`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      history,
      document_ids: documentIds,
      ...memoryContext,
    }),
  });
  return expectJson<QueryResponse>(response);
}

export interface SaveReportResult {
  document_id: string;
  document_version: string;
  ingestion_status: LibraryDoc["ingestion_status"];
  title: string;
  status: "skipped" | "overwritten" | "created";
}

export interface LibraryDoc {
  active_version?: string | null;
  document_id: string;
  document_version: string;
  ingestion_status: "uploaded" | "parsing" | "indexing" | "ready" | "failed";
  title: string;
  source_type: string;
  added_at: string;
}

export async function saveReport(title: string, content: string): Promise<SaveReportResult> {
  const response = await fetch(`${API_BASE}/api/library/save-report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, content }),
  });
  return expectJson<SaveReportResult>(response);
}

export async function listDocs(): Promise<LibraryDoc[]> {
  const response = await fetch(`${API_BASE}/api/library`);
  return expectJson<LibraryDoc[]>(response);
}

export async function deleteDoc(documentId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/library/${encodeURIComponent(documentId)}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error(`删除失败 (${response.status})`);
}

export async function uploadFile(file: File): Promise<SaveReportResult> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/api/library/upload`, {
    method: "POST",
    body: form,
  });
  return expectJson<SaveReportResult>(response);
}

export type ResearchEvent =
  | { type: "initializing_tools" | "research_started" | "audit_started"; message: string }
  | { type: "audit_claim_checked"; claim: { claim_id: string }; support_status: string; citation_status?: "present" | "missing" }
  | { type: "tool_available" | "tool_unavailable"; server: string; message: string; reason?: string | null }
  | { type: "plan"; sub_questions: { label: string; question: string }[] }
  | { type: "research"; question: string; preview: string; tool_call_counts: Record<string, number>; status?: "completed" | "failed" }
  | { type: "review"; round: number; review_summary: { question: string; verdict: string; failed: Record<string, boolean>; evidence?: Record<string, string> }[]; missing_dimensions?: string }
  | { type: "write"; report: string; citations?: Record<string, string>; remaining_gaps?: string[] }
  | { type: "research_planned"; tasks: { task_id: string; objective: string; status: string; revision: number }[] }
  | { type: "research_task_started"; task_id: string }
  | { type: "research_task_completed"; note: { task_id: string; evidence_ids: string[]; unknowns: string[]; conflicts: string[] } }
  | { type: "gap_reviewed"; covered: string[]; missing: string[]; unknowns: string[]; conflicts: string[] }
  | { type: "report_drafting"; phase: "started" | "completed"; claim_count?: number }
  | { type: "citation_verifying"; phase: "started" | "completed" }
  | { type: "report_revising"; revision: number }
  | { type: "report_completed"; citations: Record<string, string>; remaining_gaps: string[] }
  | { type: "stats"; total_calls: number; prompt_tokens: number; completion_tokens: number; total_tokens: number }
  | { type: "run_finished"; status: "completed" | "partial"; reason: string | null; remaining_gaps: string[] }
  | { type: "run_cancelled" | "run_interrupted" }
  | { type: "error"; message: string };

export function startRunEvents(
  runId: string,
  onEvent: (event: ResearchEvent) => void,
  terminalAfter = 0,
): () => void {
  const controller = new AbortController();
  let cursor = 0;
  let terminal = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let retries = 0;
  async function connect(): Promise<void> {
    try {
      const response = await fetch(`${API_BASE}/api/runs/${encodeURIComponent(runId)}/events`, {
        signal: controller.signal, headers: cursor ? { "Last-Event-ID": String(cursor) } : {},
      });
      if (!response.ok || !response.body) throw new Error(`事件流连接失败 (${response.status})`);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (!controller.signal.aborted) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop()!;
        for (const block of blocks) {
          let type = "", data = "", sequence = 0;
          for (const line of block.split("\n")) {
            if (line.startsWith("event: ")) type = line.slice(7);
            if (line.startsWith("data: ")) data = line.slice(6);
            if (line.startsWith("id: ")) sequence = Number(line.slice(4));
          }
          if (sequence && sequence <= cursor) continue;
          if (type && data) {
            const payload = JSON.parse(data);
            const isTerminal = ["run_finished", "run_cancelled", "run_interrupted", "error"].includes(type);
            // A resumed run replays its history; old interruption/error events are not its current end.
            if (!isTerminal || !sequence || sequence >= terminalAfter) {
              onEvent({ type, ...payload } as ResearchEvent);
              if (isTerminal) terminal = true;
            }
            if (sequence) cursor = sequence;
            retries = 0;
          }
        }
      }
    } catch (error) {
      if (controller.signal.aborted) return;
      if (retries >= 4) {
        onEvent({ type: "error", message: error instanceof Error ? error.message : "事件流连接失败，可从任务列表重新查看" });
        return;
      }
    }
    if (!terminal && !controller.signal.aborted) timer = setTimeout(() => { void connect(); }, Math.min(1000 * 2 ** retries++, 8000));
  }
  void connect();
  return () => { controller.abort(); clearTimeout(timer); };
}

export interface MemoryMessage {
  message_index: number; message_id: string | null; history_index: number | null;
  role: "user" | "assistant"; content: string; quote: string;
}

export interface MemorySource {
  kind: string; reference: string | null; quote: string; evidence_ids: string[];
  messages?: MemoryMessage[];
  relation?: "direct" | "accepted" | "rejected" | "modified";
  grounding_reason?: string | null;
  run_id: string | null; research_note_id: string | null; task_id: string | null; claim_id: string | null;
}

export interface MemoryExtractionJob {
  job_id: string; source_type: string; source_id: string;
  status: "pending" | "running" | "retryable" | "succeeded" | "failed";
  attempts: number; created_at: string; updated_at: string; error: string | null;
}

export interface SavedMemory {
  memory_id: string;
  content: string;
  memory_type: "preference" | "note" | "decision" | "background" | "research_note";
  type: SavedMemory["memory_type"];
  scope: "user" | "workspace" | "session";
  workspace_id: string | null;
  session_id: string | null;
  source: MemorySource;
  sources: MemorySource[];
  status: "active" | "superseded" | "deleted";
  importance: number;
  confidence: number;
  version: number;
  supersedes: string[];
  superseded_by: string | null;
  created_at: string;
  updated_at: string;
}

export interface SavedRun {
  execution?: { workspace_id: string; session_id: string | null };
  run_id: string;
  workflow: Workflow;
  query: string;
  status: "pending" | "running" | "cancel_requested" | "completed" | "partial" | "failed" | "cancelled" | "interrupted";
  result: { report?: string; audit_items?: unknown[] } | null;
  error: string | null;
  reason: string | null;
  remaining_gaps: string[];
  restart_required: boolean;
  event_cursor: number;
  consumed_budget: Record<string, number>;
}

export async function getRun(runId: string): Promise<SavedRun> {
  return (await expectJson<{ run: SavedRun }>(await fetch(`${API_BASE}/api/runs/${encodeURIComponent(runId)}`))).run;
}
export async function listRuns(): Promise<SavedRun[]> {
  return (await expectJson<{ runs: SavedRun[] }>(await fetch(`${API_BASE}/api/runs`))).runs;
}
export async function runAction(runId: string, action: "resume" | "restart" | "cancel"): Promise<SavedRun> {
  return (await expectJson<{ run: SavedRun }>(await fetch(`${API_BASE}/api/runs/${encodeURIComponent(runId)}/${action}`, { method: "POST" }))).run;
}
export async function auditReport(runId: string): Promise<SavedRun> {
  return (await expectJson<{ run: SavedRun }>(await fetch(`${API_BASE}/api/audits`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ report_run_id: runId }),
  }))).run;
}
export async function listMemories(workspaceId = "default", sessionId?: string): Promise<SavedMemory[]> {
  const params = new URLSearchParams({ workspace_id: workspaceId, include_inactive: "true" });
  if (sessionId) params.set("session_id", sessionId);
  return expectJson(await fetch(`${API_BASE}/api/memories?${params}`));
}
export async function saveMemory(content: string, type: SavedMemory["type"], memoryId?: string,
  scope: SavedMemory["scope"] = "workspace", workspaceId = "default", sessionId?: string): Promise<SavedMemory> {
  return expectJson(await fetch(`${API_BASE}/api/memories${memoryId ? `/${encodeURIComponent(memoryId)}` : ""}`, {
    method: memoryId ? "PUT" : "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, memory_type: type, scope, workspace_id: workspaceId, session_id: sessionId }),
  }));
}
export async function deleteMemory(memoryId: string): Promise<void> {
  await expectJson(await fetch(`${API_BASE}/api/memories/${encodeURIComponent(memoryId)}`, { method: "DELETE" }));
}
export async function extractMemories(input: { messages?: ChatMessage[]; run_id?: string; workspace_id?: string; session_id?: string }): Promise<MemoryExtractionJob> {
  return expectJson(await fetch(`${API_BASE}/api/memories/extract`, { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(input) }));
}

export async function listMemoryJobs(workspaceId: string, sessionId?: string): Promise<MemoryExtractionJob[]> {
  const params = new URLSearchParams({ workspace_id: workspaceId });
  if (sessionId) params.set("session_id", sessionId);
  return expectJson(await fetch(`${API_BASE}/api/memories/jobs?${params}`));
}
