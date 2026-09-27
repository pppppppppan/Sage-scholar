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
    const detail = await response.text();
    throw new Error(detail || `请求失败 (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export async function submitQuery(
  query: string,
  history: ChatMessage[] = [],
): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE}/api/queries`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      history,
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
): () => void {
  const controller = new AbortController();
  fetch(`${API_BASE}/api/runs/${encodeURIComponent(runId)}/events`, {
    signal: controller.signal,
  })
    .then((response) => {
      if (!response.ok || !response.body) {
        throw new Error(`事件流连接失败 (${response.status})`);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      function pump(): Promise<void> {
        return reader.read().then(({ done, value }) => {
          if (done) return;
          buffer += decoder.decode(value, { stream: true });
          const blocks = buffer.split("\n\n");
          buffer = blocks.pop()!;
          for (const block of blocks) {
            let eventType = "";
            let data = "";
            for (const line of block.split("\n")) {
              if (line.startsWith("event: ")) eventType = line.slice(7);
              else if (line.startsWith("data: ")) data = line.slice(6);
            }
            if (
              eventType &&
              data &&
              ["plan", "research", "review", "write", "stats", "run_finished", "run_cancelled", "run_interrupted", "error",
               "research_planned", "research_task_started", "research_task_completed", "gap_reviewed",
               "report_drafting", "citation_verifying", "report_revising", "report_completed"].includes(eventType)
            ) {
              onEvent({ type: eventType, ...JSON.parse(data) } as ResearchEvent);
            }
          }
          return pump();
        });
      }
      return pump();
    })
    .catch((error: Error) => {
      if (error.name !== "AbortError") {
        onEvent({ type: "error", message: error.message });
      }
    });
  return () => controller.abort();
}
