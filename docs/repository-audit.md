# SAGE Research Repository Audit

Audit source: `/Users/pan/Desktop/1/sage-research`, commit `f15917913227196eae3f052360998baa7fa418c3`, including the user's uncommitted local compatibility fixes. The source repository was read only.

## 1. Actual structure

- `sage_research/api`: one FastAPI module with clarification, research SSE and library endpoints.
- `sage_research/orchestrator.py`: composition root for LLM, context, RAG, MCP tools, research agents and document manager.
- `sage_research/graph`: one LangGraph research graph.
- `sage_research/agents`: Clarifier, Supervisor, Researcher and Writer. Researcher owns a synchronous ReAct loop.
- `sage_research/context`: token counting and LLM-based history compaction.
- `sage_research/tools`, `mcp`, `search`: tool protocol/registry, MCP stdio clients and Brave/Tavily adapter.
- `sage_research/rag`, `library`: in-memory/JSON hybrid retrieval and document conversion/index management.
- `web`: Next.js single-page research UI, clarification view, progress timeline, report view and library drawer.
- No application service layer, SQL database, conversation model, persistent run manager, LangGraph checkpointer, Docker configuration or tests were present in the tracked source.

## 2. Original request chain

```text
web/app/page.tsx
  → ClarifyPanel
  → POST /api/clarify
  → Clarifier.analyze (LLM)
  → optional POST /api/clarify/refine
  → ResearchProgress
  → POST /api/research (one long-lived HTTP request)
  → Orchestrator.run_research
  → LangGraph.stream
  → Supervisor.plan
  → parallel Researcher.run / direct tool calls
  → Supervisor.review
  → retry or replan
  → Writer.run
  → Pipeline.add_text(report)
  → FastAPI converts graph node output to SSE
  → frontend timeline and report
```

The same path was used for every submitted query. Closing the HTTP connection aborted observation of the work; there was no independent run identity or replayable event store.

At application startup, `Orchestrator` created the embedding/reranker, connected configured MCP subprocesses and built the tool registry. This made optional integration failures capable of preventing the whole API from starting.

Document upload synchronously read the whole file, converted PDFs through `pdfmux`, embedded all chunks, updated `vector_store.json`, and updated `index.json`. The UI listed, uploaded and deleted files, but it did not provide a real PDF evidence viewer or stable page/section evidence identifiers.

## 3. Existing capability check

| Capability | Existing state |
|---|---|
| Router | No. Clarifier only decides whether a query is specific enough for research. |
| Research Graph | Yes. Plan → parallel research → review → retry/replan → write. |
| Tool Executor | No unified executor. `Researcher` validates tool name then calls `tool.run_tool` directly. |
| Tool registry / MCP | Yes, synchronous registry and stdio MCP adapters. Lifecycle/error behavior needs adaptation. |
| RAG | Yes, local JSON chunks with BM25 + cosine RRF + cross-encoder reranking. No source-scope filter in the original. |
| Document parsing | Yes, Markdown/text copy and PDF/arXiv conversion through MCP. Processing is synchronous and has no visible state machine. |
| SSE | Yes, but tied to the original `/api/research` response and not replayable. |
| Database / sessions | No. Library and vector data use JSON/files; there is no conversation/session persistence. |
| Checkpoint / resume | No checkpointer is passed to `graph.compile()`. |
| Tests | No tracked backend or frontend tests. |
| Docker / Compose | No tracked Dockerfile or compose file. GitHub MCP may itself be launched through Docker. |

## 4. Reuse decisions

### A. Direct reuse

- Frontend styles, `StreamingText`, most report rendering and most research progress presentation.
- Base `Message`, logging setup, tool schema/registry shape, search result model.
- Prompt content and the broad Supervisor/Researcher/Writer responsibilities.

### B. Adapt

- Research graph and agents: retained behind `ResearchService`; later work must add durable boundaries and evidence-aware state.
- Context: retained, but the current implementation budgets only system prompt plus message content and mutates history summaries. It needs complete-request accounting and structured evidence reread.
- LLM client: wrapped by a gateway; per-request usage, async provider calls, rate limits and tracing still need implementation.
- RAG: retained and extended with source filtering and structured evidence output; JSON storage remains transitional.
- MCP/search/library: retained as infrastructure adapters and initialized lazily; they no longer own API routing.
- Frontend: changed from Clarifier-first to unified query routing.

### C. Replace

- `api/app.py` and API schemas: replaced by unified Query, Run and Library APIs.
- `Orchestrator` as a god object: replaced by an application container plus a research-only runtime.
- Clarifier as the universal entry: replaced by `QueryRouter`; clarification is a route result.
- Connection-bound research SSE: replaced by RunManager-owned execution and run event subscription.
- Unstructured RAG strings as the application contract: replaced by `EvidenceRef` at the QA boundary.

### D. Remove / not needed now

- Multi-user authentication, RBAC, quotas, distributed workers and message queues.
- Original `ClarifyPanel` in the active UI.
- Runtime logs, `.next`, `node_modules`, `.env`, local indexes and other generated source-repository state.
- Source setup assumptions that force Python 3.14 or CUDA-specific Torch.

## 5. Target layer mapping

| Layer | Modules | Responsibility |
|---|---|---|
| API | `sage_scholar/api` | HTTP/SSE schemas and validation only |
| Application | `sage_scholar/application` | Query dispatch, Direct, QA, Document and Research use cases |
| Routing | `sage_scholar/routing` | Workflow, source scope, execution mode and clarification decision |
| Runtime | `sage_scholar/runtime` | In-process task lifecycle, cancellation flag, event replay and status |
| Workflow | `sage_scholar/graph`, `agents`, `research` | LangGraph research state and migrated agent behavior |
| Retrieval / Evidence | `sage_scholar/rag`, `infrastructure/retrieval.py`, `domain/evidence.py` | Retrieval implementation and stable application result shape |
| Model / Tools | `base`, `tools`, `mcp`, `search`, `infrastructure/model_gateway.py` | Provider calls and external capabilities |
| Persistence | current JSON/files behind services | Transitional implementation; PostgreSQL repositories are not claimed as implemented |

## 6. Main risks found

- Original LLM usage counters are process totals, not request totals.
- Original graph has no checkpointer; the first slice provides run lifecycle but does not claim restart-safe resume.
- Tool calls are synchronous and may not stop until the current call returns after cancellation.
- Existing RAG chunks lack page/section/parser version, so first-slice evidence is chunk-level and marked `legacy-markdown`.
- PDF parsing relies on an external MCP command and is unavailable if that service cannot start.
- The original Writer renumbers textual citations; it does not verify semantic claim support.
- Source repository had local changes for optional MCP configuration and macOS Torch compatibility. The new project preserves the useful behavior without editing the source.
