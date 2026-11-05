# Architecture

Sage Scholar is a single-process modular monolith. FastAPI validates HTTP requests; QueryService builds the execution request; RunManager owns lifecycle, background tasks, cancellation and persisted SSE events. LangGraph owns workflow state and checkpoints, not a second scheduler in RunManager.

The entry QueryGraph uses one local Ollama classification call, then SourceResolver and PolicyValidator resolve actual documents and enforce permissions. Conditional edges select Direct, document QA, Clarify (interrupt/resume), or the Deep Research subgraph. All are associated with persisted conversation/Run state.

Deep Research is Supervisor → parallel Researchers → Supervisor Review → Writer. Planning favors the smallest sufficient set of independent questions. Review can request bounded follow-ups. There is no mandatory Citation Audit workflow, separate synthesis/outline/reviewer chain, or Paper2Code implementation.

Shared services:

- **ContextManager:** agent-specific context budgets, protected constraints, history compression, externalized tool results and evidence rereading.
- **Memory:** persisted, scoped research background and preferences; relevant recall rather than injecting all memories. Memory is never citation evidence.
- **EvidenceStore and document catalog:** stable source identity, document versions, original text, page/section/element provenance.
- **Skills/MCP:** lazy instructions and capability-scoped external/internal tools with SourcePolicy enforced.
- **SQLite:** RunStore/events/conversations/tool journal and separate LangGraph checkpoints; file stores retain sources, indexes and memory artifacts.
- **Observability:** optional metadata-only traces, separate from authoritative execution state.

Docker adds a streaming same-origin HTTP bridge in Next.js, one FastAPI worker, and optional Ollama. It does not change workflow topology, retrieval, checkpoint serialization or algorithms. The Router's additional host allowlist is deployment configuration only.

For earlier design rationale and migration decisions, see [historical implementation notes](implementation-notes.md), [repository audit](repository-audit.md) and [migration map](migration-map.md). Historical documents describe their own implementation dates, not necessarily current limitations.
