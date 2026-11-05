# Migration Map

Source commit: `f15917913227196eae3f052360998baa7fa418c3`. `sage-research` remains untouched.

| SAGE Research source | SAGE Scholar target | Type | Reason / change |
|---|---|---|---|
| `web/app/globals.css`, icons | same paths under `web/` | reuse | Visual system retained; local system fonts replace build-time Google font downloads. |
| `web/components/StreamingText.tsx` | same | reuse | Animation retained; hook dependency corrected. |
| `web/components/ReportView.tsx` | same | adapt | Report UI retained; lint issue fixed. |
| `web/components/ResearchProgress.tsx` | same | adapt | Subscribes to `/api/runs/{id}/events` instead of owning `/api/research`. |
| `web/components/LibraryDrawer.tsx` | same | reuse | Existing upload/list/delete interaction retained. |
| `web/components/ClarifyPanel.tsx` | removed | replace | Universal clarification conflicts with workflow router. |
| `web/app/page.tsx`, `web/lib/api.ts` | same | replace/adapt | Unified query response chooses inline answer or long Run. |
| `sage_research/api/app.py`, `api/schemas.py` | `sage_scholar/api/*` | replace | API no longer executes research logic directly. |
| `sage_research/orchestrator.py` | `sage_scholar/infrastructure/container.py`, `research/runtime.py` | replace/adapt | Split application composition from research-only legacy assembly; heavy resources are lazy. |
| `sage_research/agents/{supervisor,researcher,writer}.py` | `sage_scholar/agents/*` | adapt | Research behavior retained; invoked only by ResearchRuntime. |
| `sage_research/agents/clarifier.py` | `sage_scholar/routing/*` | replace | Specificity checking is no longer the application router; the obsolete target copy was removed. |
| `sage_research/agents/prompts.py` | `sage_scholar/agents/prompts.py` | reuse | Existing research prompts retained for baseline behavior. |
| `sage_research/graph/graph_builder.py` | `sage_scholar/graph/graph_builder.py` | adapt | Existing graph retained as baseline; RunManager owns lifecycle, graph owns workflow. |
| `sage_research/base/*` | `sage_scholar/base/*` | adapt | Existing provider compatibility retained behind `LegacyModelGateway`. |
| `sage_research/context/*` | `sage_scholar/context/*` | adapt | Current compaction preserved; future Context Manager contract remains separate. |
| `sage_research/tools/*` | `sage_scholar/tools/*` | adapt | Registry and tools retained; unavailable tools are omitted rather than blocking all research. |
| `sage_research/mcp/*` | `sage_scholar/mcp/*` | adapt | MCP adapter retained; missing config or failed optional server no longer prevents all startup. |
| `sage_research/search/*` | `sage_scholar/search/*` | reuse/adapt | Brave/Tavily fallback retained behind research runtime. |
| `sage_research/rag/*` | `sage_scholar/rag/*`, `infrastructure/retrieval.py` | adapt | Adds document-scope filtering and structured `EvidenceRef`; storage is still JSON. |
| `sage_research/library/*` | `sage_scholar/library/*`, `application/document_service.py` | adapt | HTTP concerns removed from callers; PDF/arXiv still reuse existing converters. |
| `configs/agents.json`, `configs/mcp_servers.json` | same | adapt | Existing tool configuration retained with optional environment handling. |
| `requirements.txt` | `requirements.txt`, `pyproject.toml` | replace | Removes CUDA-specific Torch pin, pins MCP v1 API compatibility, adds API/test requirements. |
| `setup.sh` | `setup.sh` | replace | Uses current Python environment, preserves secrets, starts the modular application. |
| `.env`, `data/*`, logs, caches | not migrated | remove | User secrets and runtime/generated state are not copied. |

## New SAGE Scholar modules

- `routing/models.py`, `routing/router.py`: workflow/source/depth decision.
- `application/query_service.py`: one application entry point.
- `application/direct_service.py`: simple answer and local date behavior.
- `application/qa_service.py`: evidence-grounded local document answer.
- `application/document_service.py`: document use-case facade.
- `application/research_service.py`: synchronous graph to async Run adapter.
- `runtime/models.py`, `runtime/run_manager.py`: task lifecycle and replayable in-memory events.
- `domain/evidence.py`: shared evidence contract.
- `infrastructure/model_gateway.py`, `retrieval.py`, `container.py`: legacy adapters and dependency composition.
- `web/components/InlineAnswer.tsx`: direct/QA result and evidence display.

## 2026-09-25 incremental source-policy/evidence iteration
| Original/reference | Scholar target | Type | Reason |
| --- | --- | --- | --- |
| Existing Scholar Router/QueryService | domain/execution.py; application/query_service.py | adapt/new | Preserve route fields through execution. |
| sage_research/tools/registry.py, tool_rag.py | tools/registry.py, tool_rag.py, policy.py | adapt | Run-local tool filtering, enforcement, shared retrieval/evidence. |
| sage_research/rag/* | rag/*; infrastructure/retrieval.py | reuse/adapt | Keep hybrid algorithm; resolve stable document IDs and read sections. |
| sage_research/library/library_manager.py | library/library_manager.py; infrastructure/document_store.py | adapt | UUID identity, explicit status, version-isolated ingestion, rollback. |
| sage_research/library/converter.py | library/converter.py | reuse | Continue existing PDF/arXiv/Markdown conversion. |
| Existing Scholar EvidenceRef | domain/evidence.py; infrastructure/evidence_store.py | adapt/new | Stable evidence IDs, version/location, persistent original text. |
| sage_research/agents/*, graph/graph_builder.py | agents/*, graph/graph_builder.py | reuse/adapt | Roles retained; only add missing-review gap handling in graph. |
| Existing Scholar api/app.py | api/routes/*, api/dependencies.py | adapt | Preserve URLs; split HTTP responsibilities. |
| Existing Scholar RunManager | runtime/*, application/research_service.py | adapt | Full request contract and partial outcome. |
| Existing Scholar QA | application/{qa_service,reading,citations}.py | adapt/new | Whole-document reading and identity-based citation rendering. |
| sage_research web UI | web/components/{LibraryDrawer,ResearchProgress,InlineAnswer}.tsx | adapt | Stable document ID operations, partial status and evidence display. |

## 2026-09-26 V2.1 identity/version/status patch
| Prior Scholar implementation | Target | Type | Reason |
| --- | --- | --- | --- |
| domain/documents.py | same | adapt | Separate logical identity, active version, input hash and successful version history. |
| library/library_manager.py | same | adapt | Stable source/explicit ID matching; independent ingestion lock; publish only indexed versions. |
| infrastructure/document_store.py | same | adapt | Migrate old single-version catalogs while preserving document/evidence identity. |
| infrastructure/retrieval.py | same | adapt | Retrieve active paths even when latest ingestion attempt fails. |
| rag/vector_store.py | same + existing json_store.write_json | adapt/reuse | Atomic index file replacement, existing retrieval algorithms unchanged. |
| application/document_service.py, api library/schema | same | adapt | Optional document_id update target; upload identity independent of temporary path. |


## 2026-09-26 V3 Research Evidence 化

| Source / prior implementation | Scholar target | Type | Reason |
| --- | --- | --- | --- |
| sage_research/agents/researcher.py → Scholar V2.1 | agents/researcher.py | adapt | ReAct/history loop reused; replace free-text handoff with validated ResearchNote; source text no longer model-denoised. |
| sage_research/agents/supervisor.py → Scholar V2.1 | agents/supervisor.py | adapt | Keep planning parser/history; add stable task IDs/dependencies and Contract adapter. |
| sage_research/agents/writer.py → Scholar V2.1 | agents/writer.py | replace boundary | Remove Sources/URL dedup/numbering; compose typed Claims using selected Evidence originals. |
| sage_research/graph/graph_builder.py → Scholar V2.1 | graph/graph_builder.py | adapt | Keep StateGraph/Send parallelism; typed task/Note boundaries, gap review, bounded revisions and renderer nodes. |
| Existing Scholar domain/execution.py, domain/evidence.py | same + domain/research.py | reuse/new | ExecutionRequest/SourcePolicy/EvidenceRef stay authoritative; add research-only types. |
| Existing Scholar tools/policy.py | same; tools/base_tool.py; mcp/tool_adapter.py; tools/tool_paper.py | adapt | Only explicit source readers promote observations; errors/status/snippets remain observations. |
| Existing Scholar retrieval/evidence/version storage | same + research/evidence_context.py | reuse/new adapter | Original evidence/version APIs retained; add run-scoped authorization and bounded reread. |
| Existing Scholar research/runtime.py | same | adapt | Keep dependency composition and per-policy tools; add note evidence registry, shared call budget, Observation externalization, public events. |
| Existing Scholar application/research_service.py | same | adapt | Pass complete report/citation result and multiple public events into existing Run lifecycle. |
| Existing Scholar RunManager / API / Direct / QA / ingestion | same | reuse | No new scheduler, API routes, database or report ingestion. |
| Existing Scholar web/lib/api.ts, ResearchProgress.tsx | same | adapt | Retain timeline/terminal event behavior, show drafting/verification/revision and failed task state. |
| No old equivalent | research/{notes,structured,tasks,budget,gap_review,verification,renderer}.py | new | Focused research boundary components rather than another workflow engine. |

Full behavior, tests and known limits: `docs/v3-research-evidence.md`.

## V4 增量（2026-09-28）

| 来源 | Scholar 路径 | 类型 | 说明 |
|---|---|---|---|
| Scholar V3 research/verification.py | audit/service.py、audit/graph.py | reuse/adapt | 审查工作流调用原 CitationVerifier；没有复制第二套核验 |
| Scholar V3 research/evidence_context.py | 原路径、context/manager.py | adapt | 保留身份/版本/策略检查，新增共享 token 窗口选择 |
| SAGE Research base/llm_client.py、context/token_counter.py | base/llm_client.py、context/manager.py | adapt/reuse | 现有 SDK 和 tokenizer，统一完整请求预算 |
| Scholar context/context_builder.py | 原路径 | adapt | 兼容既有 Agent，最终预算和压缩集中到统一调用入口 |
| Scholar runtime/run_manager.py | 原路径、runtime/store.py、checkpoints.py、journal.py | adapt/new | 生命周期 SQLite、官方图 checkpoint、结果复用边界 |
| Scholar graph/graph_builder.py | 原路径 | adapt | 注入 checkpointer；中断传播；主链不改 |
| 无独立旧实现 | memory/、api/routes/memory.py | new | 明确确认的 Memory CRUD，不进入 Evidence |
| Scholar Web API/ResearchProgress | web/lib/api.ts、ResearchProgress.tsx、WorkspacePanel.tsx | adapt/new | SSE 重连/回放、任务恢复入口、Memory 管理 |

完整变更与来源借鉴边界见 v4-delivery.md；本轮没有修改 sage-research 源目录。
