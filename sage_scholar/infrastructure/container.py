from __future__ import annotations

from sage_scholar.application.direct_service import DirectService
from sage_scholar.application.document_resolver import DocumentResolver
from sage_scholar.application.document_service import DocumentService
from sage_scholar.application.qa_service import QAService
from sage_scholar.application.query_service import QueryService
from sage_scholar.application.research_service import ResearchService
from sage_scholar.config import Config
from sage_scholar.research.runtime import ResearchRuntimeProvider
from sage_scholar.routing import QueryRouter, Workflow
from sage_scholar.runtime import RunManager
from sage_scholar.runtime.store import RunStore
from sage_scholar.runtime.checkpoints import Checkpoints
from sage_scholar.context.manager import ContextManager
from sage_scholar.memory import MemoryStore, MemoryService, MemoryRecall
from sage_scholar.rag.embedding import Embedding
from sage_scholar.memory.background import MemoryBackground
from sage_scholar.memory.jobs import MemoryJobStore
from sage_scholar.audit import CitationAuditService
from sage_scholar.audit.fetch import AuditFetch

from .model_gateway import LegacyModelGateway
from .retrieval import LegacyRetrievalAdapter


class ApplicationContainer:
    """Single-process composition root. Heavy legacy resources stay lazy."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.memory_store = MemoryStore(self.config.data_dir)
        self.run_store = RunStore(self.config.data_dir)
        self.run_store.conversations.adopt_existing_runs()
        self.checkpoints = Checkpoints(self.config.data_dir)
        self.context_manager = ContextManager(self.config.data_dir, max_tokens=self.config.context.max_tokens,
            reserve_ratio=self.config.context.reserve_ratio)
        self.context_manager.run_store = self.run_store
        self.model_gateway = LegacyModelGateway(self.config)
        self.model_gateway.context_manager = self.context_manager
        self.retrieval = LegacyRetrievalAdapter(self.config)
        self.memory_service = MemoryService(self.memory_store,
            MemoryRecall(self.config.data_dir, embedding_factory=Embedding),
            llm_factory=lambda: self.model_gateway.client)
        self.context_manager.memories = self.memory_service

        self.evidence_store = self.retrieval.evidence_store
        self.memory_jobs = MemoryJobStore(self.run_store)
        self.memory_background = MemoryBackground(self.memory_service, self.retrieval, self.memory_jobs, self.checkpoints)
        self.run_manager = RunManager(max_concurrent_runs=1, store=self.run_store, checkpoints=self.checkpoints,
            on_completed=self.memory_background.completed)
        self.runtime_provider = ResearchRuntimeProvider(
            self.config,
            llm_client_factory=lambda: self.model_gateway.client,
            retrieval=self.retrieval,
            store=self.run_store, checkpoints=self.checkpoints, context_manager=self.context_manager,
        )
        self.research_service = ResearchService(self.runtime_provider)
        self.audit_service = CitationAuditService(self.retrieval, lambda: self.model_gateway.client,
            AuditFetch(self.runtime_provider, self.retrieval, self.run_store).for_request,
            checkpoints=self.checkpoints, run_lookup=self.run_manager.get)
        self.runners = {Workflow.DEEP_RESEARCH: self.research_service, Workflow.CITATION_AUDIT: self.audit_service}
        self.run_manager.register_runners(self.runners)
        self.document_service = DocumentService(
            self.runtime_provider,
            self.config.data_dir,
            pipeline_factory=lambda: self.retrieval.pipeline,
        )
        self.query_service = QueryService(
            router=QueryRouter(),
            document_resolver=DocumentResolver(self.retrieval.documents),
            direct=DirectService(self.model_gateway, context_manager=self.context_manager),
            qa=QAService(self.retrieval, self.model_gateway, context_manager=self.context_manager),
            run_manager=self.run_manager,
            long_runners=self.runners,
            memory_store=self.memory_service,
            memory_background=self.memory_background,
        )

        self.runners.update({Workflow.DIRECT: self.query_service.inline_runner, Workflow.QA: self.query_service.inline_runner})
        from sage_scholar.conversations.service import ConversationService
        self.conversations = ConversationService(self.run_store.conversations, self.run_manager,
            self.query_service, self.memory_background)

    async def close(self) -> None:
        await self.run_manager.shutdown()
        await self.memory_background.close()
        self.runtime_provider.close()
        self.checkpoints.close()
        self.run_store.close()
