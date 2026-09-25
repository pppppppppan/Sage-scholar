from __future__ import annotations

from sage_scholar.application.direct_service import DirectService
from sage_scholar.application.document_service import DocumentService
from sage_scholar.application.qa_service import QAService
from sage_scholar.application.query_service import QueryService
from sage_scholar.application.research_service import ResearchService
from sage_scholar.config import Config
from sage_scholar.research import ResearchRuntimeProvider
from sage_scholar.routing import QueryRouter, Workflow
from sage_scholar.runtime import RunManager

from .model_gateway import LegacyModelGateway
from .retrieval import LegacyRetrievalAdapter


class ApplicationContainer:
    """Single-process composition root. Heavy legacy resources stay lazy."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.model_gateway = LegacyModelGateway(self.config)
        self.retrieval = LegacyRetrievalAdapter(self.config)
        self.run_manager = RunManager(max_concurrent_runs=1)
        self.runtime_provider = ResearchRuntimeProvider(
            self.config,
            llm_client_factory=lambda: self.model_gateway.client,
            pipeline_factory=lambda: self.retrieval.pipeline,
        )
        self.research_service = ResearchService(self.runtime_provider)
        self.document_service = DocumentService(
            self.runtime_provider,
            self.config.data_dir,
            pipeline_factory=lambda: self.retrieval.pipeline,
        )
        self.query_service = QueryService(
            router=QueryRouter(),
            direct=DirectService(self.model_gateway),
            qa=QAService(self.retrieval, self.model_gateway),
            run_manager=self.run_manager,
            long_runners={Workflow.DEEP_RESEARCH: self.research_service},
        )

    async def close(self) -> None:
        await self.run_manager.shutdown()
        self.runtime_provider.close()
