from __future__ import annotations

import json
import logging
import os
import threading

from sage_scholar.agents import Researcher, Supervisor, Writer
from sage_scholar.base import LLMClient
from sage_scholar.config import Config
from sage_scholar.context import ContextBuilder, HistoryCompactor, TokenCounter
from sage_scholar.graph import build_graph
from sage_scholar.library import LibraryManager
from sage_scholar.mcp import create_mcp_clients, register_mcp_tools
from sage_scholar.rag import Pipeline
from sage_scholar.search import SearchTool
from sage_scholar.tools import PaperReaderTool, RAGTool, ToolRegistry

logger = logging.getLogger(__name__)


class ResearchRuntime:
    """Composition root for migrated SAGE Research capabilities."""

    def __init__(self, config: Config, llm_client: LLMClient | None = None, pipeline: Pipeline | None = None):
        self.config = config
        self.llm_client = llm_client or LLMClient(model=config.llm.model, timeout=config.llm.timeout)
        token_counter = TokenCounter()
        history_compactor = HistoryCompactor(llm_client=self.llm_client, token_counter=token_counter)
        self.context_builder = ContextBuilder(
            history_compactor=history_compactor,
            token_counter=token_counter,
            reserve_ratio=config.context.reserve_ratio,
            max_tokens=config.context.max_tokens,
        )
        self.rag_pipeline = pipeline or Pipeline(data_dir=config.data_dir, llm_client=self.llm_client)
        self.mcp_clients = create_mcp_clients(os.path.join(config.config_dir, "mcp_servers.json"))

        registry = ToolRegistry()
        register_mcp_tools(registry=registry, clients=self.mcp_clients)
        brave = registry.tools.get("mcp__brave-search__brave_web_search")
        tavily = registry.tools.get("mcp__tavily__tavily_search")
        if brave is not None or tavily is not None:
            registry.register_tool(SearchTool(brave_tool=brave, tavily_tool=tavily))

        download = registry.tools.get("mcp__paper-search__download_arxiv")
        read = registry.tools.get("mcp__paper-search__read_arxiv_paper")
        self.paper_reader = PaperReaderTool(download_tool=download, read_tool=read) if download and read else None
        if self.paper_reader:
            registry.register_tool(self.paper_reader)
        self.pdfmux_tool = registry.tools.get("mcp__pdfmux__convert_pdf")
        registry.register_tool(RAGTool(pipeline=self.rag_pipeline))

        with open(os.path.join(config.config_dir, "agents.json"), encoding="utf-8") as file:
            agent_configs = json.load(file)
        whitelist = [name for name in agent_configs["researcher"]["allowed_tools"] if name in registry.tools]
        unavailable = sorted(set(agent_configs["researcher"]["allowed_tools"]) - set(whitelist))
        if unavailable:
            logger.warning("[Tools] unavailable tools omitted: %s", ", ".join(unavailable))
        self.researcher_tools = registry.get_tools(whitelist)

    def run_research(self, brief: str):
        review_llm = (
            LLMClient(model=self.config.llm.review_model, timeout=self.config.llm.timeout)
            if self.config.llm.review_model
            else None
        )
        supervisor = Supervisor(
            llm=self.llm_client,
            context_builder=self.context_builder,
            max_steps=self.config.max_steps,
            review_llm=review_llm,
        )
        writer = Writer(
            llm=self.llm_client,
            context_builder=self.context_builder,
            temperature=self.config.llm.writer_temperature,
        )

        def create_researcher(researcher_id: str = "R-?"):
            return Researcher(
                name=researcher_id,
                llm=self.llm_client,
                context_builder=self.context_builder,
                tool_list=self.researcher_tools,
                max_steps=self.config.max_steps,
                temperature=self.config.llm.research_temperature,
            )

        graph = build_graph(
            supervisor,
            create_researcher,
            writer,
            max_rounds=self.config.max_rounds,
        )
        yield from graph.stream({"research_brief": brief})

    def create_library_manager(self) -> LibraryManager:
        return LibraryManager(
            data_dir=self.config.data_dir,
            paper_tool=self.paper_reader,
            pdfmux_tool=self.pdfmux_tool,
            pipeline=self.rag_pipeline,
        )

    def usage_snapshot(self) -> dict:
        return {
            "total_calls": self.llm_client.total_calls,
            "prompt_tokens": self.llm_client.total_prompt_tokens,
            "completion_tokens": self.llm_client.total_completion_tokens,
            "total_tokens": self.llm_client.total_prompt_tokens + self.llm_client.total_completion_tokens,
        }

    @staticmethod
    def to_public_event(event: dict) -> tuple[str, dict] | None:
        for node_name, output in event.items():
            if node_name == "plan_node":
                return "plan", {
                    "sub_questions": [
                        {"label": item.label, "question": item.question}
                        for item in output["sub_questions"]
                    ]
                }
            if node_name == "research_node":
                question, note = output["pending_review_pairs"][0]
                return "research", {
                    "question": question,
                    "preview": note[:150] + ("..." if len(note) > 150 else ""),
                    "tool_call_counts": output.get("tool_call_counts", {}),
                }
            if node_name == "review_node":
                return "review", {
                    "round": output["refine_round"],
                    "review_summary": output["review_summary"],
                    "missing_dimensions": output.get("missing_dimensions", ""),
                }
            if node_name == "write_node":
                return "write", {"report": output["final_report"]}
        return None

    def close(self) -> None:
        if self.paper_reader:
            self.paper_reader.cleanup()
        for client in self.mcp_clients:
            client.disconnect()


class ResearchRuntimeProvider:
    def __init__(
        self,
        config: Config,
        llm_client_factory,
        pipeline_factory,
    ):
        self._config = config
        self._llm_client_factory = llm_client_factory
        self._pipeline_factory = pipeline_factory
        self._runtime: ResearchRuntime | None = None
        self._lock = threading.Lock()

    def get(self) -> ResearchRuntime:
        if self._runtime is None:
            with self._lock:
                if self._runtime is None:
                    self._runtime = ResearchRuntime(
                        self._config,
                        llm_client=self._llm_client_factory(),
                        pipeline=self._pipeline_factory(),
                    )
        return self._runtime

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
