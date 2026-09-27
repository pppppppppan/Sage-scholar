from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from uuid import uuid4

from sage_scholar.agents import Researcher, Supervisor, Writer
from sage_scholar.base import LLMClient
from sage_scholar.config import Config
from sage_scholar.domain.execution import ExecutionRequest
from sage_scholar.domain.research import ResearchContract
from sage_scholar.infrastructure.json_store import write_json
from .budget import ResearchBudget, BudgetedTool
from .evidence_context import ResearchEvidenceContext
from .gap_review import GapReviewer
from .verification import CitationVerifier
from sage_scholar.infrastructure.retrieval import LegacyRetrievalAdapter
from sage_scholar.tools.policy import EvidenceTool
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

    def __init__(self, config: Config, llm_client: LLMClient | None = None, pipeline: Pipeline | None = None, retrieval=None, connect_external: bool = True):
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
        self.retrieval = retrieval or LegacyRetrievalAdapter(config, pipeline=pipeline)
        self.mcp_clients = create_mcp_clients(os.path.join(config.config_dir, "mcp_servers.json")) if connect_external else []

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
        self.registry = registry

        with open(os.path.join(config.config_dir, "agents.json"), encoding="utf-8") as file:
            agent_configs = json.load(file)
        self.whitelist = agent_configs["researcher"]["allowed_tools"]

    def run_research(self, request: ExecutionRequest):
        contract = ResearchContract.from_request(
            request, max_rounds=self.config.max_rounds, max_steps=self.config.max_steps,
        )
        policy = contract.source_policy
        context = ResearchEvidenceContext(self.retrieval.evidence_store, policy, self.retrieval.documents)
        budget = ResearchBudget(contract.limits.max_tool_calls)
        observation_run = uuid4().hex

        def make_tools(task_id, observed_ids, issues):
            def record(evidence):
                context.register(evidence)
                observed_ids.add(evidence.evidence_id)

            def observe(observation):
                if observation.get("kind") == "error":
                    issues.append(f"{observation['tool']} 返回错误，来源不可访问")
                write_json(
                    Path(self.config.data_dir) / "observations" / observation_run / f"{uuid4().hex}.json",
                    {"task_id": task_id, **observation},
                )

            def wrap(tool):
                if tool.source_access == "public":
                    tool = EvidenceTool(tool, self.retrieval.evidence_store, record, observe)
                return BudgetedTool(tool, budget)

            return self.registry.for_policy(
                self.whitelist, policy,
                local_tools=[RAGTool(self.retrieval, policy, record)], wrap=wrap,
            )

        available = make_tools("capabilities", set(), [])
        yield {"policy_node": {
            "source_policy": policy.model_dump(mode="json"),
            "available_tools": [tool.name for tool in available],
        }}
        if not available:
            yield {"outcome_node": {"remaining_gaps": ["资料策略下没有可用工具，无法进行有证据的研究"]}}
            return

        review_llm = (
            LLMClient(model=self.config.llm.review_model, timeout=self.config.llm.timeout)
            if self.config.llm.review_model else self.llm_client
        )
        supervisor = Supervisor(
            llm=self.llm_client, context_builder=self.context_builder,
            max_steps=contract.limits.max_steps, review_llm=review_llm,
        )
        writer = Writer(
            llm=self.llm_client, context_builder=self.context_builder, evidence_context=context,
            temperature=self.config.llm.writer_temperature,
        )

        def create_researcher(task_id):
            observed_ids, issues = set(), []
            return Researcher(
                name=task_id, llm=self.llm_client, context_builder=self.context_builder,
                tool_list=make_tools(task_id, observed_ids, issues),
                max_steps=contract.limits.max_steps, temperature=self.config.llm.research_temperature,
                evidence_context=context, observed_ids=lambda: sorted(observed_ids),
                observed_issues=lambda: list(issues),
            )

        graph = build_graph(
            supervisor, create_researcher, writer,
            gap_reviewer=GapReviewer(review_llm, context),
            verifier=CitationVerifier(review_llm, context), budget=budget,
        )
        final_report = None
        for _mode, event in graph.stream(
            {"research_contract": contract},
            config={"recursion_limit": 20 + 3 * contract.limits.max_tasks * (contract.limits.max_rounds + 1)},
            stream_mode=["updates", "custom"],
        ):
            if "render_node" in event:
                final_report = event["render_node"]["report_result"]
            yield event
        yield {"evidence_node": {"evidence_ids": context.ids()}}
        yield {"outcome_node": {
            "remaining_gaps": final_report.remaining_gaps if final_report else ["研究未生成报告"],
            "tool_calls_used": budget.used,
        }}

    def create_library_manager(self) -> LibraryManager:
        return LibraryManager(
            data_dir=self.config.data_dir,
            paper_tool=self.paper_reader,
            pdfmux_tool=self.pdfmux_tool,
            pipeline=self.retrieval.pipeline,
        )

    def usage_snapshot(self) -> dict:
        return {
            "total_calls": self.llm_client.total_calls,
            "prompt_tokens": self.llm_client.total_prompt_tokens,
            "completion_tokens": self.llm_client.total_completion_tokens,
            "total_tokens": self.llm_client.total_prompt_tokens + self.llm_client.total_completion_tokens,
        }

    @staticmethod
    def to_public_events(event):
        events = []
        for node_name, output in event.items():
            if node_name == "progress_node":
                events.append((output["event"], {"phase": output["phase"]}))
            elif node_name in {"policy_node", "evidence_node", "outcome_node"}:
                name = {"policy_node": "source_policy", "evidence_node": "evidence", "outcome_node": "outcome"}[node_name]
                events.append((name, output))
            elif node_name == "plan_node":
                tasks = list(output.get("tasks", {}).values())
                events.append(("plan", {"sub_questions": [
                    {"label": task.label or task.task_id, "question": task.objective}
                    for task in tasks if task.status == "pending"
                ]}))
                events.append(("research_planned", {"tasks": [task.model_dump() for task in tasks]}))
            elif node_name == "dispatch_node":
                for identity in output["ready_tasks"]:
                    events.append(("research_task_started", {"task_id": identity}))
            elif node_name == "research_node":
                for note in output["research_notes"].values():
                    events.append(("research", {
                        "question": note.question, "preview": "; ".join(f.text for f in note.findings)[:150],
                        "tool_call_counts": note.tool_usage,
                        "status": output["tasks"][note.task_id].status,
                    }))
                    events.append(("research_task_completed", {"note": note.model_dump()}))
            elif node_name == "review_node":
                review = output["gap_review"]
                events.append(("review", {"round": output["refine_round"], "review_summary": [],
                                          "missing_dimensions": "; ".join(review.gaps)}))
                events.append(("gap_reviewed", review.model_dump()))
            elif node_name == "write_node":
                events.append(("report_drafting", {"phase": "completed", "claim_count": len(output["draft"].claims)}))
            elif node_name == "verify_node":
                events.append(("citation_verifying", {"phase": "completed", **output["verification_result"].model_dump()}))
            elif node_name == "revision_node":
                events.append(("report_revising", {"revision": output["writer_revision"]}))
            elif node_name == "render_node":
                result = output["report_result"].model_dump(mode="json")
                events.append(("write", result))
                events.append(("report_completed", {"citations": result["citations"],
                                                   "remaining_gaps": result["remaining_gaps"]}))
        return events

    @staticmethod
    def to_public_event(event):
        events = ResearchRuntime.to_public_events(event)
        return events[0] if events else None

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
        retrieval,
    ):
        self._config = config
        self._llm_client_factory = llm_client_factory
        self._retrieval = retrieval
        self._runtimes: dict[bool, ResearchRuntime] = {}
        self._lock = threading.Lock()

    def get(self, require_external: bool = True) -> ResearchRuntime:
        with self._lock:
            if require_external not in self._runtimes:
                self._runtimes[require_external] = ResearchRuntime(
                    self._config, llm_client=self._llm_client_factory(),
                    retrieval=self._retrieval, connect_external=require_external,
                )
            return self._runtimes[require_external]

    def close(self) -> None:
        for runtime in self._runtimes.values():
            runtime.close()
