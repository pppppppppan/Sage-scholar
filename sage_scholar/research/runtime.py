from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from uuid import uuid4
from sage_scholar.context.tools import ContextReadTool
from sage_scholar.runtime.control import check_control
from sage_scholar.runtime.interaction import assess, resume_command
from sage_scholar.runtime.human import public_interrupts
from .continuation import continue_graph
from sage_scholar.runtime.journal import DurableResearcher, JournaledTool, WorkflowInterrupted

from sage_scholar.agents import Researcher, Supervisor, Writer
from sage_scholar.base import LLMClient
from sage_scholar.config import Config
from sage_scholar.domain.execution import ExecutionRequest
from sage_scholar.domain.research import ResearchContract
from sage_scholar.infrastructure.json_store import write_json
from .budget import ResearchBudget, BudgetedTool
from .evidence_context import ResearchEvidenceContext
from .gap_review import GapReviewer
from .synthesis import ResearchSynthesis
from .report_review import ReportReviewer
from sage_scholar.infrastructure.retrieval import LegacyRetrievalAdapter
from sage_scholar.tools.policy import EvidenceTool
from sage_scholar.graph import build_graph
from sage_scholar.library import LibraryManager
from sage_scholar.mcp import create_mcp_clients, register_mcp_tools
from sage_scholar.mcp.loader import CAPABILITY_SERVERS, RESEARCH_SERVERS, iter_mcp_clients
from sage_scholar.rag import Pipeline
from sage_scholar.search import SearchTool
from sage_scholar.tools import PaperReaderTool, RAGTool, ToolRegistry

logger = logging.getLogger(__name__)


class ResearchRuntime:
    """Composition root for migrated SAGE Research capabilities."""

    def __init__(self, config: Config, llm_client: LLMClient | None = None, pipeline: Pipeline | None = None, retrieval=None, connect_external: bool = True, store=None, checkpoints=None, context_manager=None):
        self.config = config
        self.store, self.checkpoints, self.context_manager = store, checkpoints, context_manager
        self.llm_client = llm_client or LLMClient(model=config.llm.model, timeout=config.llm.timeout, context_manager=context_manager)
        self.retrieval = retrieval or LegacyRetrievalAdapter(config, pipeline=pipeline)
        self.mcp_clients = []
        self.registry = ToolRegistry()
        self.paper_reader = self.pdfmux_tool = None
        self._connect_external = connect_external
        self._tools_initialized = not connect_external
        self._tools_lock = threading.Lock()
        self.tool_statuses = []

        with open(os.path.join(config.config_dir, "agents.json"), encoding="utf-8") as file:
            agent_configs = json.load(file)
        self.whitelist = agent_configs["researcher"]["allowed_tools"]

    def _initialize_research_tools(self):
        with self._tools_lock:
            if not self._tools_initialized:
                for status in self.tool_statuses:
                    yield {"initialization_node": status}
                remaining = RESEARCH_SERVERS - {status["server"] for status in self.tool_statuses}
                for status, client in iter_mcp_clients(
                    os.path.join(self.config.config_dir, "mcp_servers.json"),
                    server_names=remaining, connect_timeout=self.config.mcp_connect_timeout,
                ):
                    self.tool_statuses.append(status)
                    if client is not None:
                        self.mcp_clients.append(client)
                        register_mcp_tools(self.registry, [client])
                    check_control()
                    yield {"initialization_node": status}
                brave = self.registry.tools.get("mcp__brave-search__brave_web_search")
                tavily = self.registry.tools.get("mcp__tavily__tavily_search")
                if brave or tavily:
                    self.registry.register_tool(SearchTool(brave_tool=brave, tavily_tool=tavily))
                download = self.registry.tools.get("mcp__paper-search__download_arxiv")
                read = self.registry.tools.get("mcp__paper-search__read_arxiv_paper")
                if download and read:
                    self.paper_reader = PaperReaderTool(download_tool=download, read_tool=read)
                    self.registry.register_tool(self.paper_reader)
                self._tools_initialized = True
            else:
                for status in self.tool_statuses:
                    yield {"initialization_node": status}

    def run_research(self, request: ExecutionRequest, cancelled=lambda: False):
        yield {"initialization_node": {"status": "initializing", "message": "正在初始化研究工具…"}}
        if request.source_policy.permits_web and self._connect_external:
            yield from self._initialize_research_tools()
        contract = ResearchContract.from_request(
            request, max_rounds=self.config.max_rounds, max_steps=self.config.max_steps,
        )
        policy = contract.source_policy
        context = ResearchEvidenceContext(self.retrieval.evidence_store, policy, self.retrieval.documents, self.context_manager)
        durable = bool(self.store and self.checkpoints and request.run_id)
        budget = ResearchBudget(contract.limits.max_tool_calls, self.store if durable else None, request.run_id)
        observation_run = request.run_id or uuid4().hex
        if durable:
            for identity in self.store.evidence_ids(request.run_id):
                item = self.retrieval.evidence_store.get(identity)
                if item:
                    context.register(item)

        def make_tools(task_id, observed_ids, issues):
            def record(evidence):
                context.register(evidence)
                observed_ids.add(evidence.evidence_id)
                if durable:
                    self.store.register_evidence(request.run_id, evidence.evidence_id)

            def observe(observation):
                if observation.get("kind") == "error":
                    issues.append(f"{observation['tool']} 返回错误，来源不可访问")
                write_json(
                    Path(self.config.data_dir) / "observations" / observation_run / f"{uuid4().hex}.json",
                    {"task_id": task_id, **observation},
                )

            def wrap(tool):
                from sage_scholar.runtime.metrics import MeasuredTool
                if tool.source_access == "public":
                    tool = EvidenceTool(tool, self.retrieval.evidence_store, record, observe)
                tool = MeasuredTool(tool)
                tool = BudgetedTool(tool, budget)
                if durable:
                    def replay(result):
                        for identity in set(re.findall(r"ev_[a-zA-Z0-9_-]+", result)):
                            try:
                                record(context.get(identity))
                            except ValueError:
                                pass
                    tool = JournaledTool(tool, self.store, request.run_id, task_id, cancelled, replay)
                return tool

            tools = self.registry.for_policy(
                self.whitelist, policy,
                local_tools=[RAGTool(self.retrieval, policy, record)], wrap=wrap,
            )
            if self.context_manager:
                tools.append(ContextReadTool(self.context_manager, context, observation_run))
            return tools

        available = make_tools("capabilities", set(), [])
        yield {"policy_node": {
            "source_policy": policy.model_dump(mode="json"),
            "available_tools": [tool.name for tool in available],
        }}
        if not any(t.name != "read_context_artifact" for t in available) and not (durable and self.checkpoints.saver.get_tuple(self.checkpoints.config(request.run_id))):
            yield {"outcome_node": {"remaining_gaps": ["资料策略下没有可用工具，无法进行有证据的研究"]}}
            return

        yield {"initialization_node": {"status": "started", "message": "正在规划研究任务…"}}
        review_llm = (
            LLMClient(model=self.config.llm.review_model, timeout=self.config.llm.timeout, context_manager=self.context_manager)
            if self.config.llm.review_model else self.llm_client
        )
        supervisor = Supervisor(
            llm=self.llm_client,
            max_steps=contract.limits.max_steps, review_llm=review_llm,
        )
        writer = Writer(
            llm=self.llm_client, evidence_context=context,
            temperature=self.config.llm.writer_temperature,
        )

        def build_researcher(task_id, revision=0):
            observed_ids, issues = set(), []
            return Researcher(
                name=task_id, llm=self.llm_client,
                tool_list=make_tools(f"{task_id}:{revision}", observed_ids, issues),
                max_steps=contract.limits.max_steps, temperature=self.config.llm.research_temperature,
                evidence_context=context, observed_ids=lambda: sorted(observed_ids),
                observed_issues=lambda: list(issues),
            )

        def create_researcher(task_id):
            if durable:
                return DurableResearcher(lambda task: build_researcher(task.task_id, task.revision),
                                         self.store, request.run_id, cancelled)
            return build_researcher(task_id)

        graph = build_graph(
            supervisor, create_researcher, writer,
            gap_reviewer=GapReviewer(review_llm, context),
            synthesizer=ResearchSynthesis(self.llm_client, context),
            report_reviewer=ReportReviewer(review_llm, context), budget=budget,
            checkpointer=self.checkpoints.saver if durable else None,
            interaction=(lambda req: assess(req, self.llm_client)) if durable else None,
        )
        final_report = None
        graph_config = self.checkpoints.config(request.run_id) if durable else {}
        graph_config["recursion_limit"] = 250 + 3 * contract.limits.max_tasks * (contract.limits.max_rounds + 1)
        snapshot = graph.get_state(graph_config) if durable else None
        if durable and snapshot:
            snapshot = continue_graph(graph, graph_config, snapshot, request, contract.limits)
        graph_input = None if snapshot and snapshot.values else {"research_contract": contract}
        graph_input = resume_command(snapshot, request.metadata.get("_resume_input", {}), graph_input)
        if snapshot and snapshot.values and not snapshot.next:
            final_report = snapshot.values.get("report_result")
            if final_report:
                yield {"finalize_node": {"report_result": final_report}}
        else:
            for _mode, event in graph.stream(graph_input, config=graph_config,
                                            stream_mode=["updates", "custom"], **({"durability": "sync"} if durable else {})):
                if cancelled():
                    raise WorkflowInterrupted("研究已中断，checkpoint 已保留")
                if "finalize_node" in event:
                    final_report = event["finalize_node"]["report_result"]
                yield event
        yield {"evidence_node": {"evidence_ids": context.ids()}}
        yield {"outcome_node": {
            "remaining_gaps": final_report.remaining_gaps if final_report else ["研究未生成报告"],
            "tool_calls_used": budget.used,
            "blocked_by_budget": budget.blocked_by_budget,
            "denied_calls": budget.denied_calls,
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
            if node_name == "__interrupt__":
                events.append(("human_input", {"interrupts": public_interrupts(output)}))
            elif node_name == "initialization_node":
                event_type = {"initializing": "initializing_tools", "available": "tool_available",
                              "unavailable": "tool_unavailable", "started": "research_started"}[output["status"]]
                events.append((event_type, output))
            elif node_name == "progress_node":
                events.append((output["event"], {key: value for key, value in output.items() if key != "event"}))
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
            elif node_name in {"research_node", "section_sources_node"}:
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
                events.append(("report_drafting", {"phase": "section_completed", "section_index": output["section_index"]}))
            elif node_name == "report_review_node":
                events.append(("report_reviewing", {"phase": "section_completed"}))
            elif node_name == "revise_section_node":
                events.append(("report_revising", {"phase": "section_completed"}))
            elif node_name == "finalize_node":
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
        store=None, checkpoints=None, context_manager=None,
    ):
        self.store, self.checkpoints, self.context_manager = store, checkpoints, context_manager
        self._config = config
        self._llm_client_factory = llm_client_factory
        self._retrieval = retrieval
        self._runtimes: dict[bool, ResearchRuntime] = {}
        self._library_tools = {}
        self._library_clients = []
        self._library_readers = []
        self._lock = threading.Lock()

    def get(self, require_external: bool = True) -> ResearchRuntime:
        with self._lock:
            if require_external not in self._runtimes:
                self._runtimes[require_external] = ResearchRuntime(
                    self._config, llm_client=self._llm_client_factory(),
                    retrieval=self._retrieval, connect_external=require_external,
                    store=self.store, checkpoints=self.checkpoints, context_manager=self.context_manager,
                )
            return self._runtimes[require_external]

    def get_library_manager(self, capability: str) -> LibraryManager:
        """PDF/arXiv ingestion has no reason to instantiate a research LLM or web tools."""
        names = CAPABILITY_SERVERS[capability]
        with self._lock:
            if capability not in self._library_tools:
                clients = create_mcp_clients(
                    os.path.join(self._config.config_dir, "mcp_servers.json"),
                    server_names=names, connect_timeout=self._config.mcp_connect_timeout,
                )
                self._library_clients.extend(clients)
                registry = ToolRegistry()
                register_mcp_tools(registry, clients)
                pdf = registry.tools.get("mcp__pdfmux__convert_pdf")
                download = registry.tools.get("mcp__paper-search__download_arxiv")
                read = registry.tools.get("mcp__paper-search__read_arxiv_paper")
                paper = PaperReaderTool(download, read) if download and read else None
                if paper:
                    self._library_readers.append(paper)
                self._library_tools[capability] = (paper, pdf)
            paper, pdf = self._library_tools[capability]
        if (capability == "pdf" and pdf is None) or (capability == "paper" and paper is None):
            raise RuntimeError("PDF 解析工具不可用，请检查 pdfmux 配置后重试。" if capability == "pdf"
                               else "论文读取工具不可用，请检查 paper-search 配置后重试。")
        return LibraryManager(self._config.data_dir, paper, pdf, self._retrieval.pipeline)

    def close(self) -> None:
        for runtime in self._runtimes.values():
            runtime.close()
        for client in self._library_clients:
            client.disconnect()
        for reader in self._library_readers:
            reader.cleanup()
