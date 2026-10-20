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
from sage_scholar.runtime.limits import execution_budget
from sage_scholar.infrastructure.json_store import write_json
from .budget import ResearchBudget, BudgetedTool
from .progress import ResearchProgress
from .evidence_context import ResearchEvidenceContext
from sage_scholar.infrastructure.retrieval import LegacyRetrievalAdapter
from sage_scholar.tools.policy import EvidenceTool
from sage_scholar.graph import build_graph
from sage_scholar.library import LibraryManager
from sage_scholar.mcp.lifecycle import MCPLifecycle
from sage_scholar.mcp.loader import CAPABILITY_SERVERS, RESEARCH_SERVERS, iter_mcp_clients
from sage_scholar.rag import Pipeline
from sage_scholar.search import SearchTool
from sage_scholar.tools import PaperReaderTool, RAGTool

logger = logging.getLogger(__name__)


class ResearchRuntime:
    """Composition root for migrated SAGE Research capabilities."""

    def __init__(self, config: Config, llm_client: LLMClient | None = None, pipeline: Pipeline | None = None, retrieval=None, connect_external: bool = True, store=None, checkpoints=None, context_manager=None, mcp_lifecycle=None):
        self.config = config
        self.store, self.checkpoints, self.context_manager = store, checkpoints, context_manager
        self.llm_client = llm_client or LLMClient(model=config.llm.model, timeout=config.llm.timeout, context_manager=context_manager)
        self.retrieval = retrieval or LegacyRetrievalAdapter(config, pipeline=pipeline)
        from sage_scholar.infrastructure.model_gateway import LegacyModelGateway
        self.visual_gateway = LegacyModelGateway(config, context_manager)
        self._owns_mcp = mcp_lifecycle is None
        self._mcp = mcp_lifecycle or MCPLifecycle(os.path.join(config.config_dir, "mcp_servers.json"), config.mcp_connect_timeout)
        self.mcp_clients = self._mcp.clients
        self.registry = self._mcp.registry
        self.skills = getattr(context_manager, "skills", None)
        self.paper_reader = self.pdfmux_tool = None
        self._connect_external = connect_external
        self._tools_lock = threading.Lock()
        self.tool_statuses = []

        with open(os.path.join(config.config_dir, "agents.json"), encoding="utf-8") as file:
            agent_configs = json.load(file)
        self.whitelist = agent_configs["researcher"]["allowed_tools"]

    def _initialize_research_tools(self):
        with self._tools_lock:
            self.tool_statuses = self._mcp.ensure(RESEARCH_SERVERS, iter_mcp_clients)
            brave = self.registry.tools.get("mcp__brave-search__brave_web_search")
            tavily = self.registry.tools.get("mcp__tavily__tavily_search")
            dependencies = (brave, tavily)
            if any(dependencies) and dependencies != getattr(self, "_search_dependencies", None):
                self.registry.tools["search"] = SearchTool(brave_tool=brave, tavily_tool=tavily)
                self._search_dependencies = dependencies
            if not any(dependencies):
                self.registry.tools.pop("search", None)
                self._search_dependencies = None
            download = self.registry.tools.get("mcp__paper-search__download_arxiv")
            read = self.registry.tools.get("mcp__paper-search__read_arxiv_paper")
            if download and read and (download, read) != getattr(self, "_paper_dependencies", None):
                if self.paper_reader:
                    self.paper_reader.cleanup()
                self._paper_dependencies = (download, read)
                self.paper_reader = PaperReaderTool(download_tool=download, read_tool=read)
                self.registry.tools[self.paper_reader.name] = self.paper_reader
            if not (download and read) and self.paper_reader:
                self.registry.tools.pop(self.paper_reader.name, None)
                self.paper_reader.cleanup()
                self.paper_reader = None
                self._paper_dependencies = None
            for status in self.tool_statuses:
                yield {"initialization_node": status}

    def run_research(self, request: ExecutionRequest, cancelled=lambda: False):
        yield {"initialization_node": {"status": "initializing", "message": "正在初始化研究工具…"}}
        if request.source_policy.permits_web and self._connect_external:
            yield from self._initialize_research_tools()
        request = request.model_copy(update={"budget": execution_budget(request.workflow, request.budget)})
        policy = request.source_policy
        if self.skills:
            self.skills.explicit(request)
        context = ResearchEvidenceContext(self.retrieval.evidence_store, policy, self.retrieval.documents, self.context_manager)
        durable = bool(self.store and self.checkpoints and request.run_id)
        budget = ResearchBudget(request.budget["max_tool_calls"], self.store if durable else None, request.run_id)
        observation_run = request.run_id or uuid4().hex
        if durable:
            for identity in self.store.evidence_ids(request.run_id):
                item = self.retrieval.evidence_store.get(identity)
                if item:
                    context.register(item)

        def make_tools(task_id, observed_ids):
            def record(evidence):
                context.register(evidence)
                observed_ids.add(evidence.evidence_id)
                if durable:
                    self.store.register_evidence(request.run_id, evidence.evidence_id)

            def observe(observation):
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

            from sage_scholar.tools.tool_pdf_visual import PDFVisualTool
            local_tools = [RAGTool(self.retrieval, policy, record)]
            if getattr(self.retrieval, "visual", None) and self.retrieval.visual.config.enabled:
                local_tools.append(PDFVisualTool(self.retrieval.visual, context, policy, self.visual_gateway, record))
            tools = self.registry.for_policy(
                self.whitelist, policy,
                local_tools=local_tools, wrap=wrap,
            )
            if self.skills:
                tools.extend(wrap(tool) for tool in self.skills.tools(request))
            from sage_scholar.tools.tool_conversation import ConversationReadTool
            tools.append(wrap(ConversationReadTool(request, self.store, context, record)))
            if policy.permits_local:
                from sage_scholar.tools.tool_sections import DocumentSectionsTool
                tools.append(wrap(DocumentSectionsTool(self.retrieval, policy, record)))
            if self.context_manager:
                tools.append(ContextReadTool(self.context_manager, context, observation_run))
            return tools

        available = make_tools("capabilities", set())
        yield {"policy_node": {
            "source_policy": policy.model_dump(mode="json"),
            "available_tools": [tool.name for tool in available],
        }}
        if not any(t.source_access in {"local", "public"} for t in available) and not (durable and self.checkpoints.saver.get_tuple(self.checkpoints.research_config(request))):
            yield {"outcome_node": {"remaining_gaps": ["资料策略下没有可用工具，无法进行有证据的研究"]}}
            return

        yield {"initialization_node": {"status": "started", "message": "正在规划研究任务…"}}
        review_llm = (
            LLMClient(model=self.config.llm.review_model, timeout=self.config.llm.timeout, context_manager=self.context_manager)
            if self.config.llm.review_model else self.llm_client
        )
        supervisor = Supervisor(
            llm=self.llm_client,
            max_steps=request.budget["max_steps"], review_llm=review_llm, skills=self.skills,
        )
        writer = Writer(
            llm=self.llm_client, evidence_context=context,
            temperature=self.config.llm.writer_temperature, max_output_tokens=self.config.generation.writer_tokens, skills=self.skills,
        )

        def build_researcher(task_id, revision=0):
            observed_ids = set()
            return Researcher(
                name=task_id, llm=self.llm_client,
                tool_list=make_tools(f"{task_id}:{revision}", observed_ids),
                max_steps=request.budget["max_steps"], temperature=self.config.llm.research_temperature,
                evidence_context=context, observed_ids=lambda: sorted(observed_ids), generation=self.config.generation,
                progress=ResearchProgress(self.store, request.run_id, task_id, revision) if durable else None,
            )

        def create_researcher(task_id):
            if durable:
                return DurableResearcher(lambda task: build_researcher(task.task_id, task.revision),
                                         self.store, request.run_id, cancelled)
            return build_researcher(task_id)

        graph = build_graph(
            supervisor, create_researcher, writer,
            budget=budget, execution_request=request,
            checkpointer=self.checkpoints.saver if durable else None,
            interaction=(lambda req: assess(req, self.llm_client)) if durable else None,
        )
        final_report = None
        graph_config = self.checkpoints.research_config(request) if durable else {}
        graph_config["recursion_limit"] = 250 + 3 * request.budget["max_tasks"] * (request.budget["max_rounds"] + 1)
        snapshot = graph.get_state(graph_config) if durable else None
        if durable and snapshot:
            snapshot = continue_graph(graph, graph_config, snapshot, request)
        graph_input = None if snapshot and snapshot.values else {"execution": request}
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
            visual=self.retrieval.visual,
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
                if output.get("tasks"):
                    events.append(("research_planned", {
                        "tasks": [task.model_dump() for task in output["tasks"].values()],
                    }))
                for identity in output["ready_tasks"]:
                    events.append(("research_task_started", {"task_id": identity}))
            elif node_name == "research_node":
                for note in output["research_notes"].values():
                    events.append(("research", {
                        "question": note.question, "preview": note.content[:150],
                        "tool_call_counts": {},
                        "status": output["tasks"][note.task_id].status,
                    }))
                    events.append(("research_task_completed", {"note": note.model_dump()}))
            elif node_name == "review_node":
                review = output["supervisor_review"]
                events.append(("review", {"round": output["review_round"], "review_summary": [],
                    "missing_dimensions": "; ".join(review.remaining_gaps)}))
                events.append(("supervisor_reviewed", review.model_dump()))
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
        if self._owns_mcp:
            self._mcp.close()


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
        self._mcp = MCPLifecycle(os.path.join(config.config_dir, "mcp_servers.json"), config.mcp_connect_timeout)
        self._library_clients = self._mcp.clients
        self._library_readers = []
        self._closed = False
        self._lock = threading.Lock()

    def get(self, require_external: bool = True) -> ResearchRuntime:
        with self._lock:
            if self._closed:
                raise RuntimeError("MCP provider is closed")
            if require_external not in self._runtimes:
                self._runtimes[require_external] = ResearchRuntime(
                    self._config, llm_client=self._llm_client_factory(),
                    retrieval=self._retrieval, connect_external=require_external,
                    store=self.store, checkpoints=self.checkpoints, context_manager=self.context_manager, mcp_lifecycle=self._mcp,
                )
            return self._runtimes[require_external]

    def get_library_manager(self, capability: str) -> LibraryManager:
        """PDF/arXiv ingestion has no reason to instantiate a research LLM or web tools."""
        names = CAPABILITY_SERVERS[capability]
        with self._lock:
            if self._closed:
                raise RuntimeError("MCP provider is closed")
            self._mcp.ensure(names, iter_mcp_clients)
            registry = self._mcp.registry
            pdf = registry.tools.get("mcp__pdfmux__convert_pdf")
            download = registry.tools.get("mcp__paper-search__download_arxiv")
            read = registry.tools.get("mcp__paper-search__read_arxiv_paper")
            paper = self._library_tools.get("paper") if download and read else None
            if download and read and (download, read) != self._library_tools.get("paper_dependencies"):
                paper = PaperReaderTool(download, read)
                self._library_readers.append(paper)
                self._library_tools["paper"] = paper
                self._library_tools["paper_dependencies"] = (download, read)
        if (capability == "pdf" and pdf is None) or (capability == "paper" and paper is None):
            raise RuntimeError("PDF 解析工具不可用，请检查 pdfmux 配置后重试。" if capability == "pdf"
                               else "论文读取工具不可用，请检查 paper-search 配置后重试。")
        return LibraryManager(self._config.data_dir, paper, pdf, self._retrieval.pipeline, self._retrieval.visual)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for runtime in self._runtimes.values():
                runtime.close()
            self._mcp.close()
            for reader in self._library_readers:
                reader.cleanup()
