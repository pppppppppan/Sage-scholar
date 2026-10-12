"""Single durable entry graph. Research retains its existing private graph/state."""
import asyncio
from concurrent.futures import Future, TimeoutError as FutureTimeout
import re
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from sage_scholar.context.manager import request_context
from sage_scholar.domain.execution import ExecutionRequest, RunOutcome
from sage_scholar.routing import RouteDecision, Workflow, ExecutionMode, SourceScope
from sage_scholar.runtime.control import CURRENT_CONTROL, check_control, cooperative_thread
from sage_scholar.runtime.human import HumanQuestion, RunWaiting, public_interrupts
from sage_scholar.runtime.interaction import resume_command
from sage_scholar.runtime.limits import execution_budget

ENTRY_VERSION = "semantic-query-v1"


class QueryState(TypedDict):
    execution: ExecutionRequest
    route: RouteDecision
    clarification_count: int
    outcome: RunOutcome
    continuation_generation: int
    entry_version: str


class QueryGraphRunner:
    def __init__(self, router, direct, qa, research, *, documents=None, memory=None, checkpoints=None):
        self.router, self.direct, self.qa, self.research = router, direct, qa, research
        self.documents, self.memory, self.checkpoints = documents, memory, checkpoints

    async def run(self, request, emit, cancelled):
        loop = asyncio.get_running_loop()

        def wait(coro):
            # Keep service I/O on the application loop. On Stop, wait for cancellation cleanup
            # before allowing the graph thread to exit or its SQLite saver to be closed.
            result = Future()
            tasks = []
            def finished(task):
                try:
                    result.set_result(task.result())
                except BaseException as exc:
                    result.set_exception(exc)
            def start():
                task = loop.create_task(coro)
                tasks.append(task)
                task.add_done_callback(finished)
            loop.call_soon_threadsafe(start)
            try:
                while True:
                    try:
                        return result.result(timeout=0.1)
                    except FutureTimeout:
                        if result.done():
                            raise  # The service itself timed out; do not spin on its exception.
                        check_control()
                        if cancelled():
                            from sage_scholar.runtime.journal import WorkflowInterrupted
                            raise WorkflowInterrupted("用户已停止任务")
            except BaseException:
                # FIFO callbacks ensure start runs before this cancellation callback.
                loop.call_soon_threadsafe(lambda: [task.cancel() for task in tasks if not task.done()])
                try:
                    result.result()
                except BaseException:
                    pass
                raise

        return await cooperative_thread(self._execute, request, emit, cancelled, wait)

    def _execute(self, request, emit, cancelled, wait):
        durable = bool(self.checkpoints and request.run_id)

        def current(state):
            check_control()
            old = state["execution"]
            # Resume updates limits and lifecycle metadata, never reroutes an already chosen branch.
            budget = request.budget if request.workflow == old.workflow and request.workflow not in {Workflow.ROUTER, Workflow.CLARIFY} else old.budget
            metadata = {**request.metadata, **old.metadata, "continuation": request.metadata.get("continuation", 0)}
            if state.get("clarification_count"):
                metadata.pop("human_question", None)
            return old.model_copy(update={"budget": budget, "metadata": metadata})

        def activate(execution, route):
            control = CURRENT_CONTROL.get()
            if control:
                control.request = execution
            wait(emit("route_selected", {"route": route.model_dump(mode="json"),
                                         "execution": execution.model_dump(mode="json")}))

        def router_node(state):
            execution = current(state)
            wait(emit("routing", {"message": "正在使用本地模型理解问题…"}))
            options = execution.metadata.get("_routing", {})
            selected = list(options.get("document_ids", execution.document_ids))
            catalog = wait(self.documents.catalog()) if self.documents else []
            resolution = None
            if self.documents and selected:
                # Only explicit UI/API selection is a hard boundary. A title mentioned in
                # text (possibly negated) is classified against the catalog by the model.
                resolved = wait(self.documents.resolve(execution.query, selected))
                resolution = resolved.message
            # An explicit save command is an operation, not a keyword-based intent classifier.
            remember = re.fullmatch(r"(?:请)?记住[：:]\s*(.+)", execution.query.strip(), re.S)
            if remember:
                route = RouteDecision(workflow=Workflow.DIRECT, source_scope=SourceScope.CONVERSATION,
                    allow_web=False, execution_mode=ExecutionMode.RUN, reason_code="explicit_memory_operation")
            else:
                route = self.router.route(execution.query,
                    requested_workflow=options.get("workflow"), requested_scope=options.get("source_scope"),
                    document_ids=selected, allow_web=options.get("allow_web"), history=execution.history,
                    catalog=catalog, resolution_message=resolution)
            if self.documents and route.document_ids and not route.needs_clarification:
                checked = wait(self.documents.resolve(execution.query, route.document_ids))
                if checked.message:
                    route = route.model_copy(update={"workflow": Workflow.CLARIFY, "needs_clarification": True,
                                                      "clarification_message": checked.message})
            explicit_question = execution.metadata.get("human_question")
            if explicit_question and not state.get("clarification_count"):
                question = HumanQuestion.model_validate(explicit_question)
                route = route.model_copy(update={"workflow": Workflow.CLARIFY, "needs_clarification": True,
                                                  "clarification_message": question.question})
            limits = execution_budget(route.workflow, options.get("budget", {}))
            routed = execution.model_copy(update={**route.model_dump(include={
                "workflow", "source_scope", "allow_web", "execution_mode"}),
                "document_ids": tuple(route.document_ids), "budget": limits})
            return {"route": route, "execution": routed, "entry_version": ENTRY_VERSION}

        def clarify_node(state):
            execution = current(state)
            activate(execution, state["route"])
            if state.get("clarification_count", 0) >= 3:
                raise ValueError("连续澄清后仍无法确定任务，请停止后明确提供问题与资料范围")
            explicit = execution.metadata.get("human_question")
            question = HumanQuestion.model_validate(explicit) if explicit and not state.get("clarification_count") else HumanQuestion(
                question=state["route"].clarification_message or "请补充具体问题和资料范围。")
            answer = interrupt(question.model_dump())
            text = answer.get("answer", "").strip()
            if answer.get("skip"):
                text = "请按合理默认范围自主决定。"
            if not text:
                raise ValueError("澄清回答不能为空")
            metadata = {**execution.metadata, "constraints": [*execution.metadata.get("constraints", []), text]}
            metadata.pop("human_question", None)
            updated = execution.model_copy(update={"query": execution.query + "\n用户补充：" + text,
                                                    "metadata": metadata})
            return {"execution": updated, "clarification_count": state.get("clarification_count", 0) + 1}

        def direct_node(state):
            execution = current(state)
            activate(execution, state["route"])
            with request_context(execution):
                explicit = re.fullmatch(r"(?:请)?记住[：:]\s*(.+)", execution.query.strip(), re.S)
                if explicit and self.memory:
                    record = self.memory.remember(explicit[1], workspace_id=execution.workspace_id,
                                                  session_id=execution.session_id or execution.conversation_id)
                    answer = f"已记住：{record.content}。可在记忆管理中修改或删除。"
                else:
                    answer = wait(self.direct.answer(execution))
                check_control()
            return {"outcome": RunOutcome(result={"answer": answer}), "execution": execution}

        def qa_node(state):
            execution = current(state)
            activate(execution, state["route"])
            with request_context(execution):
                value = wait(self.qa.answer(execution))
                check_control()
            return {"outcome": RunOutcome(result={"answer": value.answer,
                "evidence": [e.model_dump(mode="json") for e in value.evidence], "citations": value.citations,
                "citation_issues": value.citation_issues, "reading_strategy": value.reading_strategy}), "execution": execution}

        async def research_emit(kind, data):
            # Parent interrupt IDs are authoritative at the HTTP boundary.
            if kind != "human_input":
                await emit(kind, data)

        def deep_research_node(state):
            execution = current(state)
            activate(execution, state["route"])
            if not self.research:
                raise ValueError("Deep Research runner 未配置")
            metadata = dict(execution.metadata)
            metadata.pop("_resume_input", None)
            child_request = execution.model_copy(update={"metadata": metadata})
            # The existing Research graph has private state/checkpoints, addressed by the same Run.
            # A completed child result is replayed on parent recovery rather than regenerating a report.
            with request_context(child_request):
                try:
                    outcome = wait(self.research.run(child_request, research_emit, cancelled))
                except RunWaiting as waiting:
                    answers = {}
                    for question in waiting.interrupts:
                        answer = interrupt({k: v for k, v in question.items() if k != "interrupt_id"})
                        answers[question["interrupt_id"]] = answer
                    child_request = child_request.model_copy(update={"metadata": {**metadata, "_resume_input": answers}})
                    outcome = wait(self.research.run(child_request, research_emit, cancelled))
            return {"outcome": outcome, "execution": execution}

        builder = StateGraph(QueryState)
        for name, node in (("router", router_node), ("clarify", clarify_node), ("direct", direct_node),
                           ("qa", qa_node), ("deep_research", deep_research_node)):
            builder.add_node(name, node)
        builder.add_node("continue", lambda state: {})
        builder.add_edge(START, "router")
        branches = {name: name for name in ("direct", "qa", "clarify", "deep_research")}
        builder.add_conditional_edges("router", lambda s: s["route"].workflow.value, branches)
        builder.add_conditional_edges("continue", lambda s: s["route"].workflow.value, branches)
        builder.add_edge("clarify", "router")
        for name in ("direct", "qa", "deep_research"):
            builder.add_edge(name, END)
        graph = builder.compile(checkpointer=self.checkpoints.saver if durable else None)
        config = self.checkpoints.config(request.run_id) if durable else {"recursion_limit": 40}
        snapshot = graph.get_state(config) if durable else None
        if snapshot and snapshot.values and not snapshot.next:
            generation = int(request.metadata.get("continuation", 0))
            if generation > snapshot.values.get("continuation_generation", 0):
                graph.update_state(config, {"continuation_generation": generation}, as_node="continue")
                snapshot = graph.get_state(config)
        initial = None if snapshot and snapshot.values else {"execution": request, "entry_version": ENTRY_VERSION}
        initial = resume_command(snapshot, request.metadata.get("_resume_input", {}), initial)
        result = snapshot.values.get("outcome") if snapshot and not snapshot.next else None
        for update in graph.stream(initial, config, stream_mode="updates", **({"durability": "sync"} if durable else {})):
            check_control()
            if "__interrupt__" in update:
                raise RunWaiting(public_interrupts(update["__interrupt__"]))
            for value in update.values():
                if isinstance(value, dict) and value.get("outcome"):
                    result = value["outcome"]
        if result is None:
            raise RuntimeError("Query Graph 未产生结果")
        if "answer" in result.result:
            wait(emit("answer", result.result))
        return result
